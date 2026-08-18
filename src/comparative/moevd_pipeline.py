import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from moevd import MoEVDExpert, MoEVDRouter, focal_loss
from seed_utils import set_seed, worker_init_fn, make_generator
from logger import setup_logging, get_logger

CLASSES = ['cwe-022', 'cwe-078', 'cwe-079', 'cwe-089']
CWE2CLS = {c: i for i, c in enumerate(CLASSES)}

def build_mapping(mode):
    if mode == 'group2':
        classes = ['group-s', 'group-w']
        cwe2cls = {'cwe-022': 0, 'cwe-078': 0, 'cwe-079': 1, 'cwe-089': 1}
    else:
        classes = ['cwe-022', 'cwe-078', 'cwe-079', 'cwe-089']
        cwe2cls = {c: i for i, c in enumerate(classes)}
    return classes, cwe2cls

def load_jsonl(path):
    return pd.DataFrame([json.loads(l) for l in open(path)])

class CodeDS(Dataset):

    def __init__(self, df, tok, target, max_len=512):
        self.df = df.reset_index(drop=True); self.tok = tok; self.target = target; self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        enc = self.tok(r['code'], truncation=True, padding='max_length',
                       max_length=self.max_len, return_tensors='pt')
        idx = CWE2CLS.get(str(r['original_CWE_ID']).lower().strip(), -1)
        if self.target == 'router':
            y = idx
        elif self.target.startswith('expert'):
            c = int(self.target[6:])
            y = 1 if (int(r['label']) == 1 and idx == c) else 0
        else:
            y = int(r['label'])
        return (enc['input_ids'].flatten(), enc['attention_mask'].flatten(),
                torch.tensor(y, dtype=torch.long))

def loader(df, tok, target, bs, shuffle, seed):
    return DataLoader(CodeDS(df, tok, target), batch_size=bs, shuffle=shuffle,
                      num_workers=2, pin_memory=True,
                      worker_init_fn=worker_init_fn, generator=make_generator(seed))

def train_expert(backbone, df_tr, df_va, tok, dev, c, args, logger, seed):
    model = MoEVDExpert(backbone, num_labels=1).to(dev)
    cls_idx = df_tr['original_CWE_ID'].str.lower().str.strip().map(CWE2CLS)
    npos = int(((df_tr['label'] == 1) & (cls_idx == c)).sum())
    nneg = len(df_tr) - npos
    pw = torch.tensor([nneg / max(npos, 1)], device=dev)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = AdamW(model.parameters(), lr=args.lr)
    tl = loader(df_tr, tok, f'expert{c}', args.batch_expert, True, seed)
    vl = loader(df_va, tok, f'expert{c}', args.batch_expert, False, seed)
    logger.info(f"[expert {c}={CLASSES[c]}] pos={npos} neg={nneg} pos_weight={pw.item():.2f}")
    best_f1, best_state, patience = -1.0, None, 0
    for ep in range(args.expert_epochs):
        model.train()
        for ids, m, y in tl:
            opt.zero_grad()
            z = model(ids.to(dev), m.to(dev)).squeeze(-1)
            crit(z, y.float().to(dev)).backward(); opt.step()
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for ids, m, y in vl:
                p = torch.sigmoid(model(ids.to(dev), m.to(dev)).squeeze(-1))
                P += (p >= 0.5).int().cpu().tolist(); Y += y.tolist()
        f1 = f1_score(Y, P, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, patience = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                logger.info(f"[expert {c}] early stop ep{ep+1}"); break
    model.load_state_dict(best_state); model.eval()
    logger.info(f"[expert {c}] best val macroF1={best_f1:.4f}")
    return model

def train_router(backbone, df_tr, df_va, tok, dev, args, logger, seed):
    model = MoEVDRouter(backbone, num_experts=len(CLASSES)).to(dev)
    tr = df_tr[df_tr['label'] == 1]; va = df_va[df_va['label'] == 1]
    opt = AdamW(model.parameters(), lr=args.lr)
    tl = loader(tr, tok, 'router', args.batch_router, True, seed)
    vl = loader(va, tok, 'router', args.batch_router, False, seed)
    logger.info(f"[router] train vul={len(tr)} val vul={len(va)}")
    best_f1, best_state, patience = -1.0, None, 0
    for ep in range(args.router_epochs):
        model.train()
        for ids, m, y in tl:
            opt.zero_grad()
            focal_loss(model(ids.to(dev), m.to(dev)), y.to(dev), gamma=args.focal_gamma).backward()
            opt.step()
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for ids, m, y in vl:
                P += torch.argmax(model(ids.to(dev), m.to(dev)), 1).cpu().tolist(); Y += y.tolist()
        f1 = f1_score(Y, P, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, patience = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                logger.info(f"[router] early stop ep{ep+1}"); break
    model.load_state_dict(best_state); model.eval()
    logger.info(f"[router] best val macroF1={best_f1:.4f}")
    return model

@torch.no_grad()
def combiner_probs(router, experts, df, tok, dev, top_k, bs=32):

    dl = DataLoader(CodeDS(df, tok, 'raw'), batch_size=bs, shuffle=False, num_workers=2)
    out = []
    for ids, m, _ in dl:
        ids, m = ids.to(dev), m.to(dev)
        cwe_prob = torch.softmax(router(ids, m), dim=-1)
        tk_p, tk_i = torch.topk(cwe_prob, top_k, dim=1)
        tk_w = torch.softmax(tk_p, dim=-1)
        ep = torch.stack([torch.sigmoid(e(ids, m).squeeze(-1)) for e in experts], dim=1)
        gathered = torch.gather(ep, 1, tk_i)
        out.append((tk_w * gathered).sum(dim=1).cpu())
    return torch.cat(out).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--extra_train_files', nargs='*', default=[],
                    help='additional jsonl files concatenated into the TRAIN set '
                         '(e.g. in-distribution augmentation) — fairness control for MoEVD+enh')
    ap.add_argument('--pretrained_model', default='microsoft/codebert-base')
    ap.add_argument('--batch_expert', type=int, default=12)
    ap.add_argument('--batch_router', type=int, default=16)
    ap.add_argument('--expert_epochs', type=int, default=25)
    ap.add_argument('--router_epochs', type=int, default=15)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--focal_gamma', type=float, default=2.0)
    ap.add_argument('--top_k', type=int, default=2)
    ap.add_argument('--mode', choices=['cwe4', 'group2'], default='cwe4',
                    help='cwe4 = one expert per CWE (faithful MoEVD); '
                         'group2 = one expert per S/W CWE-group (this repo grouping)')
    args = ap.parse_args()

    global CLASSES, CWE2CLS
    CLASSES, CWE2CLS = build_mapping(args.mode)
    if args.top_k > len(CLASSES):
        args.top_k = len(CLASSES)

    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(log_file=os.path.join(args.out_dir, 'moevd.log'), reset_file=True)
    logger = get_logger(__name__)
    set_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"MoEVD[{args.mode}] fold{args.fold} seed{args.seed} dev={dev} | "
                f"classes={CLASSES} | expert(ep{args.expert_epochs},bs{args.batch_expert}) "
                f"router(ep{args.router_epochs},bs{args.batch_router},gamma{args.focal_gamma}) top_k={args.top_k}")

    tok = AutoTokenizer.from_pretrained(args.pretrained_model)
    fd = f"{args.data_dir}/fold{args.fold}"
    df_tr = load_jsonl(f"{fd}/train.jsonl")
    df_va, df_te = load_jsonl(f"{fd}/val.jsonl"), load_jsonl(f"{fd}/test.jsonl")
    for ef in args.extra_train_files:
        n0 = len(df_tr); df_tr = pd.concat([df_tr, load_jsonl(ef)], ignore_index=True)
        logger.info(f"[data] +extra train {ef}: {n0} -> {len(df_tr)} rows")

    experts = []
    for c in range(len(CLASSES)):
        set_seed(args.seed + c)
        bb = AutoModel.from_pretrained(args.pretrained_model)
        experts.append(train_expert(bb, df_tr, df_va, tok, dev, c, args, logger, args.seed + c))
    set_seed(args.seed + 100)
    router = train_router(AutoModel.from_pretrained(args.pretrained_model),
                          df_tr, df_va, tok, dev, args, logger, args.seed + 100)

    for split, df in [('test', df_te), ('val', df_va)]:
        vp = combiner_probs(router, experts, df, tok, dev, args.top_k)
        pd.DataFrame({'label': df['label'].astype(int).values,
                      'original_CWE_ID': df['original_CWE_ID'].values,
                      'vul_prob': vp}).to_csv(f"{args.out_dir}/{split}_probs.csv", index=False)
        f1 = f1_score(df['label'].astype(int).values, (vp >= 0.5).astype(int), average='macro', zero_division=0)
        logger.info(f"[combiner/{split}] n={len(df)} macroF1@0.5={f1:.4f} -> {args.out_dir}/{split}_probs.csv")
    logger.info("MoEVD pipeline done.")

if __name__ == '__main__':
    main()
