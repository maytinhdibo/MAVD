import argparse, json, os, sys
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from seed_utils import set_seed
from logger import setup_logging, get_logger

class CodeBERTClassifier(nn.Module):

    def __init__(self, pretrained, num_labels=2, dropout=0.1, pad_id=1, class_weight=None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_labels)
        self.pad_id = pad_id
        self.register_buffer('cw', class_weight if class_weight is not None else None)

    def forward(self, input_ids, labels=None):
        am = (input_ids != self.pad_id).long()
        pooled = self.backbone(input_ids=input_ids, attention_mask=am).pooler_output
        logits = self.classifier(self.dropout(pooled))
        if labels is not None:
            loss = F.cross_entropy(logits, labels, weight=self.cw)
            return loss, logits
        return logits

def compute_difficulties(model, batches, return_batch=False):

    model.eval(); out = []
    with torch.no_grad():
        for ids, labels in batches:
            logits = model(ids)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            conf = torch.abs(probs[:, 0] - probs[:, 1])
            correct = (preds == labels).float()
            d = ((1 - conf) * correct + (1 + conf) * (1 - correct)) / 2.0
            avg = torch.mean(d).item()
            out.append(((ids, labels), avg) if return_batch else avg)
    return out

def compute_local_std(diffs, lambda_, local_ratio=0.2):
    diffs = sorted(diffs); n = len(diffs)
    k = max(1, int(n * local_ratio))
    c = min(range(n), key=lambda i: abs(diffs[i] - lambda_))
    s = max(0, c - k // 2); e = min(n, s + k)
    loc = diffs[s:e]
    return torch.std(torch.tensor(loc)).item() if len(loc) > 1 else 0.0

@torch.no_grad()
def predict_probs(model, ids_tensor, bs=32):
    model.eval(); probs = []
    for i in range(0, ids_tensor.size(0), bs):
        logits = model(ids_tensor[i:i+bs])
        probs += torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
    return np.array(probs)

def evaluate_f1(model, val_ids, val_y):
    p = predict_probs(model, val_ids)
    return f1_score(val_y, (p >= 0.5).astype(int), average='macro', zero_division=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_file', required=True)
    ap.add_argument('--val_file', required=True)
    ap.add_argument('--test_file', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--extra_train_files', nargs='*', default=[],
                    help='additional jsonl files concatenated into the TRAIN set '
                         '(e.g. in-distribution augmentation) — fairness control for SPLVD+enh')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pretrained_model', default='microsoft/codebert-base')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--max_len', type=int, default=512)
    ap.add_argument('--patience', type=int, default=10)

    ap.add_argument('--gamma', type=float, default=0.025)
    ap.add_argument('--alpha', type=float, default=0.3)
    ap.add_argument('--max_increase_ratio', type=float, default=0.1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    setup_logging(log_file=os.path.join(args.out_dir, 'splvd.log'), reset_file=True)
    logger = get_logger(__name__)
    set_seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    tok = AutoTokenizer.from_pretrained(args.pretrained_model)
    def load(p):
        df = pd.DataFrame([json.loads(l) for l in open(p)])
        return df
    df_tr, df_va, df_te = load(args.train_file), load(args.val_file), load(args.test_file)
    for ef in args.extra_train_files:
        n0 = len(df_tr); df_tr = pd.concat([df_tr, load(ef)], ignore_index=True)
        logger.info(f"[data] +extra train {ef}: {n0} -> {len(df_tr)} rows")

    def enc(df):
        e = tok(list(df['code']), truncation=True, padding='max_length',
                max_length=args.max_len, return_tensors='pt')
        return e['input_ids'].to(dev)
    tr_ids = enc(df_tr); tr_y = torch.tensor(df_tr['label'].astype(int).values).to(dev)
    va_ids = enc(df_va); va_y = df_va['label'].astype(int).values
    te_ids = enc(df_te)

    cnt = np.bincount(df_tr['label'].astype(int).values, minlength=2).astype(float)
    cw = torch.tensor(np.clip(cnt.sum() / (2 * np.maximum(cnt, 1)), 0, 3.0), dtype=torch.float).to(dev)
    logger.info(f"SPLVD fold seed{args.seed} | train={len(df_tr)} class={cnt.tolist()} cw={cw.tolist()} "
                f"val={len(df_va)} test={len(df_te)} | DynSPL gamma={args.gamma} alpha={args.alpha}")

    model = CodeBERTClassifier(args.pretrained_model, num_labels=2, class_weight=cw).to(dev)
    opt = AdamW(model.parameters(), lr=args.lr)
    loader = DataLoader(TensorDataset(tr_ids, tr_y), batch_size=args.batch_size, shuffle=True)

    init_batches = [(b[0], b[1]) for b in loader]
    diffs0 = compute_difficulties(model, init_batches)
    diffs0.sort()
    lambda_ = diffs0[int(0.1 * len(diffs0))] if diffs0 else 0.5
    logger.info(f"Initial lambda={lambda_:.4f}")

    best_f1, best_state, no_imp, prev_avg = -1.0, None, 0, None
    for ep in range(args.epochs):

        batches = [(b[0], b[1]) for b in DataLoader(TensorDataset(tr_ids, tr_y),
                   batch_size=args.batch_size, shuffle=True)]
        diff_b = compute_difficulties(model, batches, return_batch=True)
        sd = [d for _, d in diff_b]
        local_std = compute_local_std(sd, lambda_)
        cur_ratio = sum(1 for _, d in diff_b if d <= lambda_) / max(1, len(diff_b))
        adaptive_gamma = args.gamma * (1 + args.alpha * (1 - cur_ratio))
        avg_d = float(np.mean(sd)) if sd else 0.5
        diff_delta = 0.0 if prev_avg is None else avg_d - prev_avg
        stability = 1.0 / (1.0 + local_std * 10)
        new_lambda = lambda_ + adaptive_gamma * (1 - avg_d) * stability + (diff_delta if diff_delta < 0 else 0)
        new_lambda = min(1.0, max(0.0, new_lambda))

        new_ratio = sum(1 for _, d in diff_b if d <= new_lambda) / max(1, len(diff_b))
        if new_ratio - cur_ratio > args.max_increase_ratio:
            tgt = int((cur_ratio + args.max_increase_ratio) * len(diff_b))
            ss = sorted(sd)
            new_lambda = ss[min(max(tgt - 1, 0), len(ss) - 1)]
        prev_avg = avg_d
        lambda_ = new_lambda

        selected = [b for b, d in diff_b if d <= lambda_]
        model.train(); total = 0.0
        for ids, labels in selected:
            opt.zero_grad()
            loss, _ = model(ids, labels=labels)
            loss.backward(); opt.step(); total += loss.item()
        f1 = evaluate_f1(model, va_ids, va_y)
        logger.info(f"ep{ep+1}/{args.epochs} lambda={lambda_:.4f} sel={len(selected)}/{len(diff_b)} "
                    f"avg_d={avg_d:.4f} trainloss={total/max(1,len(selected)):.4f} valF1={f1:.4f}")
        if f1 > best_f1:
            best_f1, no_imp = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= args.patience:
                logger.info(f"early stop ep{ep+1}"); break

    model.load_state_dict(best_state)
    logger.info(f"best val macroF1={best_f1:.4f}")
    for split, ids, df in [('test', te_ids, df_te), ('val', va_ids, df_va)]:
        vp = predict_probs(model, ids)
        pd.DataFrame({'label': df['label'].astype(int).values,
                      'original_CWE_ID': df['original_CWE_ID'].astype(str).values,
                      'vul_prob': vp}).to_csv(os.path.join(args.out_dir, f'{split}_probs.csv'), index=False)
    logger.info("dumped test_probs.csv + val_probs.csv")

if __name__ == '__main__':
    main()
