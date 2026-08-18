import argparse
import csv
import os
import sys

import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mavd import BaselineCodeBERT, MulVulMultiTaskV2, MulVulSoftMoEV3, MulVulV2WithLangAdapter
from train_v2_multitask import add_group_labels
from ultis import load_data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_type', choices=['baseline', 'v2', 'v3', 'adapter'], required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--files', nargs='+', required=True)
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--pretrained_model', default='microsoft/codebert-base')
    ap.add_argument('--max_len', type=int, default=512)
    ap.add_argument('--num_groups', type=int, default=2)
    ap.add_argument('--adapter_dim', type=int, default=64)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(args.pretrained_model)
    backbone = AutoModel.from_pretrained(args.pretrained_model)
    if args.model_type == 'baseline':
        model = BaselineCodeBERT(backbone, num_labels=1)
    elif args.model_type == 'v3':
        model = MulVulSoftMoEV3(backbone, num_groups=args.num_groups)
    elif args.model_type == 'adapter':

        model = MulVulV2WithLangAdapter(backbone, num_groups=args.num_groups,
                                        adapter_dim=args.adapter_dim)
    else:
        model = MulVulMultiTaskV2(backbone, num_groups=args.num_groups)

    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    model.to(device).eval()

    df = add_group_labels(load_data(args.files), allow_unmapped=True)
    rows = []
    with torch.no_grad():
        for _, r in df.iterrows():
            enc = tok(r['code'], truncation=True, padding='max_length',
                      max_length=args.max_len, return_tensors='pt').to(device)
            out = model(enc['input_ids'], enc['attention_mask'])
            logit = out[0] if isinstance(out, tuple) else out
            prob = float(torch.sigmoid(logit.squeeze()).item())
            rows.append({'label': int(r['label']),
                         'original_CWE_ID': str(r.get('original_CWE_ID', '')),
                         'vul_prob': prob})
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['label', 'original_CWE_ID', 'vul_prob'])
        w.writeheader()
        w.writerows(rows)
    print(f"dumped {len(rows)} -> {args.out_csv}")

if __name__ == '__main__':
    main()
