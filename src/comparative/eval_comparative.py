import argparse, os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, accuracy_score
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

CWES = ["cwe-022", "cwe-078", "cwe-079", "cwe-089"]

CONFIGS = [

    ("baseline",       "baseline"),
    ("V2(warm)",       "v2_warm"),
    ("DL3(adapter)",   "v2_ccpp_adapter64_zd_dl3_nd"),
    ("baseline+enh",   "baseline_aug_enhanced"),
    ("v2_aug(2x)",     "v2_aug"),
    ("enh(V2,5.7x)",   "v2_aug_enhanced"),
    ("DL3+enh",        "dl3_aug_enhanced"),

    ("DANN",           "dann"),
    ("CDAN",           "cdan"),
    ("MoEVD(group)",   "moevd_group2"),
    ("MulVuln-pool",   "mulvuln_pool"),
    ("SPLVD",          "splvd"),
]

def best_thr(y, p):
    ts = np.unique(np.concatenate([[0.0], p, [1.0]])); bt, bf = 0.5, -1
    for t in ts:
        f = f1_score(y, (p >= t).astype(int), average='macro', zero_division=0)
        if f > bf: bf, bt = f, t
    return bt

def vecs(root, cfg, seeds, folds):
    R = []
    for s in seeds:
        for fo in folds:
            d = f"{root}/{cfg}/seed{s}/fold{fo}"
            if not os.path.exists(f"{d}/test_probs.csv"):
                return None
            dv = pd.read_csv(f"{d}/val_probs.csv"); dt = pd.read_csv(f"{d}/test_probs.csv")
            yv, pv = dv['label'].astype(int).values, dv['vul_prob'].values
            yt, pt = dt['label'].astype(int).values, dt['vul_prob'].values
            t = best_thr(yv, pv)
            R.append([f1_score(yt, (pt >= 0.5).astype(int), average='macro', zero_division=0),
                      f1_score(yt, (pt >= t).astype(int), average='macro', zero_division=0),
                      roc_auc_score(yt, pt), average_precision_score(yt, pt)])
    return np.array(R)

def per_cwe(root, cfg, seeds, folds):
    Y = []; PR = []; CW = []
    for s in seeds:
        for fo in folds:
            dt = pd.read_csv(f"{root}/{cfg}/seed{s}/fold{fo}/test_probs.csv")
            yt, pt = dt['label'].astype(int).values, dt['vul_prob'].values
            Y += yt.tolist(); PR += (pt >= 0.5).astype(int).tolist()
            CW += dt['original_CWE_ID'].astype(str).str.lower().str.strip().tolist()
    Y, PR, CW = np.array(Y), np.array(PR), np.array(CW)
    return [accuracy_score(Y[CW == c], PR[CW == c]) if (CW == c).any() else float('nan') for c in CWES]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', default='userU')
    ap.add_argument('--root', default=None)
    ap.add_argument('--seeds', default='42,7,1234')
    ap.add_argument('--folds', default='1,2,3,4,5')
    a = ap.parse_args()
    root = a.root or f"exp_results/{a.protocol}"
    seeds = a.seeds.split(','); folds = a.folds.split(',')
    n = len(seeds) * len(folds)
    data = {name: vecs(root, cfg, seeds, folds) for name, cfg in CONFIGS}

    lab = ["mF1@0.5", "mF1@valcal", "ROC", "PR"]
    print(f"\n###### OVERALL (n={n}, mean +/- std) | root={root} ######")
    print(f"{'config':<15}" + "".join(f"{h:>15}" for h in lab) + f"{'rng@0.5':>9}")
    for name, _ in CONFIGS:
        A = data[name]
        if A is None:
            print(f"{name:<15}{'(missing)':>15}"); continue
        m, sd = A.mean(0), A.std(0, ddof=1)
        print(f"{name:<15}" + "".join(f"{m[i]:>7.4f}+/-{sd[i]:.3f}" for i in range(4))
              + f"{A[:,0].max()-A[:,0].min():>9.4f}")

    print(f"\n###### Per-CWE accuracy@0.5 (pooled n={n}) ######")
    print(f"{'config':<15}" + "".join(f"{c:>9}" for c in CWES))
    for name, cfg in CONFIGS:
        if data[name] is None: continue
        print(f"{name:<15}" + "".join(f"{v:>9.4f}" for v in per_cwe(root, cfg, seeds, folds)))

    if wilcoxon is not None:
        for ref in ["baseline", "V2(warm)"]:
            if data.get(ref) is None: continue
            print(f"\n###### Paired Wilcoxon vs {ref} (n={n}) ######")
            for name, _ in CONFIGS:
                A = data[name]
                if A is None or name == ref: continue
                cells = []
                for i, l in enumerate(lab):
                    dd = A[:, i] - data[ref][:, i]
                    p = wilcoxon(A[:, i], data[ref][:, i]).pvalue if np.any(dd != 0) else 1.0
                    cells.append(f"{l}={dd.mean()*100:+.2f}pp(p={p:.3f})")
                print(f"  {name:<14} " + "  ".join(cells))

if __name__ == '__main__':
    main()
