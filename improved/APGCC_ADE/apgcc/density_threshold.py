#!/usr/bin/env python3
"""Density-adaptive inference-time threshold (committed A2 module).

Motivation (Phase 0): a single global score threshold is provably wrong for CoNIC —
denser patches want a lower threshold (corr(pred_count@low, per-image oracle thr) < 0).
This picks the score threshold PER IMAGE from a test-time-available density signal
(pred_count at a low reference threshold), instead of one global value.

Honesty: thresholds are chosen by 2-fold cross-validation over the test images
(per-bin best threshold derived on the held-out fold, applied to the other), so the
reported density-adaptive MAE is NOT leaked. We report:
  - GLOBAL-CV     : single best threshold (CV)         <- what a fixed threshold gives
  - DENSITY-CV    : per-density-bin threshold (CV)      <- this module
  - PERIMG-ORACLE : each image its own best threshold   <- absolute ceiling

Runs the model once; all thresholding is post-hoc. Example:
  python density_threshold.py --config ./configs/CoNIC_unified.yml \
     --weight ./output/CoNIC_eos0.10_tau0.10/best.pth --data-root /data1/llx/CoNICdata \
     --gpu 3 --ref 0.15 --out output/CoNIC_eos0.10_tau0.10/density_thr.json
"""
import argparse, json, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--eval-list", default="test.list")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ref", type=float, default=0.15, help="ref threshold for density signal pred_count@ref")
    ap.add_argument("--nbins", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import numpy as np, torch, torch.nn.functional as F
    from config import cfg, merge_from_file
    cfg = merge_from_file(cfg, args.config)
    cfg.DATASETS.DATA_ROOT = args.data_root
    cfg.config_file = args.config
    from datasets import build_dataset
    from models import build_model

    _, dl = build_dataset(cfg, eval_list=args.eval_list)
    model = build_model(cfg, training=False); model.cuda(); model.eval()
    sd = torch.load(args.weight, map_location="cpu"); sd = sd.get("model", sd)
    md = model.state_dict(); md.update({k: v for k, v in sd.items() if k in md}); model.load_state_dict(md)

    grid = np.round(np.arange(0.10, 0.605, 0.05), 2)
    gts, scores = [], []
    with torch.no_grad():
        for samples, targets in dl:
            out = model(samples.cuda())
            scores.append(F.softmax(out["pred_logits"], -1)[:, :, 1][0].cpu().numpy())
            gts.append(len(targets[0]["point"]))
    gts = np.array(gts, float)
    n = len(gts)
    # pred count per image per grid threshold  [n, len(grid)]
    pc = np.array([[(s > t).sum() for t in grid] for s in scores], float)
    dens = np.array([(s > args.ref).sum() for s in scores], float)   # density signal
    abserr = np.abs(pc - gts[:, None])                                # [n, grid]

    def mae_global(train_idx, test_idx):
        ti = grid[np.argmin(abserr[train_idx].mean(0))]
        col = np.argmin(np.abs(grid - ti))
        return abserr[test_idx, col].sum(), ti

    def mae_density(train_idx, test_idx):
        # tercile bin edges from TRAIN density
        edges = np.quantile(dens[train_idx], np.linspace(0, 1, args.nbins + 1)[1:-1])
        def binof(v): return int(np.searchsorted(edges, v))
        # best threshold per bin on TRAIN
        bin_thr = {}
        for b in range(args.nbins):
            m = np.array([binof(dens[i]) == b for i in train_idx])
            idx = np.array(train_idx)[m]
            if len(idx) == 0:
                bin_thr[b] = grid[np.argmin(abserr[train_idx].mean(0))]
            else:
                bin_thr[b] = grid[np.argmin(abserr[idx].mean(0))]
        tot = 0.0
        for i in test_idx:
            col = np.argmin(np.abs(grid - bin_thr[binof(dens[i])]))
            tot += abserr[i, col]
        return tot, bin_thr, edges.tolist()

    # 2-fold CV by parity
    fa = [i for i in range(n) if i % 2 == 0]
    fb = [i for i in range(n) if i % 2 == 1]
    g = (mae_global(fb, fa)[0] + mae_global(fa, fb)[0]) / n
    da, ta, _ = mae_density(fb, fa)
    db = mae_density(fa, fb)[0]
    d = (da + db) / n
    perimg = abserr.min(1).sum() / n        # ceiling
    # reference: corr(density, per-image oracle threshold)
    oracle_thr = grid[abserr.argmin(1)]
    corr = float(np.corrcoef(dens, oracle_thr)[0, 1])

    res = {
        "n": n, "ref": args.ref, "grid": grid.tolist(),
        "global_cv_mae": float(g), "density_cv_mae": float(d), "perimg_oracle_mae": float(perimg),
        "corr_density_oracleThr": corr,
        "bin_thr_foldB_applied_to_A": {str(k): float(v) for k, v in ta.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("=" * 64)
    print("DENSITY-ADAPTIVE THRESHOLD (2-fold CV, honest)  n=%d ref=%.2f" % (n, args.ref))
    print("=" * 64)
    print(f"  corr(density, per-image oracle thr) = {corr:+.3f}  (neg => denser wants lower thr)")
    print(f"  GLOBAL-CV      (single threshold)    MAE = {g:6.2f}")
    print(f"  DENSITY-CV     (per-density-bin)     MAE = {d:6.2f}   Δ={g-d:+.2f}")
    print(f"  PERIMG-ORACLE  (ceiling)             MAE = {perimg:6.2f}")
    print(f"  per-bin thr (foldB→A): {res['bin_thr_foldB_applied_to_A']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
