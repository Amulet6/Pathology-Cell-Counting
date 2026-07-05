#!/usr/bin/env python3
"""Density-adaptive inference-time threshold, CORRECT val->test protocol.

Supersedes density_threshold.py's 2-fold-CV-on-test estimate, which tuned the
per-bin thresholds using the test set itself and is NOT a legitimate protocol.

Here every hyperparameter that touches labels is fit on the VALIDATION split and
then FROZEN and applied unchanged to the test split:
  - tercile density-bin edges  : quantiles of val density signal
  - per-bin score threshold    : argmin mean|pred_count - gt| over val bin images
  - global single threshold    : argmin mean|pred_count - gt| over all val images
Test labels are used ONLY to score the frozen thresholds. Reported:
  - GLOBAL-VAL   : single val-selected threshold, applied to test
  - DENSITY-VAL  : per-density-bin val-selected thresholds, applied to test
  - PERIMG-ORACLE: each test image its own best threshold (ceiling, not attainable)

Runs the model once per split; all thresholding is post-hoc. Example:
  python density_threshold_valtest.py --config ./configs/CoNIC_DE_stain_plain.yml \
     --weight output/CoNIC_DE_stain_plain/best.pth --data-root /data1/llx/CoNICdata \
     --gpu 0 --ref 0.15 --nbins 3 --out output/CoNIC_DE_stain_plain/density_thr_valtest.json
"""
import argparse, json, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--val-list", default="val.list")
    ap.add_argument("--test-list", default="test.list")
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

    model = build_model(cfg, training=False); model.cuda(); model.eval()
    sd = torch.load(args.weight, map_location="cpu"); sd = sd.get("model", sd)
    md = model.state_dict(); md.update({k: v for k, v in sd.items() if k in md}); model.load_state_dict(md)

    grid = np.round(np.arange(0.10, 0.605, 0.05), 2)

    def forward(eval_list):
        _, dl = build_dataset(cfg, eval_list=eval_list)
        gts, scores = [], []
        with torch.no_grad():
            for samples, targets in dl:
                out = model(samples.cuda())
                scores.append(F.softmax(out["pred_logits"], -1)[:, :, 1][0].cpu().numpy())
                gts.append(len(targets[0]["point"]))
        gts = np.array(gts, float)
        pc = np.array([[(s > t).sum() for t in grid] for s in scores], float)   # [n, grid]
        dens = np.array([(s > args.ref).sum() for s in scores], float)          # density signal
        abserr = np.abs(pc - gts[:, None])                                      # [n, grid]
        return gts, pc, dens, abserr

    vg, vpc, vdens, vabs = forward(args.val_list)
    tg, tpc, tdens, tabs = forward(args.test_list)

    # ---- fit on VAL, freeze ----
    global_thr = grid[np.argmin(vabs.mean(0))]
    edges = np.quantile(vdens, np.linspace(0, 1, args.nbins + 1)[1:-1])   # val tercile edges
    def binof(v): return int(np.searchsorted(edges, v))
    bin_thr = {}
    for b in range(args.nbins):
        m = np.array([binof(d) == b for d in vdens])
        bin_thr[b] = grid[np.argmin(vabs[m].mean(0))] if m.sum() else global_thr

    # ---- apply frozen thresholds to TEST ----
    gcol = int(np.argmin(np.abs(grid - global_thr)))
    global_mae = float(tabs[:, gcol].mean())
    dtot = 0.0
    for i in range(len(tg)):
        col = int(np.argmin(np.abs(grid - bin_thr[binof(tdens[i])])))
        dtot += tabs[i, col]
    density_mae = float(dtot / len(tg))
    perimg = float(tabs.min(1).mean())
    corr = float(np.corrcoef(tdens, grid[tabs.argmin(1)])[0, 1])

    res = {
        "protocol": "val-fit -> test-apply (frozen)",
        "n_val": int(len(vg)), "n_test": int(len(tg)), "ref": args.ref, "nbins": args.nbins,
        "grid": grid.tolist(),
        "val_bin_edges": edges.tolist(),
        "val_global_thr": float(global_thr),
        "val_bin_thr": {str(k): float(v) for k, v in bin_thr.items()},
        "test_global_val_mae": global_mae,
        "test_density_val_mae": density_mae,
        "test_perimg_oracle_mae": perimg,
        "corr_density_oracleThr_test": corr,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("=" * 68)
    print("DENSITY-ADAPTIVE THRESHOLD (val->test, honest)  n_val=%d n_test=%d ref=%.2f"
          % (len(vg), len(tg), args.ref))
    print("=" * 68)
    print(f"  val global thr = {global_thr:.2f} | val bin edges = {edges.tolist()} | val bin thr = {res['val_bin_thr']}")
    print(f"  corr(test density, per-image oracle thr) = {corr:+.3f}")
    print(f"  GLOBAL-VAL   (single val thr on test)   MAE = {global_mae:6.2f}")
    print(f"  DENSITY-VAL  (per-bin val thr on test)  MAE = {density_mae:6.2f}   Δ={global_mae-density_mae:+.2f}")
    print(f"  PERIMG-ORACLE(test ceiling)             MAE = {perimg:6.2f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
