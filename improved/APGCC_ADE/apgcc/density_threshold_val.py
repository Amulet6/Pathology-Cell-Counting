#!/usr/bin/env python3
"""Density-adaptive inference-time threshold, val->test protocol (committed A2 module).

Same idea as density_threshold.py, but NO test leakage and NO cross-validation:
the per-density-bin thresholds (and the density-bin edges) are LEARNED ON val and
applied ONCE to test. This matches the project protocol "val selects, test reports
once", so the reported MAE is a clean generalization number.

Motivation (Phase 0): a single global score threshold is provably wrong for CoNIC --
denser patches want a lower threshold (corr(density, per-image oracle thr) < 0).
This picks the score threshold PER IMAGE from a test-time-available density signal
(pred_count at a low reference threshold), instead of one global value.

We report:
  - GLOBAL  (val->test) : single best threshold learned on val   <- fixed-threshold baseline
  - DENSITY (val->test) : per-density-bin thresholds learned on val <- this module
  - PERIMG-ORACLE (test): each test image its own best threshold  <- absolute ceiling

Runs the model once per split; all thresholding is post-hoc. Example:
  python density_threshold_val.py --config ./configs/CoNIC_unified.yml \
     --weight ./output/CoNIC_eos0.10_tau0.10/best.pth --data-root /data1/llx/CoNICdata \
     --gpu 3 --ref 0.15 --train-list val.list --eval-list test.list \
     --out output/CoNIC_eos0.10_tau0.10/density_thr_val.json
"""
import argparse, json, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--train-list", default="val.list", help="list used to LEARN thresholds (no test leakage)")
    ap.add_argument("--eval-list", default="test.list", help="list to REPORT on, once")
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

    def run(eval_list):
        """Run model once over a list; return gts[n], dens[n], abserr[n, grid]."""
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
        return gts, dens, abserr

    # ---- LEARN on train-list (val) ----
    vg, vdens, vabserr = run(args.train_list)
    # global: single best threshold on val
    global_thr = float(grid[np.argmin(vabserr.mean(0))])
    # density: tercile edges + per-bin best threshold, all from val
    edges = np.quantile(vdens, np.linspace(0, 1, args.nbins + 1)[1:-1])
    def binof(v): return int(np.searchsorted(edges, v))
    bin_thr = {}
    for b in range(args.nbins):
        m = np.array([binof(d) == b for d in vdens])
        if m.sum() == 0:
            bin_thr[b] = global_thr
        else:
            bin_thr[b] = float(grid[np.argmin(vabserr[m].mean(0))])

    # ---- REPORT on eval-list (test), once ----
    tg, tdens, tabserr = run(args.eval_list)
    nt = len(tg)
    gcol = int(np.argmin(np.abs(grid - global_thr)))
    global_mae = float(tabserr[:, gcol].sum() / nt)
    dtot = 0.0
    for i in range(nt):
        col = int(np.argmin(np.abs(grid - bin_thr[binof(tdens[i])])))
        dtot += tabserr[i, col]
    density_mae = float(dtot / nt)
    perimg = float(tabserr.min(1).sum() / nt)                       # ceiling
    oracle_thr = grid[tabserr.argmin(1)]
    corr = float(np.corrcoef(tdens, oracle_thr)[0, 1])             # reference, on test

    res = {
        "train_list": args.train_list, "eval_list": args.eval_list,
        "n_train": int(len(vg)), "n_eval": nt, "ref": args.ref, "nbins": args.nbins, "grid": grid.tolist(),
        "learned_on_val": {
            "global_thr": global_thr,
            "bin_edges": edges.tolist(),
            "bin_thr": {str(k): v for k, v in bin_thr.items()},
        },
        "global_val2test_mae": global_mae,
        "density_val2test_mae": density_mae,
        "perimg_oracle_mae": perimg,
        "corr_density_oracleThr_test": corr,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print("=" * 64)
    print("DENSITY-ADAPTIVE THRESHOLD (val->test, no leakage)  ref=%.2f" % args.ref)
    print("  learn on %s (n=%d)  ->  report on %s (n=%d)" % (args.train_list, len(vg), args.eval_list, nt))
    print("=" * 64)
    print(f"  corr(density, per-image oracle thr) [test] = {corr:+.3f}  (neg => denser wants lower thr)")
    print(f"  learned global thr      = {global_thr:.2f}")
    print(f"  learned per-bin thr     = {res['learned_on_val']['bin_thr']}  edges={[round(e,1) for e in edges]}")
    print(f"  GLOBAL  (val->test)     MAE = {global_mae:6.2f}")
    print(f"  DENSITY (val->test)     MAE = {density_mae:6.2f}   Δ={global_mae-density_mae:+.2f}")
    print(f"  PERIMG-ORACLE (ceiling) MAE = {perimg:6.2f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
