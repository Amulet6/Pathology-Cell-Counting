#!/usr/bin/env python3
"""Sweep the score threshold on a (val) split to pick the calibration threshold.

Runs the model ONCE over the eval-list, caches per-sample class-1 scores + GT count,
then sweeps every score threshold in post (no extra forward passes). Reports counting
MAE/MSE/Total-Err% per threshold and prints the val-best by the fixed rule:
    min val MAE; tie -> higher threshold (more conservative).

Selection uses counting metrics only (localization is run once on test at the chosen
threshold via eval_centroid.py). Example:

  python scan_threshold.py --config ./configs/CoNIC_unified.yml \
      --weight ./output/CoNIC_unified/best.pth \
      --data-root /data1/llx/CoNICdata --eval-list val.list --gpu 3 \
      --out output/CoNIC_unified/val_scan/scan.json
"""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--eval-list", default="val.list")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import numpy as np
    import torch
    import torch.nn.functional as F

    from config import cfg, merge_from_file
    cfg = merge_from_file(cfg, args.config)
    cfg.DATASETS.DATA_ROOT = args.data_root
    cfg.config_file = args.config

    from datasets import build_dataset
    from models import build_model

    _, val_dl = build_dataset(cfg, eval_list=args.eval_list)

    model = build_model(cfg, training=False)
    model.cuda()
    model.eval()
    sd = torch.load(args.weight, map_location="cpu")
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    md = model.state_dict()
    md.update({k: v for k, v in sd.items() if k in md})
    model.load_state_dict(md)

    # Cache per-sample (gt_count, sorted scores) once.
    gt_counts, score_lists = [], []
    with torch.no_grad():
        for samples, targets in val_dl:
            samples = samples.cuda()
            outputs = model(samples)
            scores = F.softmax(outputs["pred_logits"], -1)[:, :, 1][0]
            score_lists.append(scores.detach().cpu().numpy())
            gt_counts.append(len(targets[0]["point"]))
    gt = np.array(gt_counts, dtype=np.float64)

    rows = []
    for thr in args.thresholds:
        pred = np.array([(s > thr).sum() for s in score_lists], dtype=np.float64)
        diff = pred - gt
        rows.append({
            "threshold": float(thr),
            "mae": float(np.mean(np.abs(diff))),
            "mse": float(np.mean(diff ** 2)),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "total_gt": int(gt.sum()),
            "total_pred": int(pred.sum()),
            "total_err_pct": float(diff.sum() / gt.sum() * 100) if gt.sum() else 0.0,
        })

    # Rule: min val MAE; tie -> higher threshold.
    best = sorted(rows, key=lambda r: (r["mae"], -r["threshold"]))[0]

    print("\n%s  (%s, n=%d)" % (args.config, args.eval_list, len(gt)))
    print(f"{'thr':>6} {'MAE':>9} {'MSE':>11} {'totErr%':>9} {'totPred':>9} {'totGT':>9}")
    print("-" * 60)
    for r in rows:
        mark = "  <- val-best" if r is best else ""
        print(f"{r['threshold']:>6.2f} {r['mae']:>9.3f} {r['mse']:>11.2f} "
              f"{r['total_err_pct']:>+8.1f} {r['total_pred']:>9d} {r['total_gt']:>9d}{mark}")
    print(f"\nval-best threshold = {best['threshold']:.2f} (rule: min val MAE, tie->higher)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": args.config, "weight": args.weight, "eval_list": args.eval_list,
         "n": len(gt), "rule": "min val MAE; tie -> higher threshold",
         "val_best_threshold": best["threshold"], "scan": rows}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
