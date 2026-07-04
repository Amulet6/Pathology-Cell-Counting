#!/usr/bin/env python3
"""Run a trained APGCC model on a dataset's test.list, dump predictions + GT in the
team-unified centroid_eval.py format, then invoke the repo's centroid_eval.py to get
counting (MAE/MSE/RMSE) + localization (P/R/F1 @ thresholds) metrics.

Unified json format:  {"samples": [{"id": <stem>, "points": [[x, y], ...]}, ...]}

GT points are taken from the dataloader target (same coordinate space as the model
prediction — both are scaled by the eval-time resize, so pixel thresholds are aligned).

Example:
  python eval_centroid.py --config ./configs/CoNIC_finetune.yml \
      --weight ./output/CoNIC_finetune/best.pth \
      --data-root /data1/llx/CoNICdata --gpu 2 \
      --out-dir ./output/CoNIC_finetune/centroid_eval
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--score-threshold", type=float, default=0.5,
                    help="keep predicted points with class-1 score > this")
    ap.add_argument("--eval-list", default="test.list")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--thresholds", type=int, nargs="+", default=[6, 12, 24],
                    help="distance thresholds (px) for localization P/R/F1")
    ap.add_argument("--edge-band", type=int, default=0,
                    help="edge-aware filter band (px), 0=disabled")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    import torch.nn.functional as F

    from config import cfg, merge_from_file
    cfg = merge_from_file(cfg, args.config)
    cfg.DATASETS.DATA_ROOT = args.data_root
    cfg.config_file = args.config

    from datasets import build_dataset
    from models import build_model
    from engine import filter_prediction_points

    _, val_dl = build_dataset(cfg, eval_list=args.eval_list)

    model = build_model(cfg, training=False)
    model.cuda()
    model.eval()
    sd = torch.load(args.weight, map_location="cpu")
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    model.load_state_dict(sd, strict=False)

    gt_samples, pred_samples = [], []
    with torch.no_grad():
        for samples, targets in val_dl:
            samples = samples.cuda()
            outputs = model(samples)
            scores = F.softmax(outputs["pred_logits"], -1)[:, :, 1][0]
            pts = outputs["pred_points"][0]
            pred_pts_tensor, _ = filter_prediction_points(
                pts, scores, args.score_threshold, cfg,
                samples=samples, edge_band=args.edge_band)
            pred_pts = pred_pts_tensor.detach().cpu().numpy().tolist()
            gt_pts = targets[0]["point"].detach().cpu().numpy().tolist()
            sid = os.path.splitext(targets[0]["name"])[0]
            pred_samples.append({"id": sid, "points": pred_pts})
            gt_samples.append({"id": sid, "points": gt_pts})

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gt_json = out / "gt.json"
    pred_json = out / "pred.json"
    gt_json.write_text(json.dumps({"samples": gt_samples}))
    pred_json.write_text(json.dumps({"samples": pred_samples}))
    print("wrote %s (%d samples), %s (%d samples)\n" %
          (gt_json, len(gt_samples), pred_json, len(pred_samples)))

    repo_eval = Path(__file__).resolve().parent / "centroid_eval.py"
    cmd = [sys.executable, str(repo_eval), "--gt", str(gt_json), "--pred", str(pred_json),
           "--thresholds", *map(str, args.thresholds)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
