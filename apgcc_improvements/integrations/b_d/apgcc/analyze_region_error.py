#!/usr/bin/env python3
"""
Region-level error analysis for MoNuSeg (1000×1000 images).

Splits each image into a 4×4 grid (16 regions, each 250×250 px) and computes
per-region MAE/MSE, aggregated across all test images (14 images).

Outer ring = top/bottom rows + left/right columns (12 regions per image).
Inner core = the 2×2 center region (4 regions per image).

Hypothesis: Edge Ignore suppresses false positives from truncated cells at
crop boundaries, so outer-ring regions should see a larger reduction in
over-counting (MAE, total_pred) compared to the inner core.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def split_into_regions(points, grid=4, img_size=1000):
    """Partition points into grid×grid spatial regions.

    Args:
        points: [[x, y], ...] list of point coordinates (image frame, px).
        grid: number of divisions per axis.
        img_size: image side length in px (square).

    Returns:
        dict mapping (row, col) → list of points.
    """
    cell_size = img_size / grid
    regions = defaultdict(list)
    for px, py in points:
        col = min(int(px // cell_size), grid - 1)
        row = min(int(py // cell_size), grid - 1)
        regions[(row, col)].append([px, py])
    return regions


def is_outer(row, col, grid=4):
    """Return True if the region is on the outer ring (border row or column)."""
    return row == 0 or row == grid - 1 or col == 0 or col == grid - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="gt.json from eval_centroid.py")
    ap.add_argument("--pred", required=True, help="pred.json from eval_centroid.py")
    ap.add_argument("--label", default="", help="label for printing")
    ap.add_argument("--grid", type=int, default=4, help="grid divisions per axis")
    ap.add_argument("--img-size", type=int, default=1000,
                    help="image side length in px")
    args = ap.parse_args()

    with open(args.gt) as f:
        gt_data = json.load(f)
    with open(args.pred) as f:
        pred_data = json.load(f)

    # Build lookup: sample id → points
    gt_by_id = {s["id"]: s["points"] for s in gt_data["samples"]}
    pred_by_id = {s["id"]: s["points"] for s in pred_data["samples"]}

    ids = sorted(set(gt_by_id.keys()) & set(pred_by_id.keys()))
    if not ids:
        # try matching by index
        ids = list(range(len(gt_data["samples"])))
        gt_by_id = {i: s["points"] for i, s in enumerate(gt_data["samples"])}
        pred_by_id = {i: s["points"] for i, s in enumerate(pred_data["samples"])}

    # Accumulators
    outer_ae = []       # absolute errors in outer regions
    outer_se = []       # squared errors
    outer_gt = []       # gt counts per region
    outer_pred = []     # pred counts per region
    inner_ae = []
    inner_se = []
    inner_gt = []
    inner_pred = []
    all_ae = []
    all_se = []
    all_gt = []
    all_pred = []

    for sid in ids:
        gt_pts = gt_by_id[sid]
        pred_pts = pred_by_id[sid]

        gt_regions = split_into_regions(gt_pts, args.grid, args.img_size)
        pred_regions = split_into_regions(pred_pts, args.grid, args.img_size)

        # Aggregate across all 16 regions
        all_cells = set(gt_regions.keys()) | set(pred_regions.keys())
        for (r, c) in all_cells:
            n_gt = len(gt_regions.get((r, c), []))
            n_pred = len(pred_regions.get((r, c), []))
            ae = abs(n_pred - n_gt)
            se = (n_pred - n_gt) ** 2

            all_ae.append(ae)
            all_se.append(se)
            all_gt.append(n_gt)
            all_pred.append(n_pred)

            if is_outer(r, c, args.grid):
                outer_ae.append(ae)
                outer_se.append(se)
                outer_gt.append(n_gt)
                outer_pred.append(n_pred)
            else:
                inner_ae.append(ae)
                inner_se.append(se)
                inner_gt.append(n_gt)
                inner_pred.append(n_pred)

    outer_mae = np.mean(outer_ae)
    outer_rmse = np.sqrt(np.mean(outer_se))
    outer_total_gt = sum(outer_gt)
    outer_total_pred = sum(outer_pred)
    outer_over = 100.0 * (outer_total_pred - outer_total_gt) / max(outer_total_gt, 1)

    inner_mae = np.mean(inner_ae)
    inner_rmse = np.sqrt(np.mean(inner_se))
    inner_total_gt = sum(inner_gt)
    inner_total_pred = sum(inner_pred)
    inner_over = 100.0 * (inner_total_pred - inner_total_gt) / max(inner_total_gt, 1)

    all_mae = np.mean(all_ae)
    all_rmse = np.sqrt(np.mean(all_se))
    all_total_gt = sum(all_gt)
    all_total_pred = sum(all_pred)
    all_over = 100.0 * (all_total_pred - all_total_gt) / max(all_total_gt, 1)

    tag = f"[{args.label}] " if args.label else ""
    print(f"\n{'='*55}")
    print(f"  {tag}Region-Level Error (4×4 grid, {args.img_size}×{args.img_size} px)")
    print(f"{'='*55}")
    print(f"  {'Zone':<20s} {'MAE':>8s}  {'RMSE':>8s}  {'Total GT':>8s}  {'Total Pred':>10s}  {'Over%':>7s}")
    print(f"  {'-'*55}")
    print(f"  {'外圈 (12 regions)':<20s} {outer_mae:>8.2f}  {outer_rmse:>8.2f}  {outer_total_gt:>8d}  {outer_total_pred:>10d}  {outer_over:>+6.1f}%")
    print(f"  {'内部 (4 regions)':<20s} {inner_mae:>8.2f}  {inner_rmse:>8.2f}  {inner_total_gt:>8d}  {inner_total_pred:>10d}  {inner_over:>+6.1f}%")
    print(f"  {'-'*55}")
    print(f"  {'全图 (16 regions)':<20s} {all_mae:>8.2f}  {all_rmse:>8.2f}  {all_total_gt:>8d}  {all_total_pred:>10d}  {all_over:>+6.1f}%")
    print()

    return {
        "label": args.label,
        "outer_mae": outer_mae, "outer_rmse": outer_rmse,
        "outer_gt": outer_total_gt, "outer_pred": outer_total_pred,
        "outer_over_pct": outer_over,
        "inner_mae": inner_mae, "inner_rmse": inner_rmse,
        "inner_gt": inner_total_gt, "inner_pred": inner_total_pred,
        "inner_over_pct": inner_over,
        "all_mae": all_mae, "all_rmse": all_rmse,
        "all_gt": all_total_gt, "all_pred": all_total_pred,
        "all_over_pct": all_over,
    }


if __name__ == "__main__":
    main()
