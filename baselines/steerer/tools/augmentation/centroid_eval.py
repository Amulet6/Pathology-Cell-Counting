#!/usr/bin/env python3
"""Unified point-matching evaluation for predictions.json or STEERER point txt.

Supported inputs:
  1. Shared predictions.json:
     {"samples": [{"id": "...", "points": [[x, y], ...]}, ...]}
  2. STEERER *_gt_loc.txt:
     image_id count x y box_w box_h category ...
  3. STEERER pred_points.txt:
     image_id count x y ...

Counting is evaluated on every GT sample. Missing prediction samples are treated
as empty predictions; prediction samples not present in GT are ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def parse_steerer_points(path: Path) -> dict[str, np.ndarray]:
    samples: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_no} has fewer than 2 columns")

            sample_id = parts[0]
            point_num = int(parts[1])
            values = [float(item) for item in parts[2:]]

            if point_num == 0:
                if values:
                    raise ValueError(
                        f"{path}:{line_no} declares 0 points but has coordinates")
                points = np.empty((0, 2), dtype=np.float64)
            elif len(values) == point_num * 2:
                points = np.asarray(
                    [[values[i], values[i + 1]] for i in range(0, len(values), 2)],
                    dtype=np.float64,
                )
            elif len(values) == point_num * 5:
                points = np.asarray(
                    [[values[i], values[i + 1]] for i in range(0, len(values), 5)],
                    dtype=np.float64,
                )
            else:
                raise ValueError(
                    f"{path}:{line_no} expected {point_num * 2} pred values "
                    f"or {point_num * 5} gt values, got {len(values)}")

            samples[sample_id] = points
    return samples


def load_predictions(path: Path) -> dict[str, np.ndarray]:
    text = path.read_text(encoding="utf-8").lstrip()
    if text.startswith("{"):
        data = json.loads(text)
        samples = data.get("samples", [])
        return {
            str(sample["id"]): np.asarray(sample.get("points", []), dtype=np.float64)
            for sample in samples
        }
    return parse_steerer_points(path)


def match_points(
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    threshold: int,
) -> tuple[int, int, int]:
    if len(gt_points) == 0 and len(pred_points) == 0:
        return 0, 0, 0
    if len(gt_points) == 0:
        return 0, len(pred_points), 0
    if len(pred_points) == 0:
        return 0, 0, len(gt_points)

    dist_matrix = cdist(gt_points, pred_points, metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    tp = 0
    for row, col in zip(row_ind, col_ind):
        if dist_matrix[row, col] <= threshold:
            tp += 1

    fn = len(gt_points) - tp
    fp = len(pred_points) - tp
    return tp, fp, fn


def default_output_path(pred_path: Path) -> Path:
    if pred_path.suffix.lower() == ".json":
        return pred_path.with_name(pred_path.stem + "_centroid_eval.json")
    return pred_path.with_name(pred_path.stem + "_centroid_eval.json")


def evaluate(
    gt_path: Path,
    pred_path: Path,
    thresholds: list[int],
    output_path: Path | None,
) -> dict:
    gt_dict = load_predictions(gt_path)
    pred_dict = load_predictions(pred_path)

    sample_ids = sorted(gt_dict.keys())
    if not sample_ids:
        print("ERROR: No samples found in GT.")
        print(f"  GT IDs (first 5): {list(gt_dict.keys())[:5]}")
        print(f"  Pred IDs (first 5): {list(pred_dict.keys())[:5]}")
        sys.exit(1)

    missing_pred_ids = sorted(set(gt_dict.keys()) - set(pred_dict.keys()))
    extra_pred_ids = sorted(set(pred_dict.keys()) - set(gt_dict.keys()))
    if missing_pred_ids:
        print(
            f"WARNING: {len(missing_pred_ids)} GT samples have no predictions; "
            "treating them as empty predictions.")
        print(f"  Missing pred IDs (first 5): {missing_pred_ids[:5]}")
    if extra_pred_ids:
        print(
            f"WARNING: {len(extra_pred_ids)} prediction samples are not in GT; "
            "ignoring them.")
        print(f"  Extra pred IDs (first 5): {extra_pred_ids[:5]}")

    print(f"Evaluating {len(sample_ids)} samples\n")

    counts: list[tuple[int, int]] = []
    metrics = {threshold: {"tp": 0, "fp": 0, "fn": 0} for threshold in thresholds}

    for sample_id in sample_ids:
        gt_points = gt_dict[sample_id]
        pred_points = pred_dict.get(sample_id, np.empty((0, 2), dtype=np.float64))
        counts.append((len(gt_points), len(pred_points)))

        for threshold in thresholds:
            tp, fp, fn = match_points(gt_points, pred_points, threshold)
            metrics[threshold]["tp"] += tp
            metrics[threshold]["fp"] += fp
            metrics[threshold]["fn"] += fn

    gt_array = np.asarray([item[0] for item in counts], dtype=np.int64)
    pred_array = np.asarray([item[1] for item in counts], dtype=np.int64)
    errors = pred_array - gt_array

    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))
    total_gt = int(gt_array.sum())
    total_pred = int(pred_array.sum())
    total_err_pct = None if total_gt == 0 else float((total_pred - total_gt) / total_gt * 100)

    print("=" * 60)
    print("COUNTING METRICS")
    print("=" * 60)
    print(f"  Total GT:    {total_gt:>8d}")
    if total_err_pct is None:
        print(f"  Total Pred:  {total_pred:>8d}")
    else:
        print(f"  Total Pred:  {total_pred:>8d}  ({total_err_pct:+.1f}%)")
    print(f"  MAE:         {mae:>8.2f}")
    print(f"  MSE:         {mse:>8.2f}")
    print(f"  RMSE:        {rmse:>8.2f}")

    print("\n" + "=" * 60)
    print("LOCALIZATION METRICS")
    print("=" * 60)
    print(f"{'Thr':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 40)

    result = {
        "input": {
            "gt": str(gt_path),
            "pred": str(pred_path),
            "coordinate_order": "xy",
            "coordinate_unit": "pixel",
        },
        "counting": {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "total_gt": total_gt,
            "total_pred": total_pred,
            "total_err_pct": total_err_pct,
            "num_samples": len(sample_ids),
            "missing_prediction_samples": len(missing_pred_ids),
            "ignored_extra_prediction_samples": len(extra_pred_ids),
        },
        "localization": {},
    }

    for threshold in sorted(thresholds):
        metric = metrics[threshold]
        tp, fp, fn = metric["tp"], metric["fp"], metric["fn"]
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        print(f"{threshold:>4}px {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")

        result["localization"][f"{threshold}px"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

    if output_path is None:
        output_path = default_output_path(pred_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {output_path}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Point-matching counting and localization evaluation")
    parser.add_argument("--gt", required=True, help="GT predictions.json or STEERER *_gt_loc.txt")
    parser.add_argument("--pred", required=True, help="Pred predictions.json or STEERER pred_points.txt")
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[6, 12, 24],
        help="Distance thresholds in pixels",
    )
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        Path(args.gt),
        Path(args.pred),
        args.thresholds,
        Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
