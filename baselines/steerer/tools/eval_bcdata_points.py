#!/usr/bin/env python3
"""Evaluate BCData point localization from exported STEERER predictions.

Expected prediction format:
  image_id num x1 y1 x2 y2 ...

Expected GT format:
  image_id num x1 y1 w1 h1 level1 x2 y2 w2 h2 level2 ...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - fallback for minimal environments
    linear_sum_assignment = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BCData point-level Precision/Recall/F1.")
    parser.add_argument("--gt", required=True, help="Path to *_gt_loc.txt")
    parser.add_argument("--pred", required=True, help="Path to pred_points.txt exported by tools/test_loc.py")
    parser.add_argument("--radii", nargs="+", type=float, default=[8.0, 16.0], help="Matching radii in pixels")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def empty_points() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def hungarian(matrix_tf: np.ndarray) -> tuple[int, np.ndarray]:
    edges = np.argwhere(matrix_tf)
    left_num, right_num = matrix_tf.shape
    graph = [[] for _ in range(left_num)]
    for left_idx, right_idx in edges:
        graph[left_idx].append(right_idx)

    match = [-1 for _ in range(right_num)]
    visited = [-1 for _ in range(right_num)]

    def dfs(left_idx: int) -> bool:
        for right_idx in graph[left_idx]:
            if visited[right_idx]:
                continue
            visited[right_idx] = True
            if match[right_idx] == -1 or dfs(match[right_idx]):
                match[right_idx] = left_idx
                return True
        return False

    matched = 0
    for left_idx in range(left_num):
        for right_idx in range(right_num):
            visited[right_idx] = False
        if dfs(left_idx):
            matched += 1

    assignment = np.zeros((left_num, right_num), dtype=bool)
    for right_idx, left_idx in enumerate(match):
        if left_idx >= 0:
            assignment[left_idx, right_idx] = True
    return matched, assignment


def greedy_nearest_assignment(distances: np.ndarray, radius: float) -> np.ndarray:
    assignment = np.zeros_like(distances, dtype=bool)
    pairs = np.argwhere(distances <= radius)
    if pairs.size == 0:
        return assignment

    order = np.argsort(distances[pairs[:, 0], pairs[:, 1]])
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for pair_idx in order:
        pred_idx, gt_idx = pairs[pair_idx]
        pred_idx = int(pred_idx)
        gt_idx = int(gt_idx)
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        assignment[pred_idx, gt_idx] = True
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
    return assignment


def read_points(path: str | Path) -> dict[int, np.ndarray]:
    data: dict[int, np.ndarray] = {}
    with open(path, "r") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_no} has fewer than two columns")

            image_id = int(parts[0])
            num_points = int(parts[1])
            values = [float(item) for item in parts[2:]]

            if num_points == 0:
                if values:
                    raise ValueError(f"{path}:{line_no} declares 0 points but has coordinates")
                data[image_id] = empty_points()
                continue

            if len(values) == num_points * 2:
                stride = 2
            elif len(values) == num_points * 5:
                stride = 5
            else:
                raise ValueError(
                    f"{path}:{line_no} expected {num_points * 2} prediction values "
                    f"or {num_points * 5} GT values, got {len(values)}")

            points = np.asarray(values, dtype=np.float32).reshape(num_points, stride)[:, :2]
            data[image_id] = points
    return data


def match_points(pred_points: np.ndarray, gt_points: np.ndarray, radius: float) -> dict[str, float]:
    pred_points = np.asarray(pred_points, dtype=np.float32).reshape(-1, 2)
    gt_points = np.asarray(gt_points, dtype=np.float32).reshape(-1, 2)

    pred_num = pred_points.shape[0]
    gt_num = gt_points.shape[0]
    if pred_num == 0 and gt_num == 0:
        return {"tp": 0, "fp": 0, "fn": 0, "matched_distance_sum": 0.0}
    if pred_num == 0:
        return {"tp": 0, "fp": 0, "fn": gt_num, "matched_distance_sum": 0.0}
    if gt_num == 0:
        return {"tp": 0, "fp": pred_num, "fn": 0, "matched_distance_sum": 0.0}

    distances = np.linalg.norm(pred_points[:, None, :] - gt_points[None, :, :], axis=2)
    if linear_sum_assignment is not None:
        invalid_cost = max(float(distances.max(initial=0.0)), radius) + 1e6
        cost = distances.copy()
        cost[cost > radius] = invalid_cost
        rows, cols = linear_sum_assignment(cost)
        keep = distances[rows, cols] <= radius
        assignment = np.zeros_like(distances, dtype=bool)
        assignment[rows[keep], cols[keep]] = True
    else:
        assignment = greedy_nearest_assignment(distances, radius)

    tp = int(assignment.sum())
    fp = int(pred_num - tp)
    fn = int(gt_num - tp)
    matched_distance_sum = float(distances[assignment].sum()) if tp > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "matched_distance_sum": matched_distance_sum}


def summarize(gt_data: dict[int, np.ndarray], pred_data: dict[int, np.ndarray], radii: list[float]) -> dict:
    image_ids = sorted(set(gt_data) | set(pred_data))
    summary = {
        "num_images": len(image_ids),
        "num_gt_points": int(sum(gt_data.get(image_id, empty_points()).shape[0] for image_id in image_ids)),
        "num_pred_points": int(sum(pred_data.get(image_id, empty_points()).shape[0] for image_id in image_ids)),
        "radii": {},
    }

    for radius in radii:
        totals = {"tp": 0, "fp": 0, "fn": 0, "matched_distance_sum": 0.0}
        for image_id in image_ids:
            result = match_points(
                pred_data.get(image_id, empty_points()),
                gt_data.get(image_id, empty_points()),
                radius,
            )
            for key in totals:
                totals[key] += result[key]

        precision = totals["tp"] / (totals["tp"] + totals["fp"] + 1e-20)
        recall = totals["tp"] / (totals["tp"] + totals["fn"] + 1e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        mle = totals["matched_distance_sum"] / (totals["tp"] + 1e-20)
        summary["radii"][str(radius)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mle_matched_points": mle,
            **totals,
        }
    return summary


def main() -> None:
    args = parse_args()
    gt_data = read_points(args.gt)
    pred_data = read_points(args.pred)
    summary = summarize(gt_data, pred_data, args.radii)

    for radius, result in summary["radii"].items():
        print(
            "radius={}px precision={:.6f} recall={:.6f} f1={:.6f} mle={:.6f} "
            "tp={} fp={} fn={}".format(
                radius,
                result["precision"],
                result["recall"],
                result["f1"],
                result["mle_matched_points"],
                result["tp"],
                result["fp"],
                result["fn"],
            )
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
