"""统一点匹配计数评估脚本。

输入：predictions.json（GT 和 Pred 格式相同）
输出：MAE/MSE/Precision/Recall/F1 @ 多个阈值

用法：
    python evaluation/centroid_eval.py --gt gt.json --pred pred.json --thresholds 6 12 24
"""

import argparse, json, sys
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def load_predictions(path):
    with open(path) as f:
        data = json.load(f)
    samples = data.get("samples", [])
    return {s["id"]: np.array(s["points"], dtype=np.float64) for s in samples}


def match_points(gt_points, pred_points, threshold):
    """Hungarian pairwise matching at given distance threshold.

    Returns: n_TP, n_FP, n_FN, paired_indices
    """
    if len(gt_points) == 0 and len(pred_points) == 0:
        return 0, 0, 0, ([], [])
    if len(gt_points) == 0:
        return 0, len(pred_points), 0, ([], [])
    if len(pred_points) == 0:
        return 0, 0, len(gt_points), ([], [])

    dist_matrix = cdist(gt_points, pred_points, metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    tp = 0
    paired_gt, paired_pred = [], []
    for r, c in zip(row_ind, col_ind):
        if dist_matrix[r, c] <= threshold:
            tp += 1
            paired_gt.append(r)
            paired_pred.append(c)

    fn = len(gt_points) - tp
    fp = len(pred_points) - tp
    return tp, fp, fn, (paired_gt, paired_pred)


def evaluate(gt_path, pred_path, thresholds):
    gt_dict = load_predictions(gt_path)
    pred_dict = load_predictions(pred_path)

    sample_ids = sorted(set(gt_dict.keys()) & set(pred_dict.keys()))
    if len(sample_ids) == 0:
        print("ERROR: No matching sample IDs between GT and Pred.")
        print(f"  GT IDs (first 5): {list(gt_dict.keys())[:5]}")
        print(f"  Pred IDs (first 5): {list(pred_dict.keys())[:5]}")
        sys.exit(1)
    print(f"Evaluating {len(sample_ids)} samples\n")

    all_counts = []
    metrics = {t: {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}

    for sid in sample_ids:
        gt_pts = gt_dict[sid]
        pred_pts = pred_dict[sid]
        all_counts.append((len(gt_pts), len(pred_pts)))

        for t in thresholds:
            tp, fp, fn, _ = match_points(gt_pts, pred_pts, t)
            metrics[t]["tp"] += tp
            metrics[t]["fp"] += fp
            metrics[t]["fn"] += fn

    # --- Counting ---
    gt_array = np.array([c[0] for c in all_counts])
    pred_array = np.array([c[1] for c in all_counts])
    mae = np.mean(np.abs(pred_array - gt_array))
    mse = np.mean((pred_array - gt_array) ** 2)
    rmse = np.sqrt(mse)
    total_gt = gt_array.sum()
    total_pred = pred_array.sum()
    total_err_pct = (total_pred - total_gt) / total_gt * 100

    print("=" * 60)
    print("COUNTING METRICS")
    print("=" * 60)
    print(f"  Total GT:    {total_gt:>8d}")
    print(f"  Total Pred:  {total_pred:>8d}  ({total_err_pct:+.1f}%)")
    print(f"  MAE:         {mae:>8.2f}")
    print(f"  MSE:         {mse:>8.2f}")
    print(f"  RMSE:        {rmse:>8.2f}")

    print("\n" + "=" * 60)
    print("LOCALIZATION METRICS")
    print("=" * 60)
    print(f"{'Thr':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 40)
    for t in sorted(thresholds):
        m = metrics[t]
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        print(f"{t:>4}px {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

    # output JSON
    result = {
        "counting": {"mae": float(mae), "mse": float(mse), "rmse": float(rmse),
                      "total_gt": int(total_gt), "total_pred": int(total_pred),
                      "total_err_pct": float(total_err_pct)},
        "localization": {}
    }
    for t in sorted(thresholds):
        m = metrics[t]
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        result["localization"][f"{t}px"] = {
            "precision": float(prec), "recall": float(rec),
            "f1": float(f1), "tp": tp, "fp": fp, "fn": fn
        }

    out_path = pred_path.replace(".json", "_centroid_eval.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Point-matching counting & localization eval")
    parser.add_argument("--gt", required=True, help="GT predictions.json")
    parser.add_argument("--pred", required=True, help="Pred predictions.json")
    parser.add_argument("--thresholds", type=int, nargs="+",
                        default=[6, 12, 24], help="Distance thresholds in pixels")
    args = parser.parse_args()
    evaluate(args.gt, args.pred, args.thresholds)


if __name__ == "__main__":
    main()
