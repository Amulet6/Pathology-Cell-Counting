#!/usr/bin/env python3
"""Domain + density-aware threshold calibration for APGCC.

This script keeps APGCC weights fixed. It calibrates score thresholds on val.list
for each (source domain, local density bin) pair, then evaluates the fixed rule on
test.list. Local density is measured on a 4x4 grid from candidate predictions whose
score is above a low base threshold.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


DENSITY_BINS = ("low", "mid", "high")


def domain_from_id(sample_id, domain_names):
    stem = sample_id.lower()
    for name in domain_names:
        key = name.lower()
        if stem == key or stem.startswith(key + "_") or stem.startswith(key + "-"):
            return key
    return "unknown"


def point_region(point, width, height, grid_size):
    x, y = float(point[0]), float(point[1])
    col = int(np.clip(x / max(width, 1e-6) * grid_size, 0, grid_size - 1))
    row = int(np.clip(y / max(height, 1e-6) * grid_size, 0, grid_size - 1))
    return row * grid_size + col


def sample_region_counts(sample, base_threshold, grid_size):
    counts = [0 for _ in range(grid_size * grid_size)]
    width, height = sample["size"]
    for point, score in zip(sample["points"], sample["scores"]):
        if score > base_threshold:
            counts[point_region(point, width, height, grid_size)] += 1
    return counts


def build_density_edges(raw_samples, base_threshold, grid_size):
    all_counts = []
    domain_counts = {}
    for sample in raw_samples:
        counts = sample_region_counts(sample, base_threshold, grid_size)
        all_counts.extend(counts)
        domain_counts.setdefault(sample["domain"], []).extend(counts)

    def edges(values):
        arr = np.asarray(values, dtype=np.float32)
        if len(arr) == 0:
            return [0.0, 1.0]
        q1, q2 = np.percentile(arr, [33.333, 66.667])
        if q2 <= q1:
            q2 = q1 + 1.0
        return [float(q1), float(q2)]

    result = {"global": edges(all_counts), "by_domain": {}}
    for domain, counts in domain_counts.items():
        result["by_domain"][domain] = edges(counts)
    return result


def density_bin(count, edges):
    if count <= edges[0]:
        return "low"
    if count <= edges[1]:
        return "mid"
    return "high"


def annotate_density_bins(raw_samples, density_edges, base_threshold, grid_size, use_domain_edges):
    annotated = []
    for sample in raw_samples:
        counts = sample_region_counts(sample, base_threshold, grid_size)
        edges = density_edges["by_domain"].get(sample["domain"], density_edges["global"]) if use_domain_edges else density_edges["global"]
        region_bins = [density_bin(c, edges) for c in counts]
        width, height = sample["size"]
        point_bins = [region_bins[point_region(p, width, height, grid_size)] for p in sample["points"]]
        item = dict(sample)
        item["region_counts"] = counts
        item["region_bins"] = region_bins
        item["point_bins"] = point_bins
        annotated.append(item)
    return annotated


def count_tp(gt, pred, threshold):
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, 2)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 2)
    if len(gt) == 0 or len(pred) == 0:
        return 0
    diff = pred[:, None, :] - gt[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    rows, cols = linear_sum_assignment(dist)
    return int((dist[rows, cols] <= threshold).sum())


def threshold_for_point(sample, point_bin, threshold_map, default_threshold):
    domain = sample["domain"]
    if domain in threshold_map and point_bin in threshold_map[domain]:
        return threshold_map[domain][point_bin]
    return default_threshold


def filter_samples(raw_samples, threshold_map, default_threshold):
    filtered = []
    for sample in raw_samples:
        points = []
        for point, score, point_bin in zip(sample["points"], sample["scores"], sample["point_bins"]):
            if score > threshold_for_point(sample, point_bin, threshold_map, default_threshold):
                points.append(point)
        filtered.append({
            "id": sample["id"],
            "domain": sample["domain"],
            "size": sample["size"],
            "points": points,
        })
    return filtered


def region_count_grid(points, width, height, grid_size):
    counts = [0 for _ in range(grid_size * grid_size)]
    for point in points:
        counts[point_region(point, width, height, grid_size)] += 1
    return counts


def summarize(gt_samples, pred_samples, thresholds, grid_size):
    errors = []
    total_gt = 0
    total_pred = 0
    tp_by_threshold = {int(t): 0 for t in thresholds}
    by_domain = {}
    region_abs_errors = []
    region_sq_errors = []

    for gt_sample, pred_sample in zip(gt_samples, pred_samples):
        domain = gt_sample.get("domain", "unknown")
        gt_pts = gt_sample["points"]
        pred_pts = pred_sample["points"]
        error = len(pred_pts) - len(gt_pts)
        errors.append(error)
        total_gt += len(gt_pts)
        total_pred += len(pred_pts)

        width, height = gt_sample["size"]
        gt_grid = region_count_grid(gt_pts, width, height, grid_size)
        pred_grid = region_count_grid(pred_pts, width, height, grid_size)
        for g, p in zip(gt_grid, pred_grid):
            region_abs_errors.append(abs(p - g))
            region_sq_errors.append((p - g) ** 2)

        dom = by_domain.setdefault(domain, {"errors": [], "gt_count": 0, "pred_count": 0,
                                            "tp_by_threshold": {int(t): 0 for t in thresholds}})
        dom["errors"].append(error)
        dom["gt_count"] += len(gt_pts)
        dom["pred_count"] += len(pred_pts)

        for threshold in tp_by_threshold:
            tp = count_tp(gt_pts, pred_pts, threshold)
            tp_by_threshold[threshold] += tp
            dom["tp_by_threshold"][threshold] += tp

    def metric_block(errors_arr, gt_count, pred_count, tp_map):
        errors_arr = np.asarray(errors_arr, dtype=np.float64)
        mae = float(np.mean(np.abs(errors_arr))) if len(errors_arr) else 0.0
        mse = float(np.mean(errors_arr ** 2)) if len(errors_arr) else 0.0
        out = {
            "samples": int(len(errors_arr)),
            "gt_count": int(gt_count),
            "pred_count": int(pred_count),
            "MAE": mae,
            "MSE": mse,
            "RMSE": float(math.sqrt(mse)),
            "thresholds": {},
        }
        for threshold, tp in tp_map.items():
            precision = tp / float(pred_count + 1e-10)
            recall = tp / float(gt_count + 1e-10)
            f1 = 2 * precision * recall / float(precision + recall + 1e-10)
            out["thresholds"][str(threshold)] = {
                "TP": int(tp),
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        return out

    result = metric_block(errors, total_gt, total_pred, tp_by_threshold)
    result["region_MAE"] = float(np.mean(region_abs_errors)) if region_abs_errors else 0.0
    result["region_MSE"] = float(np.mean(region_sq_errors)) if region_sq_errors else 0.0
    result["by_domain"] = {}
    for domain, dom in sorted(by_domain.items()):
        result["by_domain"][domain] = metric_block(
            dom["errors"], dom["gt_count"], dom["pred_count"], dom["tp_by_threshold"])
    return result


def collect_predictions(cfg, config_path, weight_path, data_root, eval_list, domain_names):
    import torch
    import torch.nn.functional as F

    from datasets import build_dataset
    from models import build_model

    cfg.DATASETS.DATA_ROOT = data_root
    cfg.config_file = config_path
    _, data_loader = build_dataset(cfg, eval_list=eval_list)

    model = build_model(cfg, training=False)
    model.cuda()
    model.eval()
    sd = torch.load(weight_path, map_location="cpu")
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    md = model.state_dict()
    md.update({k: v for k, v in sd.items() if k in md})
    model.load_state_dict(md)

    gt_samples = []
    raw_pred_samples = []
    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.cuda()
            outputs = model(samples)
            scores = F.softmax(outputs["pred_logits"], -1)[:, :, 1][0].detach().cpu().numpy().tolist()
            pts = outputs["pred_points"][0].detach().cpu().numpy().tolist()
            gt_pts = targets[0]["point"].detach().cpu().numpy().tolist()
            sample_id = os.path.splitext(targets[0]["name"])[0]
            domain = domain_from_id(sample_id, domain_names)
            tensor = samples.tensors if hasattr(samples, "tensors") else samples
            height = float(tensor.shape[-2])
            width = float(tensor.shape[-1])
            gt_samples.append({"id": sample_id, "domain": domain, "size": [width, height], "points": gt_pts})
            raw_pred_samples.append({
                "id": sample_id,
                "domain": domain,
                "size": [width, height],
                "points": pts,
                "scores": scores,
            })
    return gt_samples, raw_pred_samples


def select_thresholds(gt_samples, raw_samples, candidates, domain_names, match_threshold, select_metric, default_threshold, thresholds, grid_size):
    threshold_map = {}
    diagnostics = {}
    domains = sorted(set([s["domain"] for s in gt_samples]) | set([d.lower() for d in domain_names]))
    for domain in domains:
        threshold_map[domain] = {}
        diagnostics[domain] = {}
        gt_domain = [s for s in gt_samples if s["domain"] == domain]
        raw_domain = [s for s in raw_samples if s["domain"] == domain]
        if not gt_domain:
            continue
        for bin_name in DENSITY_BINS:
            best_threshold = default_threshold
            best_value = None
            stats = []
            for threshold in candidates:
                trial_map = {domain: {b: default_threshold for b in DENSITY_BINS}}
                trial_map[domain][bin_name] = threshold
                pred_samples = filter_samples(raw_domain, trial_map, default_threshold)
                eval_thresholds = thresholds if select_metric == "f1" else []
                summary = summarize(gt_domain, pred_samples, eval_thresholds, grid_size)
                if select_metric == "f1":
                    value = summary["thresholds"][str(match_threshold)]["F1"]
                    better = best_value is None or value > best_value
                elif select_metric == "region_mae":
                    value = summary["region_MAE"]
                    better = best_value is None or value < best_value
                else:
                    value = summary["MAE"]
                    better = best_value is None or value < best_value
                if better:
                    best_value = value
                    best_threshold = threshold
                stats.append({
                    "threshold": threshold,
                    "MAE": summary["MAE"],
                    "MSE": summary["MSE"],
                    "region_MAE": summary["region_MAE"],
                    "region_MSE": summary["region_MSE"],
                    "pred_count": summary["pred_count"],
                    "F1@%d" % match_threshold: summary["thresholds"].get(str(match_threshold), {}).get("F1"),
                    "Precision@%d" % match_threshold: summary["thresholds"].get(str(match_threshold), {}).get("Precision"),
                    "Recall@%d" % match_threshold: summary["thresholds"].get(str(match_threshold), {}).get("Recall"),
                })
            threshold_map[domain][bin_name] = best_threshold
            diagnostics[domain][bin_name] = {
                "selected_threshold": best_threshold,
                "selected_metric": select_metric,
                "candidates": stats,
            }
    return threshold_map, diagnostics


def compact_samples(samples):
    return {"samples": [{"id": s["id"], "points": s["points"]} for s in samples]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weight", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--val-list", default="val.list")
    parser.add_argument("--test-list", default="test.list")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[6, 12, 24])
    parser.add_argument("--match-threshold", type=int, default=12)
    parser.add_argument("--select-metric", choices=["f1", "mae", "region_mae"], default="mae")
    parser.add_argument("--score-candidates", type=float, nargs="+",
                        default=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    parser.add_argument("--default-threshold", type=float, default=0.5)
    parser.add_argument("--density-base-threshold", type=float, default=0.25)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--domain-density-edges", action="store_true",
                        help="use per-domain density tertiles instead of global tertiles")
    parser.add_argument("--count-only", action="store_true",
                        help="skip localization matching and only report count/region metrics")
    parser.add_argument("--domain-names", nargs="+", default=["crag", "dpath", "glas", "pannuke", "consep"])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from config import cfg, merge_from_file

    cfg = merge_from_file(cfg, args.config)
    val_gt, val_raw = collect_predictions(cfg, args.config, args.weight, args.data_root, args.val_list, args.domain_names)
    density_edges = build_density_edges(val_raw, args.density_base_threshold, args.grid_size)
    val_raw = annotate_density_bins(val_raw, density_edges, args.density_base_threshold, args.grid_size, args.domain_density_edges)
    threshold_map, diagnostics = select_thresholds(
        val_gt, val_raw, args.score_candidates, args.domain_names, args.match_threshold,
        args.select_metric, args.default_threshold, args.thresholds, args.grid_size)
    report_thresholds = [] if args.count_only else args.thresholds
    val_pred = filter_samples(val_raw, threshold_map, args.default_threshold)
    val_summary = summarize(val_gt, val_pred, report_thresholds, args.grid_size)

    test_gt, test_raw = collect_predictions(cfg, args.config, args.weight, args.data_root, args.test_list, args.domain_names)
    test_raw = annotate_density_bins(test_raw, density_edges, args.density_base_threshold, args.grid_size, args.domain_density_edges)
    test_pred = filter_samples(test_raw, threshold_map, args.default_threshold)
    test_summary = summarize(test_gt, test_pred, report_thresholds, args.grid_size)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "thresholds.json").write_text(json.dumps({
        "threshold_map": threshold_map,
        "density_edges": density_edges,
        "diagnostics": diagnostics,
        "val_summary": val_summary,
    }, ensure_ascii=False, indent=2))
    (out / "gt.json").write_text(json.dumps(compact_samples(test_gt)))
    (out / "pred.json").write_text(json.dumps(compact_samples(test_pred)))
    (out / "summary.json").write_text(json.dumps(test_summary, ensure_ascii=False, indent=2))
    print(json.dumps({"threshold_map": threshold_map, "test_summary": test_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
