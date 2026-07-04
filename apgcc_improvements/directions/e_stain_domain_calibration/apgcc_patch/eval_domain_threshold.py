#!/usr/bin/env python3
"""Domain-aware threshold calibration for APGCC centroid predictions.

The script selects one score threshold per source domain on a validation list,
then evaluates the fixed domain thresholds on a test list.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def domain_from_id(sample_id, domain_names):
    stem = sample_id.lower()
    for name in domain_names:
        key = name.lower()
        if stem == key or stem.startswith(key + "_") or stem.startswith(key + "-"):
            return key
    return "unknown"


def count_tp(gt, pred, threshold):
    gt = np.asarray(gt, dtype=np.float32).reshape(-1, 2)
    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 2)
    if len(gt) == 0 or len(pred) == 0:
        return 0
    diff = pred[:, None, :] - gt[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    rows, cols = linear_sum_assignment(dist)
    return int((dist[rows, cols] <= threshold).sum())


def filter_samples(raw_samples, threshold_map, default_threshold):
    filtered = []
    for sample in raw_samples:
        threshold = threshold_map.get(sample["domain"], default_threshold)
        points = [p for p, s in zip(sample["points"], sample["scores"]) if s > threshold]
        filtered.append({"id": sample["id"], "domain": sample["domain"], "points": points})
    return filtered


def summarize(gt_samples, pred_samples, thresholds):
    errors = []
    total_gt = 0
    total_pred = 0
    tp_by_threshold = {int(t): 0 for t in thresholds}
    by_domain = {}

    for gt_sample, pred_sample in zip(gt_samples, pred_samples):
        domain = gt_sample.get("domain", "unknown")
        gt_pts = gt_sample["points"]
        pred_pts = pred_sample["points"]
        error = len(pred_pts) - len(gt_pts)
        errors.append(error)
        total_gt += len(gt_pts)
        total_pred += len(pred_pts)

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
            gt_samples.append({"id": sample_id, "domain": domain, "points": gt_pts})
            raw_pred_samples.append({"id": sample_id, "domain": domain, "points": pts, "scores": scores})
    return gt_samples, raw_pred_samples


def select_thresholds(gt_samples, raw_pred_samples, candidates, domain_names, match_threshold, select_metric):
    threshold_map = {}
    diagnostics = {}
    domains = sorted(set([s["domain"] for s in gt_samples]) | set([d.lower() for d in domain_names]))
    for domain in domains:
        gt_domain = [s for s in gt_samples if s["domain"] == domain]
        raw_domain = [s for s in raw_pred_samples if s["domain"] == domain]
        if not gt_domain:
            continue
        stats = []
        best_threshold = candidates[0]
        best_value = None
        for threshold in candidates:
            pred_domain = filter_samples(raw_domain, {domain: threshold}, threshold)
            summary = summarize(gt_domain, pred_domain, [match_threshold])
            value = summary["thresholds"][str(match_threshold)]["F1"] if select_metric == "f1" else summary["MAE"]
            is_better = (best_value is None or value > best_value) if select_metric == "f1" else (best_value is None or value < best_value)
            if is_better:
                best_value = value
                best_threshold = threshold
            stats.append({
                "threshold": threshold,
                "MAE": summary["MAE"],
                "MSE": summary["MSE"],
                "pred_count": summary["pred_count"],
                "gt_count": summary["gt_count"],
                "F1@%d" % match_threshold: summary["thresholds"][str(match_threshold)]["F1"],
                "Precision@%d" % match_threshold: summary["thresholds"][str(match_threshold)]["Precision"],
                "Recall@%d" % match_threshold: summary["thresholds"][str(match_threshold)]["Recall"],
            })
        threshold_map[domain] = best_threshold
        diagnostics[domain] = {"selected_threshold": best_threshold, "selected_metric": select_metric, "candidates": stats}
    return threshold_map, diagnostics


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
    parser.add_argument("--select-metric", choices=["f1", "mae"], default="f1")
    parser.add_argument("--score-candidates", type=float, nargs="+",
                        default=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    parser.add_argument("--domain-names", nargs="+", default=["crag", "dpath", "glas", "pannuke", "consep"])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    from config import cfg, merge_from_file

    cfg = merge_from_file(cfg, args.config)
    val_gt, val_raw = collect_predictions(cfg, args.config, args.weight, args.data_root,
                                          args.val_list, args.domain_names)
    threshold_map, diagnostics = select_thresholds(
        val_gt, val_raw, args.score_candidates, args.domain_names, args.match_threshold, args.select_metric)
    val_pred = filter_samples(val_raw, threshold_map, default_threshold=0.5)
    val_summary = summarize(val_gt, val_pred, args.thresholds)

    test_gt, test_raw = collect_predictions(cfg, args.config, args.weight, args.data_root,
                                            args.test_list, args.domain_names)
    test_pred = filter_samples(test_raw, threshold_map, default_threshold=0.5)
    test_summary = summarize(test_gt, test_pred, args.thresholds)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "thresholds.json").write_text(json.dumps({
        "threshold_map": threshold_map,
        "diagnostics": diagnostics,
        "val_summary": val_summary,
    }, ensure_ascii=False, indent=2))
    (out / "gt.json").write_text(json.dumps({"samples": [{"id": s["id"], "points": s["points"]} for s in test_gt]}))
    (out / "pred.json").write_text(json.dumps({"samples": [{"id": s["id"], "points": s["points"]} for s in test_pred]}))
    (out / "summary.json").write_text(json.dumps(test_summary, ensure_ascii=False, indent=2))
    print(json.dumps({"threshold_map": threshold_map, "test_summary": test_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
