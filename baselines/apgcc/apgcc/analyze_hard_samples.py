#!/usr/bin/env python3
"""Collect and visualize hard CoNIC samples for APGCC centroid outputs.

Inputs are team-unified centroid json files:
  {"samples": [{"id": str, "points": [[x,y], ...]}, ...]}

For each selected sample this script writes a 4-panel PNG:
  1) original + GT points
  2) original + Pred points
  3) TP / FP / FN overlay at a matching radius
  4) 4x4 regional count error heatmap
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def load_samples(path):
    with open(path) as f:
        data = json.load(f)
    return {s["id"]: np.asarray(s["points"], dtype=np.float64).reshape(-1, 2)
            for s in data.get("samples", [])}


def load_image_map(data_root, split="test.list"):
    data_root = Path(data_root)
    out = {}
    with open(data_root / split) as f:
        for line in f:
            if not line.strip():
                continue
            img_rel, _ = line.split()[:2]
            sid = Path(img_rel).stem
            out[sid] = data_root / img_rel
    return out


def match_points(gt, pred, threshold):
    if len(gt) == 0 and len(pred) == 0:
        return [], [], [], []
    if len(gt) == 0:
        return [], [], [], list(range(len(pred)))
    if len(pred) == 0:
        return [], list(range(len(gt))), [], []
    dist = cdist(gt, pred)
    rows, cols = linear_sum_assignment(dist)
    tp_gt, tp_pred = [], []
    for r, c in zip(rows, cols):
        if dist[r, c] <= threshold:
            tp_gt.append(int(r))
            tp_pred.append(int(c))
    tp_gt_set, tp_pred_set = set(tp_gt), set(tp_pred)
    fn = [i for i in range(len(gt)) if i not in tp_gt_set]
    fp = [i for i in range(len(pred)) if i not in tp_pred_set]
    return tp_gt, fn, tp_pred, fp


def source_of(sid, dataset_name, group_by_prefix):
    if group_by_prefix:
        return sid.split("_")[0]
    return dataset_name


def sample_metrics(sid, gt, pred, threshold, dataset_name, group_by_prefix):
    tp_gt, fn, tp_pred, fp = match_points(gt, pred, threshold)
    tp = len(tp_gt)
    prec = tp / (tp + len(fp) + 1e-8)
    rec = tp / (tp + len(fn) + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    err = len(pred) - len(gt)
    return {
        "id": sid,
        "source": source_of(sid, dataset_name, group_by_prefix),
        "gt": len(gt),
        "pred": len(pred),
        "err": err,
        "abs_err": abs(err),
        "sq_err": err * err,
        "tp12": tp,
        "fp12": len(fp),
        "fn12": len(fn),
        "precision12": prec,
        "recall12": rec,
        "f112": f1,
    }


def draw_points(draw, pts, color, r=3, outline=None):
    for x, y in pts:
        x, y = float(x), float(y)
        box = [x - r, y - r, x + r, y + r]
        draw.ellipse(box, fill=color, outline=outline or color)


def text(draw, xy, label, fill=(255, 255, 255)):
    # Keep default bitmap font to avoid host-specific font dependencies.
    pad = 3
    bbox = draw.textbbox(xy, label)
    bg = [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad]
    draw.rectangle(bg, fill=(0, 0, 0))
    draw.text(xy, label, fill=fill)


def panel_base(img):
    return img.convert("RGB").copy()


def regional_errors(gt, pred, w, h, grid=4):
    out = np.zeros((grid, grid), dtype=int)
    for pts, sign in [(gt, -1), (pred, 1)]:
        for x, y in pts:
            gx = min(grid - 1, max(0, int(float(x) / max(w, 1) * grid)))
            gy = min(grid - 1, max(0, int(float(y) / max(h, 1) * grid)))
            out[gy, gx] += sign
    return out


def overlay_heatmap(img, err_grid):
    out = img.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    grid = err_grid.shape[0]
    max_abs = max(1, int(np.max(np.abs(err_grid))))
    for gy in range(grid):
        for gx in range(grid):
            err = int(err_grid[gy, gx])
            x0, y0 = gx * w / grid, gy * h / grid
            x1, y1 = (gx + 1) * w / grid, (gy + 1) * h / grid
            if err < 0:
                # red: under-count / FN-dominant region
                alpha = int(55 + 145 * abs(err) / max_abs)
                fill = (255, 40, 40, alpha)
            elif err > 0:
                # blue: over-count / FP-dominant region
                alpha = int(55 + 145 * abs(err) / max_abs)
                fill = (40, 120, 255, alpha)
            else:
                fill = (0, 0, 0, 0)
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(255, 255, 255, 120))
            if err != 0:
                label = f"{err:+d}"
                cx, cy = (x0 + x1) / 2 - 7, (y0 + y1) / 2 - 5
                draw.text((cx, cy), label, fill=(255, 255, 255, 255))
    return out.convert("RGB")


def make_visual(sid, img_path, gt, pred, threshold, out_path):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    gt_panel = panel_base(img)
    d = ImageDraw.Draw(gt_panel)
    draw_points(d, gt, (0, 255, 0), r=2)
    text(d, (6, 6), f"GT: {len(gt)}")

    pred_panel = panel_base(img)
    d = ImageDraw.Draw(pred_panel)
    draw_points(d, pred, (255, 60, 60), r=2)
    text(d, (6, 6), f"Pred: {len(pred)}  Err: {len(pred)-len(gt):+d}")

    tp_gt, fn, tp_pred, fp = match_points(gt, pred, threshold)
    match_panel = panel_base(img)
    d = ImageDraw.Draw(match_panel)
    draw_points(d, pred[tp_pred] if tp_pred else [], (80, 220, 80), r=2)
    draw_points(d, pred[fp] if fp else [], (255, 70, 70), r=3)
    draw_points(d, gt[fn] if fn else [], (255, 210, 0), r=3, outline=(0, 0, 0))
    text(d, (6, 6), f"TP {len(tp_pred)} / FP {len(fp)} / FN {len(fn)} @ {threshold:g}px")

    heat_panel = overlay_heatmap(img, regional_errors(gt, pred, w, h, grid=4))
    d = ImageDraw.Draw(heat_panel)
    text(d, (6, 6), "4x4 regional pred-gt error")

    title_h = 30
    margin = 8
    canvas = Image.new("RGB", (w * 2 + margin * 3, h * 2 + margin * 3 + title_h), (20, 20, 20))
    d = ImageDraw.Draw(canvas)
    d.text((margin, 8), sid, fill=(255, 255, 255))
    positions = [
        (margin, title_h + margin),
        (w + margin * 2, title_h + margin),
        (margin, title_h + h + margin * 2),
        (w + margin * 2, title_h + h + margin * 2),
    ]
    for p, panel in zip(positions, [gt_panel, pred_panel, match_panel, heat_panel]):
        canvas.paste(panel, p)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_by_source(rows):
    out = []
    groups = defaultdict(list)
    for r in rows:
        groups[r["source"]].append(r)
    for src, vals in sorted(groups.items()):
        gt = sum(v["gt"] for v in vals)
        pred = sum(v["pred"] for v in vals)
        err = pred - gt
        mae = float(np.mean([abs(v["err"]) for v in vals]))
        mse = float(np.mean([v["sq_err"] for v in vals]))
        tp = sum(v["tp12"] for v in vals)
        fp = sum(v["fp12"] for v in vals)
        fn = sum(v["fn12"] for v in vals)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        out.append({
            "source": src, "n": len(vals), "gt": gt, "pred": pred,
            "total_err": err, "total_err_pct": err / gt * 100 if gt else 0,
            "mae": mae, "mse": mse, "precision12": prec, "recall12": rec, "f112": f1,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/data1/llx/CoNICdata")
    ap.add_argument("--split", default="test.list")
    ap.add_argument("--base-dir", default="/home/lixinli/Pathology-Cell-Counting/baselines/APGCC/apgcc/output/CoNIC_unified")
    ap.add_argument("--dataset-name", default=None, help="Name used as the source group when --group-by-prefix is not set")
    ap.add_argument("--group-by-prefix", action="store_true", help="Group source by id prefix before '_' (useful for CoNIC)")
    ap.add_argument("--eval-dir", default="centroid_eval", help="baseline eval dir, e.g. centroid_eval")
    ap.add_argument("--calib-dir", default="centroid_eval_thr0.30", help="comparison/calibrated eval dir")
    ap.add_argument("--out-dir", default="/home/lixinli/Pathology-Cell-Counting/baselines/APGCC/apgcc/output/CoNIC_unified/hard_samples_analysis")
    ap.add_argument("--threshold", type=float, default=12.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--vis-limit", type=int, default=32)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    dataset_name = args.dataset_name or base_dir.name.replace("_unified", "")
    image_map = load_image_map(args.data_root, args.split)

    gt = load_samples(base_dir / args.eval_dir / "gt.json")
    pred = load_samples(base_dir / args.eval_dir / "pred.json")
    pred_cal = load_samples(base_dir / args.calib_dir / "pred.json")
    ids = sorted(set(gt) & set(pred))

    rows = []
    for sid in ids:
        r = sample_metrics(sid, gt[sid], pred[sid], args.threshold, dataset_name, args.group_by_prefix)
        if sid in pred_cal:
            rc = sample_metrics(sid, gt[sid], pred_cal[sid], args.threshold, dataset_name, args.group_by_prefix)
            r.update({
                "calib_pred": rc["pred"],
                "calib_err": rc["err"],
                "calib_abs_err": rc["abs_err"],
                "calib_sq_err": rc["sq_err"],
                "calib_recall12": rc["recall12"],
                "calib_f112": rc["f112"],
                "abs_err_delta_050_minus_030": r["abs_err"] - rc["abs_err"],
            })
        else:
            r.update({
                "calib_pred": "", "calib_err": "", "calib_abs_err": "",
                "calib_sq_err": "", "calib_recall12": "", "calib_f112": "",
                "abs_err_delta_050_minus_030": "",
            })
        rows.append(r)

    fields = ["id", "source", "gt", "pred", "err", "abs_err", "sq_err", "tp12", "fp12", "fn12",
              "precision12", "recall12", "f112", "calib_pred", "calib_err", "calib_abs_err",
              "calib_sq_err", "calib_recall12", "calib_f112", "abs_err_delta_050_minus_030"]
    rows_sorted = sorted(rows, key=lambda x: (x["abs_err"], x["gt"]), reverse=True)
    write_csv(out_dir / "hard_samples_all.csv", rows_sorted, fields)

    selected = []
    reasons = defaultdict(list)

    def add_many(items, reason):
        for r in items:
            sid = r["id"]
            if sid not in selected:
                selected.append(sid)
            reasons[sid].append(reason)

    add_many(rows_sorted[:args.top_k], "top_abs_error_thr0.50")
    add_many(sorted(rows, key=lambda x: x["err"])[:args.top_k], "top_undercount_thr0.50")
    add_many(sorted(rows, key=lambda x: x["calib_abs_err"] if x["calib_abs_err"] != "" else -1, reverse=True)[:args.top_k],
             "persistent_hard_thr0.30")
    add_many(sorted(rows, key=lambda x: x["abs_err_delta_050_minus_030"] if x["abs_err_delta_050_minus_030"] != "" else -1, reverse=True)[:args.top_k],
             "calibration_helped_most")
    for src in sorted(set(r["source"] for r in rows)):
        src_rows = [r for r in rows if r["source"] == src]
        add_many(sorted(src_rows, key=lambda x: x["abs_err"], reverse=True)[:5], f"top5_source_{src}")

    selected_rows = []
    row_by_id = {r["id"]: r for r in rows}
    for sid in selected:
        rr = dict(row_by_id[sid])
        rr["reason"] = ";".join(reasons[sid])
        selected_rows.append(rr)
    write_csv(out_dir / "hard_samples_selected.csv", selected_rows, fields + ["reason"])

    src_summary = summarize_by_source(rows)
    write_csv(out_dir / "source_summary_thr0.50.csv", src_summary,
              ["source", "n", "gt", "pred", "total_err", "total_err_pct", "mae", "mse", "precision12", "recall12", "f112"])

    # Visualize a bounded, deduplicated subset to keep the artifact set readable.
    vis_ids = selected[:args.vis_limit]
    visual_manifest = []
    for sid in vis_ids:
        if sid not in image_map:
            continue
        out_path = out_dir / "visuals" / f"{sid}.png"
        make_visual(sid, image_map[sid], gt[sid], pred[sid], args.threshold, out_path)
        rr = row_by_id[sid]
        visual_manifest.append({
            "id": sid, "source": rr["source"], "gt": rr["gt"], "pred": rr["pred"],
            "err": rr["err"], "abs_err": rr["abs_err"], "fp12": rr["fp12"], "fn12": rr["fn12"],
            "visual": str(out_path), "reason": ";".join(reasons[sid])
        })
    write_csv(out_dir / "visual_manifest.csv", visual_manifest,
              ["id", "source", "gt", "pred", "err", "abs_err", "fp12", "fn12", "visual", "reason"])

    top_under = sorted(rows, key=lambda x: x["err"])[:10]
    persistent = sorted(rows, key=lambda x: x["calib_abs_err"] if x["calib_abs_err"] != "" else -1, reverse=True)[:10]
    helped = sorted(rows, key=lambda x: x["abs_err_delta_050_minus_030"] if x["abs_err_delta_050_minus_030"] != "" else -1, reverse=True)[:10]

    def md_table(rs, cols):
        lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
        for r in rs:
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    vals.append(f"{v:.3f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    overall_gt = sum(r["gt"] for r in rows)
    overall_pred = sum(r["pred"] for r in rows)
    overall_err = overall_pred - overall_gt
    overall_mae = np.mean([abs(r["err"]) for r in rows])
    overall_mse = np.mean([r["sq_err"] for r in rows])
    md = []
    md.append("# CoNIC APGCC Hard Sample Analysis\n")
    md.append(f"Input: `{base_dir / args.eval_dir}`; comparison: `{base_dir / args.calib_dir}`.\n")
    md.append(f"Matching radius for TP/FP/FN visualization: `{args.threshold:g}px`.\n")
    md.append("## Overall Counting\n")
    md.append(f"- Samples: {len(rows)}\n- Total GT: {overall_gt}\n- Total Pred: {overall_pred} ({overall_err / overall_gt * 100:+.2f}%)\n- MAE: {overall_mae:.2f}\n- MSE: {overall_mse:.2f}\n- RMSE: {math.sqrt(overall_mse):.2f}\n")
    md.append("## Source Summary\n")
    md.append(md_table(src_summary, ["source", "n", "gt", "pred", "total_err", "total_err_pct", "mae", "mse", "recall12", "f112"]))
    md.append("\n## Top Undercount Samples at Threshold 0.50\n")
    md.append(md_table(top_under, ["id", "source", "gt", "pred", "err", "abs_err", "fn12", "fp12", "recall12"]))
    md.append("\n## Persistent Hard Samples after Threshold 0.30\n")
    md.append(md_table(persistent, ["id", "source", "gt", "pred", "err", "calib_pred", "calib_err", "calib_abs_err", "calib_recall12"]))
    md.append("\n## Samples Most Helped by Lowering Threshold 0.50 -> 0.30\n")
    md.append(md_table(helped, ["id", "source", "gt", "pred", "err", "calib_pred", "calib_err", "abs_err_delta_050_minus_030"]))
    md.append("\n## Interpretation\n")
    md.append("- The dominant hard cases are systematic under-counts in dense `pannuke`, `glas`, and `dpath` patches.\n")
    md.append("- Lowering the threshold recovers many predictions, but the persistent-hard list shows that confidence calibration alone does not fully solve dense-region recall.\n")
    md.append("- The visual panels use yellow for FN, red for FP, and green for TP. The 4x4 heatmap shows regional `pred - gt`; red cells are local under-counts and blue cells are local over-counts.\n")
    md.append("- These samples should be reused for K8 and K8+calibration qualitative comparison.\n")
    md.append("\n## Files\n")
    md.append("- `hard_samples_all.csv`: all test samples sorted by absolute count error.\n")
    md.append("- `hard_samples_selected.csv`: selected hard sample set with selection reason.\n")
    md.append("- `source_summary_thr0.50.csv`: source-level metric summary.\n")
    md.append("- `visual_manifest.csv`: generated visualization list.\n")
    md.append("- `visuals/*.png`: 4-panel visualizations.\n")
    (out_dir / "analysis.md").write_text("\n".join(md))

    print(f"Wrote {out_dir}")
    print(f"Selected {len(selected_rows)} hard samples; visualized {len(visual_manifest)}")


if __name__ == "__main__":
    main()
