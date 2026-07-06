#!/usr/bin/env python3
"""Create report-ready hard-sample figures for APGCC centroid outputs.

The earlier contact sheets are useful for screening, but they compress too
many samples into one canvas. This script makes paper/report figures:

  - one high-resolution full 2x2 figure per hard sample
  - one zoomed local-failure figure per hard sample
  - one compact top-N sheet per dataset

Inputs are the existing hard-sample CSV files and centroid eval JSON files.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


DEFAULTS = {
    "CoNIC": {
        "data_root": "/data1/llx/CoNICdata",
        "base_dir": "output/CoNIC_unified",
        "split": "test.list",
    },
    "BCData": {
        "data_root": "/data1/llx/BCData",
        "base_dir": "output/BCData_unified",
        "split": "test.list",
    },
    "MoNuSeg": {
        "data_root": "/data1/llx/MoNuSegdata",
        "base_dir": "output/MoNuSeg_unified",
        "split": "test.list",
    },
}


COLORS = {
    "gt": (0, 205, 125),
    "pred": (240, 70, 70),
    "tp": (20, 190, 90),
    "fp": (230, 40, 45),
    "fn": (255, 190, 0),
    "dark": (24, 24, 24),
    "muted": (94, 94, 94),
    "panel_bg": (255, 255, 255),
    "border": (35, 35, 35),
}


def font(size, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


FONT_TITLE = font(34, bold=True)
FONT_SUBTITLE = font(24, bold=True)
FONT_TEXT = font(22)
FONT_SMALL = font(18)


def safe_name(s):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)


def load_samples(path):
    with open(path) as f:
        data = json.load(f)
    return {
        s["id"]: np.asarray(s["points"], dtype=np.float64).reshape(-1, 2)
        for s in data.get("samples", [])
    }


def load_image_map(data_root, split):
    data_root = Path(data_root)
    out = {}
    with open(data_root / split) as f:
        for line in f:
            if not line.strip():
                continue
            img_rel, _ = line.split()[:2]
            out[Path(img_rel).stem] = data_root / img_rel
    return out


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


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
    tp_gt_set = set(tp_gt)
    tp_pred_set = set(tp_pred)
    fn = [i for i in range(len(gt)) if i not in tp_gt_set]
    fp = [i for i in range(len(pred)) if i not in tp_pred_set]
    return tp_gt, fn, tp_pred, fp


def regional_errors(gt, pred, w, h, grid=4):
    out = np.zeros((grid, grid), dtype=int)
    for pts, sign in ((gt, -1), (pred, 1)):
        for x, y in pts:
            gx = min(grid - 1, max(0, int(float(x) / max(w, 1) * grid)))
            gy = min(grid - 1, max(0, int(float(y) / max(h, 1) * grid)))
            out[gy, gx] += sign
    return out


def resize_points(points, scale, x0=0, y0=0):
    if len(points) == 0:
        return points.reshape(-1, 2)
    out = points.copy()
    out[:, 0] = (out[:, 0] - x0) * scale
    out[:, 1] = (out[:, 1] - y0) * scale
    return out


def fit_image(img, target):
    w, h = img.size
    scale = target / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target, target), (255, 255, 255))
    ox, oy = (target - nw) // 2, (target - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, scale, ox, oy


def draw_label(draw, xy, text, title=False, fill=COLORS["dark"]):
    draw.text(xy, text, fill=fill, font=FONT_SUBTITLE if title else FONT_TEXT)


def draw_marker(draw, x, y, color, r, width=3, fill=None):
    box = [x - r, y - r, x + r, y + r]
    if fill is not None:
        draw.ellipse(box, fill=fill, outline=(0, 0, 0), width=width)
    else:
        draw.ellipse(box, outline=(0, 0, 0), width=width + 2)
        draw.ellipse(box, outline=color, width=width)


def draw_points(draw, pts, color, r=7, width=3, fill_alpha=False):
    for x, y in pts:
        if fill_alpha:
            draw_marker(draw, float(x), float(y), color, r, width=width, fill=color)
        else:
            draw_marker(draw, float(x), float(y), color, r, width=width)


def add_panel_title(panel, title, subtitle=None):
    w, h = panel.size
    header_h = 58 if subtitle is None else 88
    out = Image.new("RGB", (w, h + header_h), COLORS["panel_bg"])
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, w - 1, h + header_h - 1], outline=COLORS["border"], width=2)
    draw_label(draw, (18, 12), title, title=True)
    if subtitle:
        draw.text((18, 50), subtitle, fill=COLORS["muted"], font=FONT_SMALL)
    out.paste(panel, (0, header_h))
    return out


def make_overlay_panel(img, points, target, title, subtitle, color, radius=7):
    panel, scale, ox, oy = fit_image(img, target)
    pts = resize_points(points, scale)
    pts[:, 0] += ox
    pts[:, 1] += oy
    draw = ImageDraw.Draw(panel)
    draw_points(draw, pts, color, r=radius)
    return add_panel_title(panel, title, subtitle)


def make_match_panel(img, gt, pred, threshold, target, title="TP / FP / FN"):
    panel, scale, ox, oy = fit_image(img, target)
    tp_gt, fn, tp_pred, fp = match_points(gt, pred, threshold)
    pred_scaled = resize_points(pred, scale)
    gt_scaled = resize_points(gt, scale)
    pred_scaled[:, 0] += ox
    pred_scaled[:, 1] += oy
    gt_scaled[:, 0] += ox
    gt_scaled[:, 1] += oy
    draw = ImageDraw.Draw(panel)
    draw_points(draw, pred_scaled[tp_pred] if tp_pred else np.empty((0, 2)), COLORS["tp"], r=6)
    draw_points(draw, pred_scaled[fp] if fp else np.empty((0, 2)), COLORS["fp"], r=8)
    draw_points(draw, gt_scaled[fn] if fn else np.empty((0, 2)), COLORS["fn"], r=8)
    subtitle = f"green=TP  red=FP  yellow=FN  radius={threshold:g}px"
    return add_panel_title(panel, title, subtitle), (len(tp_pred), len(fp), len(fn))


def make_heat_panel(img, gt, pred, target):
    panel, scale, ox, oy = fit_image(img, target)
    w, h = img.size
    err_grid = regional_errors(gt, pred, w, h, grid=4)
    draw = ImageDraw.Draw(panel, "RGBA")
    max_abs = max(1, int(np.max(np.abs(err_grid))))
    # Draw over the actual image area only.
    iw, ih = int(round(w * scale)), int(round(h * scale))
    for gy in range(4):
        for gx in range(4):
            err = int(err_grid[gy, gx])
            x0 = ox + gx * iw / 4
            y0 = oy + gy * ih / 4
            x1 = ox + (gx + 1) * iw / 4
            y1 = oy + (gy + 1) * ih / 4
            if err < 0:
                alpha = int(70 + 130 * abs(err) / max_abs)
                fill = (235, 40, 40, alpha)
            elif err > 0:
                alpha = int(70 + 130 * abs(err) / max_abs)
                fill = (35, 105, 230, alpha)
            else:
                fill = (255, 255, 255, 15)
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(255, 255, 255, 210), width=2)
            label = f"{err:+d}" if err else "0"
            bb = draw.textbbox((0, 0), label, font=FONT_SUBTITLE)
            tx = (x0 + x1 - (bb[2] - bb[0])) / 2
            ty = (y0 + y1 - (bb[3] - bb[1])) / 2
            draw.rectangle([tx - 6, ty - 4, tx + bb[2] - bb[0] + 6, ty + bb[3] - bb[1] + 4], fill=(0, 0, 0, 145))
            draw.text((tx, ty), label, fill=(255, 255, 255), font=FONT_SUBTITLE)
    return add_panel_title(panel, "Regional count error", "4x4 grid, value = pred - gt")


def crop_worst_region(img, gt, pred, pad_ratio=0.20):
    w, h = img.size
    grid = regional_errors(gt, pred, w, h, grid=4)
    gy, gx = np.unravel_index(np.argmax(np.abs(grid)), grid.shape)
    cell_w, cell_h = w / 4, h / 4
    pad_x, pad_y = cell_w * pad_ratio, cell_h * pad_ratio
    x0 = max(0, int(gx * cell_w - pad_x))
    y0 = max(0, int(gy * cell_h - pad_y))
    x1 = min(w, int((gx + 1) * cell_w + pad_x))
    y1 = min(h, int((gy + 1) * cell_h + pad_y))
    crop = img.crop((x0, y0, x1, y1))
    gt_mask = (gt[:, 0] >= x0) & (gt[:, 0] < x1) & (gt[:, 1] >= y0) & (gt[:, 1] < y1) if len(gt) else []
    pred_mask = (pred[:, 0] >= x0) & (pred[:, 0] < x1) & (pred[:, 1] >= y0) & (pred[:, 1] < y1) if len(pred) else []
    gt_crop = gt[gt_mask].copy() if len(gt) else np.empty((0, 2))
    pred_crop = pred[pred_mask].copy() if len(pred) else np.empty((0, 2))
    if len(gt_crop):
        gt_crop[:, 0] -= x0
        gt_crop[:, 1] -= y0
    if len(pred_crop):
        pred_crop[:, 0] -= x0
        pred_crop[:, 1] -= y0
    return crop, gt_crop, pred_crop, (gx, gy, int(grid[gy, gx]))


def paste_grid(panels, cols, gap=26, bg=(255, 255, 255)):
    rows = int(np.ceil(len(panels) / cols))
    pw = max(p.size[0] for p in panels)
    ph = max(p.size[1] for p in panels)
    out = Image.new("RGB", (cols * pw + (cols + 1) * gap, rows * ph + (rows + 1) * gap), bg)
    for i, p in enumerate(panels):
        x = gap + (i % cols) * (pw + gap)
        y = gap + (i // cols) * (ph + gap)
        out.paste(p, (x, y))
    return out


def make_full_figure(dataset, sid, img, gt, pred, threshold, out_path, panel_size):
    gt_panel = make_overlay_panel(img, gt, panel_size, "Ground truth", f"GT count = {len(gt)}", COLORS["gt"])
    pred_panel = make_overlay_panel(img, pred, panel_size, "APGCC prediction", f"Pred count = {len(pred)}, error = {len(pred)-len(gt):+d}", COLORS["pred"])
    match_panel, counts = make_match_panel(img, gt, pred, threshold, panel_size)
    heat_panel = make_heat_panel(img, gt, pred, panel_size)

    body = paste_grid([gt_panel, pred_panel, match_panel, heat_panel], cols=2)
    header_h = 94
    out = Image.new("RGB", (body.size[0], body.size[1] + header_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    draw.text((26, 18), f"{dataset} hard sample: {sid}", fill=COLORS["dark"], font=FONT_TITLE)
    tp, fp, fn = counts
    summary = f"GT {len(gt)} | Pred {len(pred)} | Err {len(pred)-len(gt):+d} | TP {tp} | FP {fp} | FN {fn}"
    draw.text((26, 58), summary, fill=COLORS["muted"], font=FONT_TEXT)
    out.paste(body, (0, header_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, dpi=(300, 300))
    return out


def make_zoom_figure(dataset, sid, img, gt, pred, threshold, out_path, panel_size):
    crop, gt_crop, pred_crop, region = crop_worst_region(img, gt, pred)
    gx, gy, local_err = region
    gt_panel = make_overlay_panel(crop, gt_crop, panel_size, "GT zoom", f"worst cell ({gx+1},{gy+1}), GT={len(gt_crop)}", COLORS["gt"], radius=9)
    pred_panel = make_overlay_panel(crop, pred_crop, panel_size, "Pred zoom", f"Pred={len(pred_crop)}, local err={len(pred_crop)-len(gt_crop):+d}", COLORS["pred"], radius=9)
    match_panel, counts = make_match_panel(crop, gt_crop, pred_crop, threshold, panel_size, title="Local TP / FP / FN")
    body = paste_grid([gt_panel, pred_panel, match_panel], cols=3)
    header_h = 94
    out = Image.new("RGB", (body.size[0], body.size[1] + header_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    draw.text((26, 18), f"{dataset}: zoomed hard region in {sid}", fill=COLORS["dark"], font=FONT_TITLE)
    tp, fp, fn = counts
    summary = f"Selected by largest 4x4 absolute error: pred-gt={local_err:+d} | local TP {tp}, FP {fp}, FN {fn}"
    draw.text((26, 58), summary, fill=COLORS["muted"], font=FONT_TEXT)
    out.paste(body, (0, header_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, dpi=(300, 300))
    return out


def make_dataset_sheet(dataset, sample_figures, out_path):
    # Use already rendered zoom figures, scaled to a readable width.
    thumbs = []
    target_w = 1200
    for fig in sample_figures:
        scale = target_w / fig.size[0]
        thumb = fig.resize((target_w, int(fig.size[1] * scale)), Image.Resampling.LANCZOS)
        thumbs.append(thumb)
    sheet = paste_grid(thumbs, cols=1, gap=34)
    header_h = 78
    out = Image.new("RGB", (sheet.size[0], sheet.size[1] + header_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    draw.text((28, 20), f"{dataset} report-ready hard samples", fill=COLORS["dark"], font=FONT_TITLE)
    out.paste(sheet, (0, header_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, dpi=(300, 300))


def make_representative_sheet(manifest, out_path):
    first_by_dataset = {}
    for row in manifest:
        first_by_dataset.setdefault(row["dataset"], row)
    thumbs = []
    target_w = 1800
    for dataset in ("CoNIC", "BCData", "MoNuSeg"):
        row = first_by_dataset.get(dataset)
        if not row:
            continue
        fig = Image.open(row["zoom_figure"]).convert("RGB")
        scale = target_w / fig.size[0]
        thumb = fig.resize((target_w, int(fig.size[1] * scale)), Image.Resampling.LANCZOS)
        thumbs.append(thumb)
    if not thumbs:
        return
    sheet = paste_grid(thumbs, cols=1, gap=38)
    header_h = 88
    out = Image.new("RGB", (sheet.size[0], sheet.size[1] + header_h), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    draw.text((30, 22), "Representative APGCC hard samples across datasets", fill=COLORS["dark"], font=FONT_TITLE)
    out.paste(sheet, (0, header_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, dpi=(300, 300))


def process_dataset(dataset, cfg, root, out_root, top_n, threshold, panel_size):
    base_dir = root / cfg["base_dir"]
    selected_csv = base_dir / "hard_samples_analysis" / "hard_samples_selected.csv"
    gt_json = base_dir / "centroid_eval" / "gt.json"
    pred_json = base_dir / "centroid_eval" / "pred.json"
    selected = read_csv(selected_csv)[:top_n]
    gt = load_samples(gt_json)
    pred = load_samples(pred_json)
    image_map = load_image_map(cfg["data_root"], cfg["split"])

    report_dir = out_root / dataset
    zoom_figures = []
    manifest = []
    for row in selected:
        sid = row["id"]
        if sid not in image_map or sid not in gt or sid not in pred:
            continue
        img = Image.open(image_map[sid]).convert("RGB")
        name = safe_name(sid)
        full_path = report_dir / "full" / f"{name}_full.png"
        zoom_path = report_dir / "zoom" / f"{name}_zoom.png"
        full = make_full_figure(dataset, sid, img, gt[sid], pred[sid], threshold, full_path, panel_size)
        zoom = make_zoom_figure(dataset, sid, img, gt[sid], pred[sid], threshold, zoom_path, panel_size)
        zoom_figures.append(zoom)
        manifest.append({
            "dataset": dataset,
            "id": sid,
            "gt": row.get("gt", ""),
            "pred": row.get("pred", ""),
            "err": row.get("err", ""),
            "full_figure": str(full_path),
            "zoom_figure": str(zoom_path),
        })
        # Keep a reference so linters do not complain when run manually.
        _ = full

    if zoom_figures:
        make_dataset_sheet(dataset, zoom_figures, report_dir / f"{dataset}_top{len(zoom_figures)}_zoom_sheet.png")

    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/lixinli/Pathology-Cell-Counting/baselines/APGCC/apgcc")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--datasets", nargs="+", default=["CoNIC", "BCData", "MoNuSeg"], choices=sorted(DEFAULTS))
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=12.0)
    ap.add_argument("--panel-size", type=int, default=820)
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_dir) if args.out_dir else root / "output" / "hard_samples_three_datasets" / "report_figures"
    all_manifest = []
    for dataset in args.datasets:
        all_manifest.extend(process_dataset(dataset, DEFAULTS[dataset], root, out_root, args.top_n, args.threshold, args.panel_size))

    manifest_path = out_root / "report_figure_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        fields = ["dataset", "id", "gt", "pred", "err", "full_figure", "zoom_figure"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_manifest)
    make_representative_sheet(all_manifest, out_root / "three_dataset_representative_zoom.png")
    print(f"Wrote {out_root}")
    print(f"Figures: {len(all_manifest)} samples")


if __name__ == "__main__":
    main()
