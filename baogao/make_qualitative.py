#!/usr/bin/env python3
"""Qualitative TP/FP/FN overlays for the baseline-reproduction report.
Left: a dense CoNIC patch (under-count / missed detections).
Right: a MoNuSeg crop (over-count / duplicate responses).
Matching: Hungarian on Euclidean distance, radius 12 px (same rule as eval_centroid)."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from PIL import Image

OUT = "/home/lixinli/Pathology-Cell-Counting/baogao/figures"
EVALDIR = "/home/lixinli/Pathology-Cell-Counting/baselines/APGCC/apgcc/output"
RADIUS = 12
GREEN, RED, YELLOW = "#2ca02c", "#d62728", "#ffd400"

def load(ds):
    g = {s["id"]: np.array(s["points"], float).reshape(-1, 2)
         for s in json.load(open(f"{EVALDIR}/{ds}_finetune/centroid_eval/gt.json"))["samples"]}
    p = {s["id"]: np.array(s["points"], float).reshape(-1, 2)
         for s in json.load(open(f"{EVALDIR}/{ds}_finetune/centroid_eval/pred.json"))["samples"]}
    return g, p

def match(gt, pred, r=RADIUS):
    if len(gt) == 0 or len(pred) == 0:
        return np.zeros(len(pred), bool), np.zeros(len(gt), bool)
    D = np.linalg.norm(gt[:, None, :] - pred[None, :, :], axis=2)  # [G,P]
    big = 1e6
    C = np.where(D <= r, D, big)
    ri, ci = linear_sum_assignment(C)
    pred_tp = np.zeros(len(pred), bool); gt_matched = np.zeros(len(gt), bool)
    for i, j in zip(ri, ci):
        if C[i, j] < big:
            pred_tp[j] = True; gt_matched[i] = True
    return pred_tp, gt_matched

def best_window(pts, W, H, win):
    """center a win x win crop on the densest region (median of points)."""
    if len(pts) == 0:
        return 0, 0
    cx, cy = np.median(pts[:, 0]), np.median(pts[:, 1])
    x0 = int(np.clip(cx - win/2, 0, max(0, W - win)))
    y0 = int(np.clip(cy - win/2, 0, max(0, H - win)))
    return x0, y0

def panel(ax, img_path, gt, pred, win, title):
    im = np.array(Image.open(img_path).convert("RGB"))
    H, W = im.shape[:2]
    pred_tp, gt_m = match(gt, pred)
    x0, y0 = best_window(gt, W, H, win)
    crop = im[y0:y0+win, x0:x0+win]
    ax.imshow(crop)
    def insel(P):
        return (P[:, 0] >= x0) & (P[:, 0] < x0+win) & (P[:, 1] >= y0) & (P[:, 1] < y0+win)
    # FN (missed GT)
    m = insel(gt) & (~gt_m)
    ax.scatter(gt[m, 0]-x0, gt[m, 1]-y0, s=46, facecolors='none', edgecolors=YELLOW, linewidths=1.6, label="FN (missed GT)")
    # TP
    m = insel(pred) & pred_tp
    ax.scatter(pred[m, 0]-x0, pred[m, 1]-y0, s=22, c=GREEN, marker='o', label="TP")
    # FP
    m = insel(pred) & (~pred_tp)
    ax.scatter(pred[m, 0]-x0, pred[m, 1]-y0, s=40, c=RED, marker='x', linewidths=1.8, label="FP (spurious)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=11)
    return len(gt), len(pred), int(pred_tp.sum())

fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))

# --- CoNIC: dense, under-count ---
gC, pC = load("CoNIC")
cid = "glas_12-0002"
g, p, tp = panel(axes[0], f"/data1/llx/CoNICdata/test/{cid}.png", gC[cid], pC[cid], 150,
      f"CoNIC ({cid})\nGT={len(gC[cid])}, Pred={len(pC[cid])}  -> under-count")

# --- MoNuSeg: pick the most over-counted image ---
gM, pM = load("MoNuSeg")
mid = max(gM, key=lambda k: len(pM.get(k, [])) - len(gM[k]))
panel(axes[1], f"/data1/llx/MoNuSegdata/test/{mid}.png", gM[mid], pM.get(mid, np.empty((0,2))), 300,
      f"MoNuSeg ({mid[:18]}...)\nGT={len(gM[mid])}, Pred={len(pM[mid])}  -> over-count")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
fig.suptitle(f"APGCC baseline qualitative results (Hungarian match @ {RADIUS}px)", fontsize=12, fontweight="bold")
plt.tight_layout(rect=[0, 0.04, 1, 0.97])
for _ext in ("png", "pdf"): plt.savefig(f"{OUT}/fig_qualitative.{_ext}", dpi=200, bbox_inches="tight")
print("saved", f"{OUT}/fig_qualitative.png")
print("CoNIC", cid, "GT/Pred/TP=", g, p, tp)
print("MoNuSeg", mid, "GT/Pred=", len(gM[mid]), len(pM[mid]))
