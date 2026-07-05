#!/usr/bin/env python3
"""Baseline detection-result figure for the reproduction report.
3 rows (BCData / MoNuSeg / CoNIC) x 2 cols (GT points | APGCC predicted points),
on representative full test images, to show APGCC actually localizes cells
(complements the failure-mode crops in fig_qualitative)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = "/home/lixinli/Pathology-Cell-Counting/baogao/figures"
EVALDIR = "/home/lixinli/Pathology-Cell-Counting/baselines/APGCC/apgcc/output"
GREEN, RED = "#2ca02c", "#d62728"

# (display name, eval dir, image root, image id, file ext)
ROWS = [
    ("BCData (Ki-67 IHC)",      "BCData_finetune",  "/data1/llx/BCData/test",     "84",            "png"),
    ("MoNuSeg (H&E)",           "MoNuSeg_finetune", "/data1/llx/MoNuSegdata/test", "TCGA-A6-6782-01A-01-BS1", "png"),
    ("CoNIC (H&E colon)",       "CoNIC_finetune",   "/data1/llx/CoNICdata/test",  "dpath_33-0011", "png"),
]

def pts(ddir, kind, iid):
    data = json.load(open(f"{EVALDIR}/{ddir}/centroid_eval/{kind}.json"))["samples"]
    for s in data:
        if s["id"] == iid:
            return np.array(s["points"], float).reshape(-1, 2)
    return np.empty((0, 2))

fig, axes = plt.subplots(3, 2, figsize=(8.4, 12.2))
for r, (name, ddir, root, iid, ext) in enumerate(ROWS):
    im = np.array(Image.open(f"{root}/{iid}.{ext}").convert("RGB"))
    g = pts(ddir, "gt", iid)
    p = pts(ddir, "pred", iid)
    for c, (P, col, lab) in enumerate([(g, GREEN, "GT"), (p, RED, "APGCC pred")]):
        ax = axes[r, c]
        ax.imshow(im)
        s = 10 if len(P) > 200 else 18
        ax.scatter(P[:, 0], P[:, 1], s=s, facecolors="none", edgecolors=col, linewidths=1.0)
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(name, fontsize=12, fontweight="bold")
        ax.set_title(f"{lab}: {len(P)}", fontsize=11, color=col if c else "black")

fig.suptitle("APGCC baseline detection results: ground-truth vs predicted cell centers",
             fontsize=13, fontweight="bold", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.985])
for ext in ("png", "pdf"):
    plt.savefig(f"{OUT}/fig_baseline_detection.{ext}", dpi=170, bbox_inches="tight")
print("saved fig_baseline_detection; counts:")
for name, ddir, root, iid, ext in ROWS:
    print(f"  {name} {iid}: GT={len(pts(ddir,'gt',iid))} Pred={len(pts(ddir,'pred',iid))}")
