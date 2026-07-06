#!/usr/bin/env python3
"""ADE inference-based visualizations (BDC-style), into baogao/figures/:
  fig_ADE_qualitative_scatter                     3x3 grid: img+GT / baseline / ADE (TP/FP/FN)
  fig_ADE_density_heatmap_{CoNIC,BCData,MoNuSeg}  img / GT density / ADE density / residual
  fig_ADE_region_analysis_{CoNIC,BCData,MoNuSeg}  4x4 region counts + full-test density-error scatter
Predictions come from the persisted per-point score dumps, thresholded at each model's
MAE-optimal threshold (baseline / ADE) — the same calibrated operating point as the tables.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
from PIL import Image

ROOT = "/home/lixinli/Pathology-Cell-Counting"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/figures"; os.makedirs(OUT, exist_ok=True)
GREEN, RED, YELLOW, GREY = "#2ca02c", "#d62728", "#ffd400", "#8a8a8a"
RADIUS = 12

# dataset -> (base_dump, base_thr, ade_dump, ade_thr, img_dir, size, sigma, rep_id, nice)
DS = {
 "CoNIC": (f"{ROOT}/baselines/APGCC/apgcc/output/CoNIC_finetune/scores_test.json", 0.45,
           f"{ROOT}/improved/APGCC_ADE/apgcc/output/CoNIC_DE_stain_plain/scores_test.json", 0.30,
           "/data1/llx/CoNICdata/test", 256, 6, "glas_12-0009", "CoNIC (H&E colon)"),
 "BCData":(f"{ROOT}/baselines/APGCC/apgcc/output/BCData_finetune/scores_test.json", 0.45,
           f"{ROOT}/improved/APGCC_ADE/apgcc/output/BCData_ADE_dcnv2_edge_stain/scores_test.json", 0.55,
           "/data1/llx/BCData/test", 640, 10, "160", "BCData (IHC breast)"),
 "MoNuSeg":(f"{ROOT}/baselines/APGCC/apgcc/output/MoNuSeg_finetune/scores_test.json", 0.80,
            f"{ROOT}/improved/APGCC_ADE/apgcc/output/MoNuSeg_ADE_dcnv2_edge_stain/scores_test.json", 0.60,
            "/data1/llx/MoNuSegdata/test", 1000, 12, "TCGA-2Z-A9J9-01A-01-TS1", "MoNuSeg (H&E multi-organ)"),
}

def load(dump, thr):
    d = {}
    for s in json.load(open(dump))["samples"]:
        pr = np.array([p[:2] for p in s["points"] if p[2] > thr], float).reshape(-1, 2)
        gt = np.array(s["gt"], float).reshape(-1, 2)
        d[s["id"]] = (gt, pr)
    return d

def match(g, p, r=RADIUS):
    if len(g) == 0 or len(p) == 0:
        return np.zeros(len(p), bool), np.zeros(len(g), bool)
    D = np.linalg.norm(g[:, None] - p[None], axis=2)
    C = np.where(D <= r, D, 1e6)
    ri, ci = linear_sum_assignment(C)
    ptp = np.zeros(len(p), bool); gm = np.zeros(len(g), bool)
    for i, j in zip(ri, ci):
        if C[i, j] < 1e6:
            ptp[j] = True; gm[i] = True
    return ptp, gm

def density(pts, size, sigma):
    m = np.zeros((size, size), float)
    for x, y in pts:
        xi, yi = int(np.clip(x, 0, size - 1)), int(np.clip(y, 0, size - 1))
        m[yi, xi] += 1
    return gaussian_filter(m, sigma)

# ---------------- 1. qualitative scatter (3x3) ----------------
def qualitative():
    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    for row, (ds, cfg) in enumerate(DS.items()):
        bdump, bt, adump, at, idir, size, sigma, rid, nice = cfg
        B = load(bdump, bt); A = load(adump, at)
        gt, _ = A[rid]
        im = np.array(Image.open(f"{idir}/{rid}.png").convert("RGB"))
        # col 0: image + GT
        ax = axes[row, 0]; ax.imshow(im)
        ax.scatter(gt[:, 0], gt[:, 1], s=8, c=GREEN, edgecolors="none")
        ax.set_title(f"{nice}\n{rid}   GT={len(gt)}", fontsize=10)
        ax.axis("off")
        for col, (tag, D, thr) in enumerate([("baseline", B, bt), ("ADE", A, at)], start=1):
            g, p = D[rid]; ptp, gm = match(g, p)
            ax = axes[row, col]; ax.imshow(im)
            if len(p):
                ax.scatter(p[ptp, 0], p[ptp, 1], s=10, c=GREEN, edgecolors="none", label="TP")
                ax.scatter(p[~ptp, 0], p[~ptp, 1], s=22, marker="x", c=RED, lw=1.1, label="FP")
            if (~gm).any():
                ax.scatter(g[~gm, 0], g[~gm, 1], s=26, facecolors="none", edgecolors=YELLOW, lw=1.1, label="FN")
            err = len(p) - len(g)
            ax.set_title(f"{tag} @{thr:.2f}   pred={len(p)} ({err:+d})\nTP={ptp.sum()} FP={(~ptp).sum()} FN={(~gm).sum()}",
                         fontsize=9.5)
            ax.axis("off")
    axes[0, 1].legend(loc="upper right", fontsize=8, markerscale=1.3,
                      facecolor="white", framealpha=0.8, frameon=True)
    fig.suptitle("ADE qualitative: GT vs baseline vs ADE (Hungarian @12px; green=TP, red x=FP, yellow o=FN)",
                 fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    for e in ("png", "pdf"):
        plt.savefig(f"{OUT}/fig_ADE_qualitative_scatter.{e}", bbox_inches="tight")
    plt.close()

# ---------------- 2. density heatmap (per dataset) ----------------
def density_fig(ds):
    bdump, bt, adump, at, idir, size, sigma, rid, nice = DS[ds]
    A = load(adump, at); gt, pr = A[rid]
    im = np.array(Image.open(f"{idir}/{rid}.png").convert("RGB"))
    dg = density(gt, size, sigma); dp = density(pr, size, sigma)
    vmax = max(dg.max(), dp.max()) or 1.0
    res = dp - dg; rlim = np.abs(res).max() or 1.0
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.5))
    ax[0].imshow(im); ax[0].set_title(f"{nice}\n{rid} (GT={len(gt)})", fontsize=10); ax[0].axis("off")
    ax[1].imshow(dg, cmap="jet", vmin=0, vmax=vmax); ax[1].set_title(f"GT density ($\\sigma$={sigma})"); ax[1].axis("off")
    ax[2].imshow(dp, cmap="jet", vmin=0, vmax=vmax); ax[2].set_title(f"ADE density (pred={len(pr)})"); ax[2].axis("off")
    im3 = ax[3].imshow(res, cmap="RdBu_r", vmin=-rlim, vmax=rlim)
    ax[3].set_title("residual (ADE $-$ GT)"); ax[3].axis("off")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)
    plt.tight_layout()
    for e in ("png", "pdf"):
        plt.savefig(f"{OUT}/fig_ADE_density_heatmap_{ds}.{e}", bbox_inches="tight")
    plt.close()

# ---------------- 3. region analysis (per dataset) ----------------
def region_grid(pts, size, n=4):
    g = np.zeros((n, n), int); step = size / n
    for x, y in pts:
        gx = min(int(x // step), n - 1); gy = min(int(y // step), n - 1)
        g[gy, gx] += 1
    return g

def _img_counts(ax, im, size, celltext, title, color):
    """Original image as background + 4x4 grid + per-cell number(s) drawn on the image."""
    ax.imshow(im); step = size / 4
    for k in (1, 2, 3):
        ax.axvline(k * step, color="w", lw=1.2, alpha=0.85)
        ax.axhline(k * step, color="w", lw=1.2, alpha=0.85)
    for i in range(4):
        for j in range(4):
            ax.text((j + 0.5) * step, (i + 0.5) * step, celltext[i][j], ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="black")])
    ax.set_xlim(0, size); ax.set_ylim(size, 0)
    ax.set_title(title, fontsize=11.5); ax.set_xticks([]); ax.set_yticks([])

def region_fig(ds):
    bdump, bt, adump, at, idir, size, sigma, rid, nice = DS[ds]
    B = load(bdump, bt); A = load(adump, at)
    gt, pr = A[rid]
    im = np.array(Image.open(f"{idir}/{rid}.png").convert("RGB"))
    gg = region_grid(gt, size); pg = region_grid(pr, size); diff = pg - gg
    fig = plt.figure(figsize=(11.5, 10.5))
    gs = fig.add_gridspec(2, 2, hspace=0.20, wspace=0.18)
    # (0,0): image + GT counts
    gt_txt = [[str(gg[i, j]) for j in range(4)] for i in range(4)]
    _img_counts(fig.add_subplot(gs[0, 0]), im, size, gt_txt,
                f"{rid}: image + GT region counts (GT n={len(gt)})", "#7CFC00")
    # (0,1): image + ADE pred counts and error
    pr_txt = [[f"{pg[i,j]}\n({diff[i,j]:+d})" for j in range(4)] for i in range(4)]
    _img_counts(fig.add_subplot(gs[0, 1]), im, size, pr_txt,
                f"image + ADE region counts ($\\Delta$ vs GT; pred n={len(pr)})", "#FFD400")
    # (1,0): per-region error heatmap (unchanged)
    ax = fig.add_subplot(gs[1, 0]); lim = np.abs(diff).max() or 1
    imh = ax.imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{diff[i,j]:+d}", ha="center", va="center", fontsize=11)
    ax.set_title("per-region error (ADE $-$ GT)"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(imh, ax=ax, fraction=0.046, pad=0.04)
    # (1,1): full-test density-error scatter (unchanged)
    ax = fig.add_subplot(gs[1, 1])
    ids = sorted(A)
    gc = np.array([len(A[i][0]) for i in ids])
    be = np.array([len(B[i][1]) - len(B[i][0]) for i in ids])
    ae = np.array([len(A[i][1]) - len(A[i][0]) for i in ids])
    ax.axhline(0, color="#888", lw=1)
    ax.scatter(gc, be, s=18, c=GREY, alpha=0.6, label=f"baseline @{bt:.2f}")
    ax.scatter(gc, ae, s=18, c=GREEN, alpha=0.6, label="ADE")
    ax.set_xlabel("GT count per image (density)"); ax.set_ylabel("count error (pred $-$ GT)")
    ax.set_title(f"full test set (n={len(ids)}): error vs density"); ax.legend()
    ax.grid(alpha=0.3)
    fig.suptitle(f"ADE region-level analysis — {nice}", fontsize=13, fontweight="bold", y=0.97)
    for e in ("png", "pdf"):
        plt.savefig(f"{OUT}/fig_ADE_region_analysis_{ds}.{e}", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    qualitative()
    for ds in DS:
        density_fig(ds); region_fig(ds)
    print("wrote ADE visualization figures to", OUT)
