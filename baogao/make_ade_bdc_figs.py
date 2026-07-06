#!/usr/bin/env python3
"""ADE report figures in the BDC-style layout (numeric plots).
Produces, into baogao/figures/:
  fig_ADE_main_comparison_{CoNIC,BCData,MoNuSeg}   baseline vs ADE (each at its MAE-optimal thr)
  fig_ADE_module_ablation_CoNIC                    baseline/EOS0.10/EOS0.25/D+E clean/+density
  fig_ADE_threshold_scan_{CoNIC,BCData,MoNuSeg}    MAE vs score threshold (baseline & ADE)
All numbers are the official centroid_eval outputs (see baogao/ade_data/*.json).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "figure.facecolor": "white",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.6, "grid.alpha": 0.6,
    "legend.frameon": False, "legend.fontsize": 9,
})
GREY, GREEN, RED, ORANGE, BLUE = "#8a8a8a", "#55A868", "#C44E52", "#DD8452", "#4C72B0"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = f"{HERE}/figures"; os.makedirs(OUT, exist_ok=True)
DATA = f"{HERE}/ade_data"

# ---- official metrics (centroid_eval), each model at its MAE-optimal threshold ----
# (mae, f1@12, thr); CoNIC ADE also has +A density-adaptive = 10.37
MAIN = {
    "CoNIC":  {"base": (11.92, 0.7937, 0.45), "ADE": (11.31, 0.7938, 0.30), "ade_density": 10.37},
    "BCData": {"base": (17.38, 0.8184, 0.45), "ADE": (18.92, 0.8062, 0.55)},
    "MoNuSeg":{"base": (22.21, 0.7785, 0.80), "ADE": (19.64, 0.6858, 0.60)},
}
TITLE = {
    "CoNIC":  "CoNIC (H&E colon, 991 test)",
    "BCData": "BCData (IHC breast, 402 test)",
    "MoNuSeg":"MoNuSeg (H&E multi-organ, 14 test)",
}

def main_comparison(ds):
    b_mae, b_f1, b_t = MAIN[ds]["base"]; a_mae, a_f1, a_t = MAIN[ds]["ADE"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.0))
    x = np.arange(2)
    # MAE
    bars = ax[0].bar(x, [b_mae, a_mae], color=[GREY, GREEN], edgecolor="white", width=0.6)
    ax[0].bar_label(bars, fmt="%.2f", fontsize=10, fontweight="bold", padding=2)
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"baseline\n@{b_t:.2f}", f"ADE\n@{a_t:.2f}"], fontsize=9)
    ax[0].set_ylabel("Counting MAE  (each @MAE-opt thr)")
    ax[0].set_ylim(0, max(b_mae, a_mae) * 1.32); ax[0].set_title("Counting MAE$\\downarrow$")
    ax[0].grid(axis="x", alpha=0)
    if "ade_density" in MAIN[ds]:
        d = MAIN[ds]["ade_density"]
        ax[0].annotate(f"+A density-adaptive\n{a_mae:.2f}$\\to${d:.2f}", xy=(1, a_mae),
                       xytext=(0.35, a_mae * 1.18), fontsize=8.5, color=GREEN, ha="center",
                       arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.1))
    # F1
    bars = ax[1].bar(x, [b_f1, a_f1], color=[GREY, GREEN], edgecolor="white", width=0.6)
    ax[1].bar_label(bars, fmt="%.3f", fontsize=10, fontweight="bold", padding=2)
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"baseline\n@{b_t:.2f}", f"ADE\n@{a_t:.2f}"], fontsize=9)
    ax[1].set_ylabel("F1@12px"); ax[1].set_ylim(0.5, max(b_f1, a_f1) * 1.12)
    ax[1].set_title("Localization F1@12$\\uparrow$"); ax[1].grid(axis="x", alpha=0)
    fig.suptitle(f"ADE vs baseline — {TITLE[ds]}", fontsize=11.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    for e in ("png", "pdf"):
        plt.savefig(f"{OUT}/fig_ADE_main_comparison_{ds}.{e}", bbox_inches="tight")
    plt.close()

def module_ablation():
    labels = ["APGCC\nbaseline", "ADE\nEOS0.10", "ADE\nEOS0.25", "ADE\nD+E clean", "D+E clean\n+A density"]
    mae = [12.93, 21.21, 15.19, 11.87, 10.37]        # val->test honest global MAE; last = density-adaptive
    f1  = [0.793, 0.777, 0.783, 0.7966, 0.7962]
    colors = [GREY, RED, ORANGE, GREEN, BLUE]
    x = np.arange(5)
    fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.3))
    bars = ax[0].bar(x, mae, color=colors, edgecolor="white", width=0.64)
    ax[0].bar_label(bars, fmt="%.2f", fontsize=9, fontweight="bold", padding=2)
    ax[0].axhline(12.93, color=GREY, ls="--", lw=1.1)
    ax[0].text(0.02, 13.2, "baseline 12.93", color=GREY, fontsize=8, va="bottom")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8.5)
    ax[0].set_ylabel("Counting MAE (val$\\to$test, CoNIC)"); ax[0].set_ylim(0, 24)
    ax[0].set_title("Counting: aggressive recipe harmful; D+E clean + A density best")
    ax[0].grid(axis="x", alpha=0)
    bars = ax[1].bar(x, f1, color=colors, edgecolor="white", width=0.64)
    ax[1].bar_label(bars, fmt="%.3f", fontsize=9, fontweight="bold", padding=2)
    ax[1].axhline(0.793, color=GREY, ls="--", lw=1.1)
    ax[1].text(4.4, 0.793, "baseline 0.793", color=GREY, fontsize=8, va="bottom", ha="right")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8.5)
    ax[1].set_ylabel("F1@12px"); ax[1].set_ylim(0.70, 0.82)
    ax[1].set_title("Localization: only D+E clean reaches/exceeds baseline")
    ax[1].grid(axis="x", alpha=0)
    fig.suptitle("ADE module ablation on CoNIC (A's training recipe: EOS0.10/0.25 vs OFF=D+E clean)",
                 fontsize=11.5, fontweight="bold", y=1.02)
    plt.tight_layout()
    for e in ("png", "pdf"):
        plt.savefig(f"{OUT}/fig_ADE_module_ablation_CoNIC.{e}", bbox_inches="tight")
    plt.close()

def threshold_scan(ds):
    S = json.load(open(f"{DATA}/ade_threshold_scans.json"))
    b = S[f"{ds}_base"]; a = S[f"{ds}_ADE"]
    g = np.array(b["grid"])
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(g, b["maecurve"], "-o", color=GREY, ms=3, lw=1.6, label="baseline")
    ax.plot(g, a["maecurve"], "-o", color=GREEN, ms=3, lw=1.6, label="ADE")
    for d_, col in [(b, GREY), (a, GREEN)]:
        bt = d_["bestT"]; bm = min(d_["maecurve"])
        ax.plot([bt], [bm], "*", color=col, ms=15, zorder=5)
        ax.annotate(f"{bm:.1f}@{bt:.2f}", xy=(bt, bm), xytext=(0, -14),
                    textcoords="offset points", color=col, fontsize=8.5, ha="center")
    ax.axvline(0.5, color="#bbbbbb", ls=":", lw=1)
    ax.set_xlabel("score threshold"); ax.set_ylabel("Counting MAE$\\downarrow$")
    ax.set_title(f"Threshold scan — {TITLE[ds]}")
    # log-y for MoNuSeg/BCData where low-thr MAE explodes
    hi = max(max(b["maecurve"]), max(a["maecurve"]))
    lo = min(min(b["maecurve"]), min(a["maecurve"]))
    if hi / max(lo, 1e-6) > 40:
        ax.set_yscale("log")
    ax.legend(loc="upper center")
    plt.tight_layout()
    for e in ("png", "pdf"):
        plt.savefig(f"{OUT}/fig_ADE_threshold_scan_{ds}.{e}", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    for ds in ("CoNIC", "BCData", "MoNuSeg"):
        main_comparison(ds); threshold_scan(ds)
    module_ablation()
    print("wrote ADE numeric figures to", OUT)
