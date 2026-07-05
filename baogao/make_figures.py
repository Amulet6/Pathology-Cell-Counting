#!/usr/bin/env python3
"""Generate publication-style baseline-reproduction figures (English labels)."""
import re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ---- global publication style ----
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "figure.facecolor": "white",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.6, "grid.alpha": 0.6,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False, "legend.fontsize": 9,
})
BLUE, ORANGE, GREEN, RED, PURPLE = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"

BASE = "/home/lixinli/Pathology-Cell-Counting/baselines/APGCC/apgcc/output"
OUT  = "/home/lixinli/Pathology-Cell-Counting/baogao/figures"
os.makedirs(OUT, exist_ok=True)
DATASETS = ["BCData", "MoNuSeg", "CoNIC"]

# ---- 1. training curves (val MAE vs epoch) ----
def curve(ds):
    eps, maes = [], []
    for line in open(f"{BASE}/{ds}_finetune/log.txt"):
        m = re.search(r"\[ep (\d+)\] Eval: MAE=([\d.]+)", line)
        if m:
            eps.append(int(m.group(1))); maes.append(float(m.group(2)))
    return np.array(eps), np.array(maes)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
for ax, ds in zip(axes, DATASETS):
    e, m = curve(ds)
    ax.plot(e, m, color=BLUE, lw=1.8, alpha=0.9, zorder=2)
    ax.fill_between(e, m, m.max(), color=BLUE, alpha=0.06, zorder=1)
    bi = int(np.argmin(m))
    ax.scatter([e[bi]], [m[bi]], s=90, marker="*", color=RED, zorder=5,
               edgecolor="white", linewidth=0.8,
               label=f"best {m[bi]:.2f} @ ep{e[bi]}")
    ax.set_title(ds); ax.set_xlabel("Epoch"); ax.set_ylabel("Validation MAE (patch)")
    ax.legend(loc="best", handletextpad=0.3)
    ax.margins(x=0.02)
fig.suptitle("APGCC baseline fine-tuning convergence", fontsize=13, fontweight="bold", y=1.03)
plt.tight_layout()
for _ext in ("png", "pdf"): plt.savefig(f"{OUT}/fig_train_curves.{_ext}", bbox_inches="tight")
plt.close()

# ---- metrics from centroid_eval (test set) ----
M = {
 "BCData":  {"mae":18.27, "rmse":23.73, "err":-3.67, "f1":[0.7176,0.8145,0.8387]},
 "MoNuSeg": {"mae":111.71,"rmse":116.01,"err":23.35, "f1":[0.7361,0.8032,0.8291]},
 "CoNIC":   {"mae":11.99, "rmse":19.74, "err":-3.08, "f1":[0.6460,0.7944,0.8794]},
}

# ---- 2. localization F1 grouped bar ----
x = np.arange(3); w = 0.26
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for i, (lab, c) in enumerate(zip(["F1 @ 6px", "F1 @ 12px", "F1 @ 24px"], [BLUE, ORANGE, GREEN])):
    bars = ax.bar(x + (i-1)*w, [M[d]["f1"][i] for d in DATASETS], w,
                  label=lab, color=c, edgecolor="white", linewidth=0.6)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
ax.set_xticks(x); ax.set_xticklabels(DATASETS); ax.set_ylim(0, 1.0)
ax.set_ylabel("F1 score"); ax.set_axisbelow(True)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12), columnspacing=1.4)
ax.grid(axis="x", alpha=0)
plt.tight_layout()
for _ext in ("png", "pdf"): plt.savefig(f"{OUT}/fig_localization_f1.{_ext}", bbox_inches="tight")
plt.close()

# ---- 3. counting bias bar ----
fig, ax = plt.subplots(figsize=(6, 4.2))
errs = [M[d]["err"] for d in DATASETS]
colors = [RED if e > 0 else BLUE for e in errs]
bars = ax.bar(DATASETS, errs, color=colors, edgecolor="white", linewidth=0.6, width=0.6)
ax.axhline(0, color="#333333", lw=1.0)
for i, e in enumerate(errs):
    ax.text(i, e + (1.0 if e > 0 else -1.0), f"{e:+.1f}%", ha="center",
            va="bottom" if e > 0 else "top", fontsize=11, fontweight="bold")
ax.set_ylabel("Total count error  (pred $-$ GT) / GT  [%]")
ax.set_title("Counting bias across datasets")
ax.set_ylim(min(errs) - 5, max(errs) + 5)
ax.set_axisbelow(True); ax.grid(axis="x", alpha=0)
plt.tight_layout()
for _ext in ("png", "pdf"): plt.savefig(f"{OUT}/fig_counting_bias.{_ext}", bbox_inches="tight")
plt.close()

print("figures written:")
for f in sorted(os.listdir(OUT)):
    print(" ", f)
