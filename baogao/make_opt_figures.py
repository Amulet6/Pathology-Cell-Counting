#!/usr/bin/env python3
"""Direction-A optimization figures (English labels). Data from Direction-A report (CoNIC unified base)."""
import os
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
BLUE, ORANGE, GREEN, RED, GREY = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8a8a8a"
OUT = "/home/lixinli/Pathology-Cell-Counting/baogao/figures"
os.makedirs(OUT, exist_ok=True)

def save(name):
    for ext in ("png", "pdf"):
        plt.savefig(f"{OUT}/{name}.{ext}", bbox_inches="tight")
    plt.close()

# ---- Fig A1: recipe ablation on CoNIC (unified base), MAE step curve ----
steps = ["APGCC\nbaseline", "K=8", "+EOS$\\downarrow$", "+$\\tau\\uparrow$", "+focal", "+density\nthr"]
mae   = [25.50, 32.66, 15.19, 14.00, 13.44, 13.54]
colors = [GREY, RED, ORANGE, ORANGE, GREEN, ORANGE]
fig, ax = plt.subplots(figsize=(7.6, 4.2))
x = np.arange(len(steps))
ax.plot(x, mae, color="#444", lw=1.4, zorder=1)
ax.scatter(x, mae, c=colors, s=120, zorder=3, edgecolor="white", linewidth=1)
for xi, m in zip(x, mae):
    ax.annotate(f"{m:.2f}", (xi, m), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, fontweight="bold")
ax.annotate("negative result\n(K=8 wrong direction)", (1, 32.66), textcoords="offset points",
            xytext=(20, -2), fontsize=8, color=RED)
ax.set_xticks(x); ax.set_xticklabels(steps, fontsize=9)
ax.set_ylabel("Counting MAE  (CoNIC test)"); ax.set_ylim(0, 36)
ax.set_title("Direction-A recipe ablation on CoNIC (unified base): MAE 25.50 $\\to$ 13.44 ($-47\\%$)")
save("fig_A_ablation")

# ---- Fig A2: three-dataset transfer (baseline vs +recipe) ----
ds   = ["CoNIC\n(dense, under)", "BCData\n(balanced)", "MoNuSeg\n(over)"]
base = [25.50, 17.19, 26.21]
rec  = [14.00, 18.35, 24.79]
x = np.arange(3); w = 0.36
fig, ax = plt.subplots(figsize=(7.2, 4.2))
b1 = ax.bar(x - w/2, base, w, label="APGCC baseline", color=BLUE, edgecolor="white")
b2 = ax.bar(x + w/2, rec,  w, label="+ Direction-A recipe", color=GREEN, edgecolor="white")
ax.bar_label(b1, fmt="%.2f", fontsize=8, padding=2)
ax.bar_label(b2, fmt="%.2f", fontsize=8, padding=2)
ax.set_xticks(x); ax.set_xticklabels(ds)
ax.set_ylabel("Counting MAE"); ax.set_ylim(0, 33)
ax.set_title("Mechanism-specific transfer: recipe fixes 'under-count', neutral elsewhere", fontsize=11)
ax.legend(loc="upper left")
ax.grid(axis="x", alpha=0)
save("fig_A_transfer")

print("Direction-A figures written:")
for f in sorted(os.listdir(OUT)):
    if f.startswith("fig_A_"): print(" ", f)
