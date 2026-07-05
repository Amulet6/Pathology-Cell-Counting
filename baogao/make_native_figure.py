#!/usr/bin/env python3
"""Direction-A native-base figure (CoNIC test, fully-trained ep>=120 runs).
Left : native EOS x tau sweep + focal, MAE bars vs baseline (all near baseline -> recipe ~neutral).
Right: native vs unified contrast (baseline -> best recipe) -> calibration value proportional to base mis-calibration.
Numbers from baselines/APGCC/apgcc/output/{native_*,CoNIC_native_*} test_scan + test_t* evals."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "figure.facecolor": "white",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.6, "grid.alpha": 0.6,
    "legend.frameon": False,
})
GREY, BLUE, GREEN, RED = "#8a8a8a", "#4C72B0", "#55A868", "#C44E52"
OUT = "/home/lixinli/Pathology-Cell-Counting/baogao/figures"
os.makedirs(OUT, exist_ok=True)

# --- native EOS x tau sweep (fully trained, sweet-spot threshold) ---
labels = ["baseline\nEOS0.5/$\\tau$.05", "EOS.25/$\\tau$.10", "EOS.10/$\\tau$.05",
          "EOS.10/$\\tau$.10", "EOS.10/$\\tau$.15", "EOS.05/$\\tau$.10", "+focal\nEOS.10/$\\tau$.10"]
mae = [12.93, 10.83, 10.54, 12.50, 15.10, 16.31, 12.15]
f1  = [0.7929, 0.7810, 0.7185, 0.7309, 0.7402, 0.6840, 0.7740]
colors = [GREY, GREEN, GREEN, BLUE, RED, RED, BLUE]
x = np.arange(len(labels))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

ax = axes[0]
bars = ax.bar(x, mae, color=colors, edgecolor="white", width=0.66)
ax.bar_label(bars, fmt="%.2f", fontsize=8.5, fontweight="bold", padding=2)
ax.axhline(12.93, color=GREY, ls="--", lw=1.2)
ax.text(len(labels)-1, 12.93, "unified baseline 12.93", color=GREY, fontsize=8, va="bottom", ha="right")
for xi, fi in zip(x, f1):
    ax.text(xi, 0.6, f"F1\n{fi:.3f}", ha="center", va="bottom", fontsize=7.3, color="#333")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.2)
ax.set_ylabel("Counting MAE (CoNIC test)"); ax.set_ylim(0, 18)
ax.set_title("Native base: A recipe sweep stays near baseline")
ax.grid(axis="x", alpha=0)

# --- native vs unified contrast ---
ax = axes[1]
groups = ["unified\n(mis-calibrated base)", "native\n(well-calibrated base)"]
base = [25.50, 12.93]   # native base = team-unified baseline (CoNIC dir-E, 12.93)
best = [13.44, 10.83]   # unified best recipe (ep180 final val-best); native MAE-best A config (EOS0.25/τ0.10 @0.40)
xx = np.arange(2); w = 0.36
b1 = ax.bar(xx - w/2, base, w, color=GREY, label="baseline", edgecolor="white")
b2 = ax.bar(xx + w/2, best, w, color=GREEN, label="+A recipe (best)", edgecolor="white")
ax.bar_label(b1, fmt="%.2f", fontsize=9, padding=2)
ax.bar_label(b2, fmt="%.2f", fontsize=9, padding=2)
ax.annotate("$-47\\%$", xy=(0, 19), fontsize=11, color=GREEN, ha="center", fontweight="bold")
ax.annotate("modest\n(worse F1)", xy=(1, 14), fontsize=9.5, color=GREY, ha="center", fontweight="bold")
ax.set_xticks(xx); ax.set_xticklabels(groups, fontsize=9.5)
ax.set_ylabel("Counting MAE (CoNIC test)"); ax.set_ylim(0, 28)
ax.set_title("Calibration gain $\\propto$ base mis-calibration")
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="x", alpha=0)

fig.suptitle("Direction A on the native (final) base: confirms recipe is corrective, not free tuning",
             fontsize=12.5, fontweight="bold", y=1.02)
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(f"{OUT}/fig_A_native.{ext}", bbox_inches="tight")
print("saved fig_A_native.{png,pdf}")
