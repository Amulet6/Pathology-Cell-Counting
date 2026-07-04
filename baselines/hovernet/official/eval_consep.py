"""Compute instance-level DICE/AJI/PQ on CoNSeP Test set from inference .mat files."""
import glob
import numpy as np
import scipy.io as sio
from metrics.stats_utils import (
    get_dice_1, get_fast_aji, get_fast_aji_plus,
    get_fast_dice_2, get_fast_pq, remap_label
)

pred_dir = "inference/consep_test/mat/"
true_dir = "dataset/CoNSeP/Test/Labels/"

true_list = glob.glob(true_dir + "*.mat")
true_list.sort()

all_dice = []
all_aji = []
all_aji_plus = []
all_pq = []

for true_path in true_list:
    fname = true_path.split("/")[-1].replace(".mat", "")
    pred_path = pred_dir + fname + ".mat"

    true_info = sio.loadmat(true_path)
    true_map = true_info["inst_map"]

    pred_info = sio.loadmat(pred_path)
    pred_map = pred_info["inst_map"]

    # remap labels to consecutive integers
    true_map = remap_label(true_map)
    pred_map = remap_label(pred_map)

    dice = get_dice_1(true_map, pred_map)
    aji = get_fast_aji(true_map, pred_map)
    aji_plus = get_fast_aji_plus(true_map, pred_map)
    pq_info = get_fast_pq(true_map, pred_map)

    all_dice.append(dice)
    all_aji.append(aji)
    all_aji_plus.append(aji_plus)
    all_pq.append(pq_info[0][2])  # overall PQ

header = f"{'Image':<12} {'DICE':>8} {'AJI':>8} {'AJI+':>8} {'PQ':>8}"
print(header)
print("-" * len(header))
for i, f in enumerate([p.split("/")[-1].replace(".mat", "") for p in true_list]):
    print(f"{f:<12} {all_dice[i]:8.4f} {all_aji[i]:8.4f} {all_aji_plus[i]:8.4f} {all_pq[i]:8.4f}")
print("-" * len(header))
print(f"{'MEAN':<12} {np.mean(all_dice):8.4f} {np.mean(all_aji):8.4f} {np.mean(all_aji_plus):8.4f} {np.mean(all_pq):8.4f}")
