"""Compute instance-level metrics on MoNuSeg Test set."""
import glob, numpy as np, scipy.io as sio
from metrics.stats_utils import (
    get_dice_1, get_fast_aji, get_fast_aji_plus,
    get_fast_pq, remap_label, pair_coordinates
)

pred_dir = "inference/monuseg_test/mat/"
true_dir = "dataset/MoNuSeg/Test/Labels/"
files = sorted(glob.glob(pred_dir + "*.mat"))
print("Found %d prediction files\n" % len(files))

dices, ajis, aji_ps, pqs, f1s = [], [], [], [], []
gts, preds, tps, fps, fns = [], [], [], [], []

for fp in files:
    fname = fp.split("/")[-1].replace(".mat", "")
    t_map = sio.loadmat(true_dir + fname + ".mat")["inst_map"].astype("int32")
    p_map = sio.loadmat(fp)["inst_map"].astype("int32")
    t_map = remap_label(t_map)
    p_map = remap_label(p_map)

    dice = get_dice_1(t_map, p_map)
    aji = get_fast_aji(t_map, p_map)
    aji_p = get_fast_aji_plus(t_map, p_map)
    pq_info = get_fast_pq(t_map, p_map, match_iou=0.5)[0]

    # Detection metrics (F1_d) — compute centroids from inst_map
    def inst_centroids(inst_map):
        ids = np.unique(inst_map)
        ids = ids[ids > 0]
        centroids = np.zeros((len(ids), 2), dtype=np.float32)
        for i, uid in enumerate(ids):
            ys, xs = np.where(inst_map == uid)
            centroids[i] = [xs.mean(), ys.mean()]
        return centroids

    t_full = sio.loadmat(true_dir + fname + ".mat")["inst_map"].astype("int32")
    pc = inst_centroids(p_map)
    tc = inst_centroids(t_full)
    gt_n = tc.shape[0]
    pred_n = pc.shape[0]
    if gt_n > 0 and pred_n > 0:
        paired, unp_true, unp_pred = pair_coordinates(tc, pc, 12)
        tp, fp, fn = paired.shape[0], unp_pred.shape[0], unp_true.shape[0]
    elif gt_n == 0 and pred_n == 0:
        tp, fp, fn = 0, 0, 0
    elif gt_n == 0:
        tp, fp, fn = 0, pred_n, 0
    else:
        tp, fp, fn = 0, 0, gt_n
    f1_d = 2 * tp / (2 * tp + fp + fn + 1e-8)

    dices.append(dice)
    ajis.append(aji)
    aji_ps.append(aji_p)
    pqs.append(pq_info[2])
    f1s.append(f1_d)
    gts.append(gt_n)
    preds.append(pred_n)
    tps.append(tp)
    fps.append(fp)
    fns.append(fn)

    print("%-30s Dice=%.4f AJI=%.4f AJI+=%.4f PQ=%.4f F1=%.4f GT=%3d Pred=%3d" % (
        fname, dice, aji, aji_p, pq_info[2], f1_d, gt_n, pred_n))

print()
print("%-20s %8s" % ("=== SUMMARY ===", ""))
print("%-20s %8.4f" % ("DICE (mean)", np.mean(dices)))
print("%-20s %8.4f" % ("AJI (mean)", np.mean(ajis)))
print("%-20s %8.4f" % ("AJI+ (mean)", np.mean(aji_ps)))
print("%-20s %8.4f" % ("PQ (mean)", np.mean(pqs)))

total_tp = sum(tps)
total_fp = sum(fps)
total_fn = sum(fns)
f1_overall = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-8)
print("%-20s %8.4f" % ("F1_d (overall)", f1_overall))
print("%-20s %8.4f" % ("Precision", total_tp / (total_tp + total_fp + 1e-8)))
print("%-20s %8.4f" % ("Recall", total_tp / (total_tp + total_fn + 1e-8)))

total_gt = sum(gts)
total_pred = sum(preds)
mae = np.mean(np.abs(np.array(preds) - np.array(gts)))
print("%-20s %8d" % ("Total GT nuclei", total_gt))
print("%-20s %8d" % ("Total Pred nuclei", total_pred))
print("%-20s %+d (%.2f%%)" % ("Total count err", total_pred - total_gt,
                               (total_pred - total_gt) / total_gt * 100))
print("%-20s %8.1f" % ("MAE", mae))
