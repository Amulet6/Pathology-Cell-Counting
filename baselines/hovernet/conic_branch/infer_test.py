"""CoNIC Test Set Inference + Instance Counting Evaluation."""
import numpy as np
import torch
import torch.nn.functional as F
import csv, os, sys

from models.hovernet.net_desc import create_model
from run_utils.utils import convert_pytorch_checkpoint

# Config
BATCH_SIZE = 4
CHKPT = "exp_output/local/models/baseline/00/model/net_epoch=46.tar"
DATA_ROOT = "exp_output/local/data"

# Load test indices (799)
import pandas as pd
split_info = pd.read_csv("/root/autodl-tmp/Pathology-Cell-Counting/data/processed/CoNIC/split_info.csv")
test_idx = split_info[split_info["split"] == "test"]["patch_idx"].values.astype(int)
print(f"Test patches: {len(test_idx)}")

# Load data
images = np.load(f"{DATA_ROOT}/images.npy", mmap_mode="r")
labels = np.load(f"{DATA_ROOT}/labels.npy", mmap_mode="r")
counts = pd.read_csv("/root/autodl-tmp/Pathology-Cell-Counting/data/processed/CoNIC/counts.csv")
type_names = list(counts.columns)
print(f"Types: {type_names}")

# Load model
device = torch.device("cuda:0")
model = create_model(
    num_types=7, freeze=False, mode="fast",
    pretrained_backbone="exp_output/local/[ImageNet]resnet50-0676ba61.pth"
)
state_dict = torch.load(CHKPT)["desc"]
state_dict = convert_pytorch_checkpoint(state_dict)
model.load_state_dict(state_dict, strict=False)
model = model.to(device)
model.eval()
print("Model loaded")

from models.hovernet.post_proc import process as post_proc

all_pred_counts = []
for start in range(0, len(test_idx), BATCH_SIZE):
    batch_idx = test_idx[start:start + BATCH_SIZE]
    batch_imgs = torch.from_numpy(images[batch_idx].copy()).float()
    batch_imgs = batch_imgs.permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        pred = model(batch_imgs)
        for k in pred:
            pred[k] = pred[k].permute(0, 2, 3, 1)

    for i in range(len(batch_idx)):
        np_out = F.softmax(pred["np"][i], dim=-1)[..., 1].cpu().numpy()
        tp_out = F.softmax(pred["tp"][i], dim=-1).cpu().numpy()

        # Stack for post_proc: [tp(7ch), np(1ch), hv(2ch)]
        # Actually post_proc expects: [type(1ch), np(1ch), hv_x(1ch), hv_y(1ch)]
        # Or: just pred_map with proper stacking
        hv_x = pred["hv"][i][..., 0:1].cpu().numpy()
        hv_y = pred["hv"][i][..., 1:2].cpu().numpy()

        # Build pred_map for post_proc: [type, np, hv_x, hv_y]
        # type is argmax of tp
        type_map = np.argmax(tp_out, axis=-1, keepdims=True).astype(np.float32)
        np_map = np_out[..., np.newaxis]
        pred_map = np.concatenate([type_map, np_map, hv_x, hv_y], axis=-1)

        # Run official post-processing (watershed + instance typing)
        inst_map, inst_info = post_proc(pred_map, nr_types=7, return_centroids=True)

        # Count instances per type (only in central 224×224 to match GT protocol)
        counts_per_type = [0] * 6
        if inst_info is not None:
            for inst_data in inst_info.values():
                inst_type = inst_data.get("type", None)
                centroid = inst_data.get("centroid", None)
                if inst_type is not None and centroid is not None:
                    cx, cy = centroid[0], centroid[1]
                    # central 224×224 region: [16, 240) in a 256×256 image
                    if 16 <= cx < 240 and 16 <= cy < 240:
                        if 1 <= inst_type <= 6:
                            counts_per_type[inst_type - 1] += 1

        all_pred_counts.append(counts_per_type)

    if (start // BATCH_SIZE) % 50 == 0:
        print(f"  Processed {start + len(batch_idx)}/{len(test_idx)}")

# Evaluate
gt_counts = counts.iloc[test_idx].values.astype(int)
pred_counts = np.array(all_pred_counts)

print(f"\n=== Overall Count Metrics ===")
total_gt = gt_counts.sum(axis=0)
total_pred = pred_counts.sum(axis=0)
print(f"{'Type':<15} {'GT':>8} {'Pred':>8} {'Error':>8} {'RelErr%':>8}")
for i, name in enumerate(type_names):
    err = total_pred[i] - total_gt[i]
    rel = err / (total_gt[i] + 1e-8) * 100
    print(f"{name:<15} {total_gt[i]:>8d} {total_pred[i]:>8d} {err:+8d} {rel:>8.1f}")
total_gt_sum = total_gt.sum()
total_pred_sum = total_pred.sum()
print(f"{'TOTAL':<15} {total_gt_sum:>8d} {total_pred_sum:>8d} {total_pred_sum-total_gt_sum:+8d} {(total_pred_sum-total_gt_sum)/total_gt_sum*100:>8.1f}")

# Per-type MAE/RMSE/R²
mae_per = np.mean(np.abs(pred_counts - gt_counts), axis=0)
rmse_per = np.sqrt(np.mean((pred_counts - gt_counts)**2, axis=0))
print(f"\n{'Type':<15} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
for i, name in enumerate(type_names):
    ss_res = np.sum((gt_counts[:,i] - pred_counts[:,i])**2)
    ss_tot = np.sum((gt_counts[:,i] - np.mean(gt_counts[:,i]))**2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    print(f"{name:<15} {mae_per[i]:>8.2f} {rmse_per[i]:>8.2f} {r2:>8.4f}")

# Overall
overall_mae = np.mean(np.abs(pred_counts - gt_counts))
overall_rmse = np.sqrt(np.mean((pred_counts - gt_counts)**2))
per_img_gt = gt_counts.sum(axis=1)
per_img_pred = pred_counts.sum(axis=1)
img_mae = np.mean(np.abs(per_img_pred - per_img_gt))
print(f"\nOverall MAE: {overall_mae:.2f}, Overall RMSE: {overall_rmse:.2f}")
print(f"Per-image MAE (total): {img_mae:.2f}")
