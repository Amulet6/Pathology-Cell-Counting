"""Generate predictions in CoNIC official evaluation format.

Output:
  exp_output/local/test_preds/seg_preds.npy   — (799, 256, 256, 2) [inst_map, class_map]
  exp_output/local/test_preds/reg_preds.csv   — 799×6 counts, same columns as counts.csv
"""
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import os, cv2

from models.hovernet.net_desc import create_model
from models.hovernet.post_proc import process as post_proc
from run_utils.utils import convert_pytorch_checkpoint

BATCH_SIZE = 4
CHKPT = "exp_output/local/models/baseline/00/model/net_epoch=46.tar"
DATA_ROOT = "exp_output/local/data"

# --- load test indices ---
split_info = pd.read_csv("/root/autodl-tmp/Pathology-Cell-Counting/data/processed/CoNIC/split_info.csv")
test_idx = split_info[split_info["split"] == "test"]["patch_idx"].values.astype(int)
print(f"Test patches: {len(test_idx)}")

# --- load data ---
images = np.load(f"{DATA_ROOT}/images.npy", mmap_mode="r")
gt_counts = pd.read_csv("/root/autodl-tmp/Pathology-Cell-Counting/data/processed/CoNIC/counts.csv")
type_names = list(gt_counts.columns)
os.makedirs("exp_output/local/test_preds", exist_ok=True)

# --- load model ---
device = torch.device("cuda:0")
model = create_model(num_types=7, freeze=False, mode="fast",
                     pretrained_backbone="exp_output/local/[ImageNet]resnet50-0676ba61.pth")
state_dict = torch.load(CHKPT)["desc"]
state_dict = convert_pytorch_checkpoint(state_dict)
model.load_state_dict(state_dict, strict=False)
model = model.to(device).eval()
print("Model loaded")

# --- inference ---
seg_preds = np.zeros((len(test_idx), 256, 256, 2), dtype=np.int32)
reg_preds = np.zeros((len(test_idx), 6), dtype=np.int32)

for start in range(0, len(test_idx), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(test_idx))
    batch_idx = test_idx[start:end]
    batch_imgs_np = images[batch_idx].copy()
    batch_imgs = torch.from_numpy(batch_imgs_np).float().permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        pred = model(batch_imgs)
        for k in pred:
            pred[k] = pred[k].permute(0, 2, 3, 1)

    for j, idx in enumerate(range(end - start)):
        np_prob = F.softmax(pred["np"][j], dim=-1)[..., 1].cpu().numpy()
        tp_prob = F.softmax(pred["tp"][j], dim=-1).cpu().numpy()
        hv_x = pred["hv"][j][..., 0:1].cpu().numpy()
        hv_y = pred["hv"][j][..., 1:2].cpu().numpy()

        # build pred_map for post_proc: [channels]
        # post_proc expects pred_map with [type_map(1ch), np(1ch), hv_x(1ch), hv_y(1ch)]
        type_argmax = np.argmax(tp_prob, axis=-1, keepdims=True).astype(np.float32)
        pred_map = np.concatenate([type_argmax, np_prob[..., np.newaxis], hv_x, hv_y], axis=-1)

        inst_map, inst_info = post_proc(pred_map, nr_types=7, return_centroids=True)

        # Save inst_map and class_map for Task 1
        seg_preds[start + j, ..., 0] = inst_map.astype(np.int32)

        # Build per-pixel class map for Task 1
        class_map = np.zeros((256, 256), dtype=np.int32)
        if inst_info is not None:
            for inst_id, info in inst_info.items():
                t = info.get("type")
                if t is not None and 1 <= t <= 6:
                    class_map[inst_map == inst_id] = int(t)
        seg_preds[start + j, ..., 1] = class_map

        # Count for Task 2 (central 224x224)
        counts = [0] * 6
        if inst_info is not None:
            for info in inst_info.values():
                t = info.get("type")
                c = info.get("centroid")
                if t is not None and c is not None and 1 <= t <= 6:
                    cx, cy = c[0], c[1]
                    if 16 <= cx < 240 and 16 <= cy < 240:
                        counts[int(t) - 1] += 1
        reg_preds[start + j] = counts

    print(f"  {end}/{len(test_idx)}")

# --- Save ---
# Task 1
np.save("exp_output/local/test_preds/seg_preds.npy", seg_preds.astype(np.uint16))
# Also save GT for comparison
labels = np.load(f"{DATA_ROOT}/labels.npy", mmap_mode="r")
gt_seg = labels[test_idx].astype(np.uint16)
np.save("exp_output/local/test_preds/gt_seg.npy", gt_seg)

# Task 2
reg_df = pd.DataFrame(reg_preds, columns=type_names)
reg_df.to_csv("exp_output/local/test_preds/reg_preds.csv", index=False)
# Save GT
gt_df = gt_counts.iloc[test_idx]
gt_df.to_csv("exp_output/local/test_preds/gt_reg.csv", index=False)

print(f"\nDone. Files saved:")
print(f"  seg_preds.npy: {seg_preds.shape}, dtype={seg_preds.dtype}")
print(f"  gt_seg.npy: {gt_seg.shape}")
print(f"  reg_preds.csv: {len(reg_df)} rows × 6 cols")
print(f"  gt_reg.csv: {len(gt_df)} rows × 6 cols")
