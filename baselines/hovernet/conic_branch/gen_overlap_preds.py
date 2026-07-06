"""为 overlap4x 训练的模型生成 CoNIC 测试集预测（fold1 PNG 数据）。

用法:
  python gen_overlap_preds.py --chkpt <path> --out <output_dir> [--gpu 0]
"""
import argparse
import os, sys, glob, cv2
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

from models.hovernet.net_desc import create_model
from models.hovernet.post_proc import process as post_proc
from run_utils.utils import convert_pytorch_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--chkpt', type=str, required=True, help='模型 checkpoint .tar 路径')
parser.add_argument('--out', type=str, required=True, help='输出目录')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.makedirs(args.out, exist_ok=True)

BATCH_SIZE = 4
TEST_DIR = "/root/autodl-tmp/Pathology-Cell-Counting/data/processed/CoNIC_overlap/conic_cellvit_patient/fold1"

# --- test data: 991 PNG + NPY patches ---
img_files = sorted(glob.glob(f"{TEST_DIR}/images/*.png"))
lbl_files = sorted(glob.glob(f"{TEST_DIR}/labels/*.npy"))
assert len(img_files) == len(lbl_files) == 991, f"Expected 991 test patches, got {len(img_files)} imgs / {len(lbl_files)} lbls"
print(f"Test patches: {len(img_files)}")

type_names = ["neutrophil", "epithelial", "lymphocyte", "plasma", "eosinophil", "connective"]

# --- model ---
device = torch.device("cuda:0")
model = create_model(num_types=7, freeze=False,
                     pretrained_backbone="exp_output/local/[ImageNet]resnet50-0676ba61.pth")
state_dict = torch.load(args.chkpt, map_location=device)["desc"]
state_dict = convert_pytorch_checkpoint(state_dict)
model.load_state_dict(state_dict, strict=False)
model = model.to(device).eval()
print(f"Model loaded: {args.chkpt}")

# --- inference ---
N = len(img_files)
seg_preds = np.zeros((N, 256, 256, 2), dtype=np.int32)
reg_preds = np.zeros((N, 6), dtype=np.int32)
gt_seg = np.zeros((N, 256, 256, 2), dtype=np.uint16)

for start in range(0, N, BATCH_SIZE):
    end = min(start + BATCH_SIZE, N)
    # 加载 PNG，BGR → RGB（训练数据是 RGB）
    batch_imgs_np = np.stack([
        cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        for f in img_files[start:end]
    ], axis=0)
    batch_imgs = torch.from_numpy(batch_imgs_np).float().permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        pred = model(batch_imgs)
        for k in pred:
            pred[k] = pred[k].permute(0, 2, 3, 1)

    for j in range(end - start):
        np_prob = F.softmax(pred["np"][j], dim=-1)[..., 1].cpu().numpy()
        tp_prob = F.softmax(pred["tp"][j], dim=-1).cpu().numpy()
        hv_x = pred["hv"][j][..., 0:1].cpu().numpy()
        hv_y = pred["hv"][j][..., 1:2].cpu().numpy()

        type_argmax = np.argmax(tp_prob, axis=-1, keepdims=True).astype(np.float32)
        pred_map = np.concatenate([type_argmax, np_prob[..., np.newaxis], hv_x, hv_y], axis=-1)

        inst_map, inst_info = post_proc(pred_map, nr_types=7, return_centroids=True)
        seg_preds[start + j, ..., 0] = inst_map.astype(np.int32)

        class_map = np.zeros((256, 256), dtype=np.int32)
        if inst_info is not None:
            for inst_id, info in inst_info.items():
                t = info.get("type")
                if t is not None and 1 <= t <= 6:
                    class_map[inst_map == inst_id] = int(t)
        seg_preds[start + j, ..., 1] = class_map

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

        # GT from NPY labels (dict with inst_map + type_map)
        gt = np.load(lbl_files[start + j], allow_pickle=True).item()
        gt_seg[start + j, ..., 0] = gt['inst_map']
        gt_seg[start + j, ..., 1] = gt['type_map']

    print(f"  {end}/{N}")

# --- Save ---
np.save(f"{args.out}/seg_preds.npy", seg_preds.astype(np.uint16))
np.save(f"{args.out}/gt_seg.npy", gt_seg)

reg_df = pd.DataFrame(reg_preds, columns=type_names)
reg_df.to_csv(f"{args.out}/reg_preds.csv", index=False)

print(f"\nDone. → {args.out}")
print(f"  seg_preds.npy: {seg_preds.shape}")
print(f"  reg_preds.csv: {len(reg_df)} rows")
