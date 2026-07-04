#!/usr/bin/env python3
"""
CoNIC overlap4x 数据转换：CellViT 格式 → HoVer-Net 格式

输入：
  data/processed/CoNIC_overlap/conic_cellvit_patient_x40_linear_withOverlap/
    ├── fold0/images/*.png   (15960 files)
    ├── fold0/labels/*.npy   (15960 files, dict{inst_map, type_map})
    ├── fold1/images/*.png   (3964 files)
    └── fold1/labels/*.npy   (3964 files)

输出：
  baselines/hovernet/conic_branch/exp_output/local/data_overlap/
    ├── images.npy           (19924, 256, 256, 3) uint8
    ├── labels.npy           (19924, 256, 256, 2) uint16
    └── splits.dat           {train-0: indices, valid-0: indices, test-0: indices}

依赖：
  - split_info.csv: patch_idx → (split, source) 的映射
  - patch_info.csv: patch_idx → patch_name 的映射

Split 策略：
  同一原始 patch 的 4 个 overlap 变体进入相同的 split。
  train/val/test 划分与队友统一的 split_info.csv 一致。
"""

import csv
import os
import sys

import cv2
import joblib
import numpy as np
from tqdm import tqdm

# 路径配置
OVERLAP_DIR = "/root/autodl-tmp/Pathology-Cell-Counting/data/processed/CoNIC_overlap/conic_cellvit_patient_x40_linear_withOverlap"
SPLIT_CSV = "/root/autodl-tmp/Pathology-Cell-Counting/data/processed/CoNIC/split_info.csv"
PATCH_CSV = "/root/autodl-tmp/Pathology-Cell-Counting/data/processed/CoNIC/patch_info.csv"
OUTPUT_DIR = "/root/autodl-tmp/Pathology-Cell-Counting/baselines/hovernet/conic_branch/exp_output/local/data_overlap"

# 每张原始 patch 的 overlap 变体数
NUM_OVERLAP = 4


def load_mappings():
    """构建 patch_name → (split, original_patch_idx) 映射"""
    # patch_info.csv: 第 0 行是 header "patch_info"，后面 4981 行是 patch name
    with open(PATCH_CSV) as f:
        lines_patch = f.read().strip().split("\n")
    patch_names = lines_patch[1:]  # 跳过 header

    # split_info.csv: patch_idx,split,source
    patch_idx_to_split = {}
    with open(SPLIT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pidx = int(row["patch_idx"])
            patch_idx_to_split[pidx] = row["split"]

    # 反向映射：patch_name → patch_idx
    patch_name_to_idx = {name: i for i, name in enumerate(patch_names)}

    return patch_names, patch_idx_to_split, patch_name_to_idx


def process_overlap_images():
    """主转换逻辑"""
    print("🔍 加载 split 映射...")
    patch_names, patch_idx_to_split, patch_name_to_idx = load_mappings()
    print(f"  patch_names: {len(patch_names)}, split entries: {len(patch_idx_to_split)}")

    # 收集所有 overlap 文件
    print("📂 扫描 overlap 文件...")
    all_files = []
    for fold_name in ["fold0", "fold1"]:
        img_dir = os.path.join(OVERLAP_DIR, fold_name, "images")
        if not os.path.isdir(img_dir):
            print(f"  ⚠ {fold_name}/images/ 不存在，跳过")
            continue
        for fname in os.listdir(img_dir):
            if not fname.endswith(".png"):
                continue
            # consep_10-0000_1.png → patch_name=consep_10-0000, overlap_idx=1
            base = fname[:-4]  # 去掉 .png
            parts = base.rsplit("_", 1)
            if len(parts) != 2:
                continue
            patch_name, overlap_str = parts
            overlap_idx = int(overlap_str)
            all_files.append((fold_name, patch_name, overlap_idx))

    print(f"  找到 {len(all_files)} 个文件")

    # 按 patch_idx 排序（同一原始 patch 的变体相邻）
    def sort_key(item):
        fold, patch_name, oidx = item
        pidx = patch_name_to_idx.get(patch_name, 99999)
        return (pidx, oidx)

    all_files.sort(key=sort_key)

    # 预分配数组
    N = len(all_files)
    H, W = 256, 256
    print(f"💾 预分配数组: ({N}, {H}, {W}, ...)")

    images = np.zeros((N, H, W, 3), dtype=np.uint8)
    labels = np.zeros((N, H, W, 2), dtype=np.uint16)

    # 收集每个 split 的索引
    split_indices = {"train": [], "val": [], "test": []}

    print("🔄 加载图像和标签...")
    for i, (fold_name, patch_name, overlap_idx) in enumerate(
        tqdm(all_files, desc="Converting")
    ):
        # 加载图像 (PNG → RGB → uint8)
        img_path = os.path.join(
            OVERLAP_DIR, fold_name, "images",
            f"{patch_name}_{overlap_idx}.png"
        )
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images[i] = img

        # 加载标签 (dict → stacked tensors)
        lab_path = os.path.join(
            OVERLAP_DIR, fold_name, "labels",
            f"{patch_name}_{overlap_idx}.npy"
        )
        lab_dict = np.load(lab_path, allow_pickle=True).item()
        inst_map = lab_dict["inst_map"].astype(np.uint16)
        type_map = lab_dict["type_map"].astype(np.uint16)
        labels[i, :, :, 0] = inst_map
        labels[i, :, :, 1] = type_map

        # 确定 split
        pidx = patch_name_to_idx.get(patch_name)
        if pidx is not None and pidx in patch_idx_to_split:
            sp = patch_idx_to_split[pidx]
            split_indices[sp].append(i)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存数据
    print("💾 保存 images.npy...")
    np.save(os.path.join(OUTPUT_DIR, "images.npy"), images)
    print(f"   → {os.path.join(OUTPUT_DIR, 'images.npy')}")
    print(f"   shape: {images.shape}, dtype: {images.dtype}")

    print("💾 保存 labels.npy...")
    np.save(os.path.join(OUTPUT_DIR, "labels.npy"), labels)
    print(f"   → {os.path.join(OUTPUT_DIR, 'labels.npy')}")
    print(f"   shape: {labels.shape}, dtype: {labels.dtype}")

    # 验证数据
    print("🔬 验证数据完整性...")
    _validate(images, labels, labels.shape)

    # 创建 splits.dat
    print("📋 生成 splits.dat...")
    splits = [{
        "train-0": np.array(split_indices["train"], dtype=np.int64),
        "valid-0": np.array(split_indices["val"], dtype=np.int64),
    }]
    splits_path = os.path.join(OUTPUT_DIR, "splits.dat")
    joblib.dump(splits, splits_path)
    print(f"   → {splits_path}")

    # 打印统计
    print("\n📊 Split 统计:")
    print(f"   train: {len(split_indices['train']):,} ("
          f"{len(split_indices['train']) // NUM_OVERLAP} patches × {NUM_OVERLAP})")
    print(f"   val:   {len(split_indices['val']):,} ("
          f"{len(split_indices['val']) // NUM_OVERLAP} patches × {NUM_OVERLAP})")
    print(f"   test:  {len(split_indices['test']):,} ("
          f"{len(split_indices['test']) // NUM_OVERLAP} patches × {NUM_OVERLAP})")
    print(f"   total: {N:,} ({N // NUM_OVERLAP} patches × {NUM_OVERLAP})")

    # 检查未匹配的 patch
    missing = [n for _, n, _ in all_files if n not in patch_name_to_idx]
    if missing:
        print(f"  ⚠ {len(missing)} 个文件未匹配到 patch_name:")
        for m in sorted(set(missing))[:10]:
            print(f"      {m}")
        if len(set(missing)) > 10:
            print(f"      ... 等 {len(set(missing))} 个")

    print("\n✅ 转换完成!")


def _validate(images, labels, orig_shape):
    """验证转换后的数据"""
    # 随机抽样检查
    rng = np.random.default_rng(42)
    check_indices = rng.choice(len(images), size=min(20, len(images)), replace=False)
    for idx in check_indices:
        img = images[idx]
        lab = labels[idx]
        assert img.shape == (256, 256, 3), f"Bad img shape at {idx}: {img.shape}"
        assert lab.shape == (256, 256, 2), f"Bad lab shape at {idx}: {lab.shape}"
        assert img.dtype == np.uint8, f"Bad img dtype at {idx}: {img.dtype}"
        assert lab.dtype == np.uint16, f"Bad lab dtype at {idx}: {lab.dtype}"

    # 检查 labels 是否和原始的 labels.npy 格式一致
    # (N, H, W, 2) uint16 — 应与原始格式一致
    assert labels.ndim == 4, f"Labels should be 4D, got {labels.ndim}D"
    assert labels.shape[1:] == (256, 256, 2), f"Bad label shape: {labels.shape}"

    print(f"   ✅ 随机抽查 {len(check_indices)} 张通过")


if __name__ == "__main__":
    process_overlap_images()
