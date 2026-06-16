"""
MoNuSeg 补丁提取（复用官方 PatchExtractor，不修改 official/ 目录）。

将已转换的 PNG + .mat 全图提取为训练补丁，
输出与官方 FileLoader 兼容的 .npy 文件：channels = [R, G, B, inst].
"""

import glob
import os
import sys

import cv2
import numpy as np
import scipy.io as sio
import tqdm

# 把 official/ 加入路径以导入 PatchExtractor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "official"))
from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir


def main():
    WIN_SIZE = [540, 540]
    STEP_SIZE = [164, 164]
    EXTRACT_TYPE = "mirror"

    dataset_root = os.path.join(
        os.path.dirname(__file__), "..", "official", "dataset", "MoNuSeg"
    )
    save_root = os.path.join(
        os.path.dirname(__file__), "..", "official", "dataset", "training_data", "monuseg"
    )

    xtractor = PatchExtractor(WIN_SIZE, STEP_SIZE)

    for split_name in ["Train", "Valid"]:
        img_dir = os.path.join(dataset_root, split_name, "Images")
        ann_dir = os.path.join(dataset_root, split_name, "Labels")

        out_dir = os.path.join(
            save_root, split_name.lower(),
            f"{WIN_SIZE[0]}x{WIN_SIZE[1]}_{STEP_SIZE[0]}x{STEP_SIZE[1]}"
        )
        rm_n_mkdir(out_dir)

        img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        print(f"\n{split_name}: {len(img_files)} images → {out_dir}")

        for img_path in tqdm.tqdm(img_files, desc=split_name):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            ann_path = os.path.join(ann_dir, stem + ".mat")

            # 加载图像
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            # 加载 inst_map 并扩展为单通道
            ann = sio.loadmat(ann_path)["inst_map"].astype("int32")
            ann = np.expand_dims(ann, axis=-1)  # (H, W, 1)

            # 拼接为 (H, W, 4) = RGB + inst
            img = np.concatenate([img.astype("int32"), ann], axis=-1)

            # 提取补丁
            sub_patches = xtractor.extract(img, EXTRACT_TYPE)

            for idx, patch in enumerate(sub_patches):
                np.save(f"{out_dir}/{stem}_{idx:03d}.npy", patch)

    print("\n完成。补丁输出目录:")
    for split in ["train", "valid"]:
        d = os.path.join(save_root, split,
                         f"{WIN_SIZE[0]}x{WIN_SIZE[1]}_{STEP_SIZE[0]}x{STEP_SIZE[1]}")
        n = len(glob.glob(os.path.join(d, "*.npy")))
        print(f"  {split}: {n} patches")


if __name__ == "__main__":
    main()
