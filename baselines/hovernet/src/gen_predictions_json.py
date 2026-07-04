#!/usr/bin/env python3
"""
生成统一点匹配评估所需的 predictions.json。

支持：
  - MoNuSeg GT：统一用 label_to_centroids.py monuseg 模式（多边形质心），再按 split 过滤
  - MoNuSeg Pred：从 run_infer.py 输出的 .mat inst_centroid 读取
  - CoNIC GT：统一用 label_to_centroids.py conic 模式
  - CoNIC Pred：从 seg_preds.npy inst_map 计算质心

输出格式草案（待队友确认）：
{
  "metadata": {
    "dataset": "MoNuSeg | CoNIC",
    "method": "HoVer-Net",
    "role": "gt | pred",
    "extraction_method": "ground_truth | watershed_hv_map + cv2_moments_centroid",
    "coordinate_order": "xy",
    "coordinate_unit": "pixel",
    "matching_thresholds_px": [6, 12, 24]
  },
  "samples": [
    {"id": "sample_name", "points": [[x1,y1], [x2,y2], ...]}
  ]
}
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio


def mask_centroids(inst_map):
    """从像素级 inst_map 计算质心。cv2.moments 等价于像素坐标算术平均。"""
    ids = np.unique(inst_map)
    ids = ids[ids > 0]
    points = np.zeros((len(ids), 2), dtype=np.float64)
    for i, uid in enumerate(ids):
        ys, xs = np.where(inst_map == uid)
        points[i, 0] = xs.mean()  # x
        points[i, 1] = ys.mean()  # y
    return points.tolist()


def gen_monuseg_pred(args):
    """MoNuSeg Pred：从 run_infer.py 输出的 .mat inst_centroid 读取。"""
    mat_dir = Path(args.mat_dir)

    samples = []
    for mat_path in sorted(mat_dir.glob("*.mat")):
        img_id = mat_path.stem
        data = sio.loadmat(str(mat_path))
        centroids = data["inst_centroid"].astype(np.float64)
        # inst_centroid 是 N×2, order = (x, y)
        points = centroids.tolist()
        samples.append({"id": img_id, "points": points})

    output = {
        "metadata": {
            "dataset": "MoNuSeg",
            "method": "HoVer-Net",
            "role": "pred",
            "extraction_method": "watershed_hv_map + cv2_moments_centroid",
            "coordinate_order": "xy",
            "coordinate_unit": "pixel",
            "matching_thresholds_px": [6, 12, 24],
        },
        "samples": samples,
    }
    return output


def gen_conic_pred(args):
    """CoNIC Pred：从 gen_official_preds.py 输出的 seg_preds.npy inst_map → 质心。"""
    seg_preds = np.load(args.seg_preds_npy, mmap_mode="r")  # (799, 256, 256, 2)
    # 使用 split_csv 获取 test patch 的真实索引（非连续）
    split_info = pd.read_csv(args.split_csv)
    test_idx = split_info[split_info["split"] == "test"]["patch_idx"].values.astype(int)

    samples = []
    for i, patch_idx in enumerate(test_idx):
        inst_map = seg_preds[i, ..., 0].astype(np.int32)
        points = mask_centroids(inst_map)
        samples.append({"id": f"conic_{patch_idx:05d}", "points": points})

    output = {
        "metadata": {
            "dataset": "CoNIC",
            "method": "HoVer-Net (HoVerNetExt)",
            "role": "pred",
            "extraction_method": "watershed_hv_map + cv2_moments_centroid + majority_vote_type",
            "coordinate_order": "xy",
            "coordinate_unit": "pixel",
            "matching_thresholds_px": [6, 12, 24],
        },
        "samples": samples,
    }
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions.json for centroid evaluation"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # MoNuSeg Pred
    p = sub.add_parser("monuseg_pred")
    p.add_argument("--mat_dir", required=True, help="run_infer.py output .mat directory")
    p.add_argument("--output", required=True)
    p.set_defaults(func=lambda a: save_json(gen_monuseg_pred(a), a.output))

    # CoNIC Pred
    p = sub.add_parser("conic_pred")
    p.add_argument("--seg_preds_npy", required=True)
    p.add_argument("--split_csv", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=lambda a: save_json(gen_conic_pred(a), a.output))

    args = parser.parse_args()
    args.func(args)


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    n_samples = len(data["samples"])
    n_points = sum(s["count"] if "count" in s else len(s["points"]) for s in data["samples"])
    print(f"Saved: {path}  ({n_samples} samples, {n_points} points)")


if __name__ == "__main__":
    main()
