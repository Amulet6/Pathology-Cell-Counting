"""
MoNuSeg 数据预处理：XML → .mat inst_map，适配官方 extract_patches.py。

流程：
1. 解析每个 XML 的多边形标注 → 栅格化为 inst_map
2. TIFF → PNG 副本（官方 __Kumar loader 只认 .png）
3. 按 split JSON 分配到 Train / Valid / Test 目录
4. 输出目录结构可直接被 extract_patches.py 使用
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio


def parse_xml_polygons(xml_path):
    """解析 MoNuSeg XML，返回多边形顶点列表。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polygons = []

    for region in root.findall(".//Region"):
        vertices = []
        for vertex in region.findall(".//Vertex"):
            x = vertex.get("X")
            y = vertex.get("Y")
            if x is None or y is None:
                continue
            vertices.append([int(float(x)), int(float(y))])
        if len(vertices) >= 3:
            polygons.append(np.array(vertices, dtype=np.int32))

    return polygons


def rasterize(polygons, h, w):
    """多边形列表 → 像素级 inst_map（ID 从 1 开始）。"""
    inst_map = np.zeros((h, w), dtype=np.int32)
    for idx, poly in enumerate(polygons, 1):
        cv2.fillPoly(inst_map, [poly], idx)
    return inst_map


def process_one(tif_path, xml_path, png_out_dir, mat_out_dir):
    """TIFF + XML → PNG 副本 + .mat inst_map。"""
    img = cv2.imread(str(tif_path))
    if img is None:
        raise RuntimeError(f"无法读取: {tif_path}")
    h, w = img.shape[:2]

    # PNG 副本
    stem = tif_path.stem
    png_path = os.path.join(png_out_dir, stem + ".png")
    cv2.imwrite(png_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # inst_map → .mat
    polygons = parse_xml_polygons(xml_path)
    inst_map = rasterize(polygons, h, w)

    mat_path = os.path.join(mat_out_dir, stem + ".mat")
    sio.savemat(mat_path, {"inst_map": inst_map})

    print(f"  {stem}: {h}×{w}, {len(polygons)} nuclei → {png_path}")
    return len(polygons)


def main():
    parser = argparse.ArgumentParser(
        description="MoNuSeg XML → .mat inst_map，适配官方 extract_patches.py"
    )
    parser.add_argument("--split_json", required=True,
                        help="monuseg_split.json 路径")
    parser.add_argument("--train_image_dir", required=True,
                        help="MoNuSeg 2018 Training Data/Tissue Images/")
    parser.add_argument("--train_annot_dir", required=True,
                        help="MoNuSeg 2018 Training Data/Annotations/")
    parser.add_argument("--test_dir", required=True,
                        help="MoNuSegTestData/ (含 .tif 和 .xml)")
    parser.add_argument("--output_root", required=True,
                        help="输出根目录，生成 Train/Valid/Test 子目录")
    args = parser.parse_args()

    with open(args.split_json) as f:
        split = json.load(f)

    train_img = Path(args.train_image_dir)
    train_ann = Path(args.train_annot_dir)
    test_dir = Path(args.test_dir)
    out = Path(args.output_root)

    total = 0

    # --- Train (30) ---
    print("=" * 50); print("Train (30 images)"); print("=" * 50)
    for item in split["train"]:
        tid = item["id"]
        os.makedirs(out / "Train" / "Images", exist_ok=True)
        os.makedirs(out / "Train" / "Labels", exist_ok=True)
        n = process_one(
            train_img / f"{tid}.tif",
            train_ann / f"{tid}.xml",
            out / "Train" / "Images",
            out / "Train" / "Labels",
        )
        total += n

    # --- Valid (7) ---
    print("=" * 50); print("Valid (7 images)"); print("=" * 50)
    for item in split["val"]:
        tid = item["id"]
        os.makedirs(out / "Valid" / "Images", exist_ok=True)
        os.makedirs(out / "Valid" / "Labels", exist_ok=True)
        n = process_one(
            train_img / f"{tid}.tif",
            train_ann / f"{tid}.xml",
            out / "Valid" / "Images",
            out / "Valid" / "Labels",
        )
        total += n

    # --- Test (14) ---
    print("=" * 50); print("Test (14 images)"); print("=" * 50)
    for item in split["test"]:
        tid = item["id"]
        os.makedirs(out / "Test" / "Images", exist_ok=True)
        os.makedirs(out / "Test" / "Labels", exist_ok=True)
        n = process_one(
            test_dir / f"{tid}.tif",
            test_dir / f"{tid}.xml",
            out / "Test" / "Images",
            out / "Test" / "Labels",
        )
        total += n

    print("=" * 50)
    print(f"完成。{total} nuclei total → {out}")
    print(f"目录结构: Train(30) / Valid(7) / Test(14)")
    print(f"下一步: 运行 extract_patches.py (kumar mode)")


if __name__ == "__main__":
    main()
