#!/usr/bin/env python3
"""
Regenerate UNIFIED point annotations for MoNuSeg and CoNIC in APGCC txt format.

Centroid logic is imported verbatim from ``label_to_centroids.py`` (the agreed,
team-unified script), so points are reproducible across teammates:
  - MoNuSeg : polygon_centroid (area-weighted polygon centroid)
  - CoNIC   : pixel-mean of each instance mask

Output (replaces any previous *_gt/*.txt + *.list):
  <root>/<split>/<id>.png         (MoNuSeg: kept; CoNIC: symlink to release png)
  <root>/<split>_gt/<id>.txt      "x y" per nucleus, 2 decimals
  <root>/<split>.list             "<split>/<id>.png <split>_gt/<id>.txt"

This driver does NOT touch the images themselves for MoNuSeg (already present);
for CoNIC it extracts the referenced members from the release zips and symlinks
the images.
"""

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np

from label_to_centroids import read_monuseg_xml, instance_centroids


# --------------------------------------------------------------------------- #
# shared
# --------------------------------------------------------------------------- #
def write_gt(gt_path, points):
    with open(gt_path, "w") as f:
        for x, y in points:
            f.write("%.2f %.2f\n" % (x, y))


def write_list(list_path, lines):
    with open(list_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# MoNuSeg
# --------------------------------------------------------------------------- #
def regen_monuseg(root, split_json, train_ann, test_dir):
    with open(split_json) as f:
        split = json.load(f)

    # (split_name, list of ids, annotation dir)
    plans = [
        ("train", [x["id"] for x in split["train"]], train_ann),
        ("val", [x["id"] for x in split["val"]], train_ann),
        ("test", [x["id"] for x in split["test"]], test_dir),
    ]

    for split_name, ids, ann_dir in plans:
        gt_dir = os.path.join(root, split_name + "_gt")
        os.makedirs(gt_dir, exist_ok=True)
        lines = []
        total = 0
        for stem in ids:
            xml_path = os.path.join(ann_dir, stem + ".xml")
            points = read_monuseg_xml(xml_path, "polygon_centroid")
            write_gt(os.path.join(gt_dir, stem + ".txt"), points)
            total += len(points)
            rel_img = os.path.join(split_name, stem + ".png")
            rel_gt = os.path.join(split_name + "_gt", stem + ".txt")
            lines.append("%s %s" % (rel_img, rel_gt))
        write_list(os.path.join(root, split_name + ".list"), lines)
        print("[MoNuSeg/%s] %d images, %d points" % (split_name, len(ids), total))


# --------------------------------------------------------------------------- #
# CoNIC
# --------------------------------------------------------------------------- #
def load_inst_map(npy_path):
    arr = np.load(npy_path, allow_pickle=True)
    if arr.dtype == object:
        obj = arr.item() if arr.ndim == 0 else arr
        if isinstance(obj, dict):
            return np.asarray(obj["inst_map"])
    if arr.ndim == 3:
        return arr[..., 0]
    return arr


def extract_members(zip_path, members, dest_dir):
    """Extract only the given archive members (skip ones already present)."""
    todo = [m for m in members if not os.path.exists(os.path.join(dest_dir, m))]
    if not todo:
        return 0
    with zipfile.ZipFile(zip_path) as zf:
        for m in todo:
            zf.extract(m, dest_dir)
    return len(todo)


def regen_conic(root, split_json, release_data_dir, withoverlap_zip, patient_zip):
    with open(split_json) as f:
        data = json.load(f)
    splits = data["splits"]

    # zip prefix -> zip path
    zip_for = {
        "conic_cellvit_patient_x40_linear_withOverlap": withoverlap_zip,
        "conic_cellvit_patient": patient_zip,
    }

    for split_name in ("train", "val", "test"):
        samples = splits[split_name]

        # 1) ensure referenced members are extracted from the release zips
        by_zip = {}
        for s in samples:
            for rel in (s["image_relpath"], s["label_relpath"]):
                top = rel.split("/", 1)[0]
                by_zip.setdefault(zip_for[top], []).append(rel)
        for zip_path, members in by_zip.items():
            n = extract_members(zip_path, members, release_data_dir)
            print("[CoNIC/%s] extracted %d new members from %s"
                  % (split_name, n, os.path.basename(zip_path)))

        # 2) generate gt + symlink images + list
        img_dir = os.path.join(root, split_name)
        gt_dir = os.path.join(root, split_name + "_gt")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        lines = []
        total = 0
        for s in samples:
            stem = s["stem"]
            inst_map = load_inst_map(os.path.join(release_data_dir, s["label_relpath"]))
            points = instance_centroids(inst_map)
            write_gt(os.path.join(gt_dir, stem + ".txt"), points)
            total += len(points)

            src_png = os.path.join(release_data_dir, s["image_relpath"])
            link_png = os.path.join(img_dir, stem + ".png")
            if os.path.islink(link_png) or os.path.exists(link_png):
                os.remove(link_png)
            os.symlink(src_png, link_png)

            rel_img = os.path.join(split_name, stem + ".png")
            rel_gt = os.path.join(split_name + "_gt", stem + ".txt")
            lines.append("%s %s" % (rel_img, rel_gt))
        write_list(os.path.join(root, split_name + ".list"), lines)
        print("[CoNIC/%s] %d patches, %d points" % (split_name, len(samples), total))


# --------------------------------------------------------------------------- #
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["monuseg", "conic", "both"], default="both")

    # MoNuSeg
    ap.add_argument("--monuseg-root", default="/data1/llx/MoNuSegdata")
    ap.add_argument("--monuseg-split-json",
                    default=os.path.join(here, "apgcc", "datasets", "monuseg_split.json"))
    ap.add_argument("--monuseg-train-ann",
                    default="/data1/llx/MoNuSeg 2018 Training Data/Annotations")
    ap.add_argument("--monuseg-test-dir", default="/data1/llx/MoNuSegTestData")

    # CoNIC
    ap.add_argument("--conic-root", default="/data1/llx/CoNICdata")
    ap.add_argument("--conic-release",
                    default="/data1/llx/cellvta_conic_release_2026-06-09")
    args = ap.parse_args()

    if args.dataset in ("monuseg", "both"):
        regen_monuseg(args.monuseg_root, args.monuseg_split_json,
                      args.monuseg_train_ann, args.monuseg_test_dir)

    if args.dataset in ("conic", "both"):
        release = args.conic_release
        data_dir = os.path.join(release, "data")
        regen_conic(
            args.conic_root,
            os.path.join(release, "docs", "conic_split_seed19.json"),
            data_dir,
            os.path.join(data_dir, "conic_cellvit_patient_x40_linear_withOverlap.zip"),
            os.path.join(data_dir, "conic_cellvit_patient.zip"),
        )

    print("Done.")


if __name__ == "__main__":
    main()
