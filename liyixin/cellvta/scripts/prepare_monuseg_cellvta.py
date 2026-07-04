#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw


def parse_split(split_json: Path) -> dict[str, list[str]]:
    data = json.loads(split_json.read_text(encoding="utf-8"))
    return {
        "train": [item["id"] for item in data["train"]],
        "val": [item["id"] for item in data["val"]],
        "test": [item["id"] for item in data["test"]],
    }


def xml_to_instance_mask(xml_path: Path, output_size: int = 1024) -> np.ndarray:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    regions_parent = root.find(".//Regions")
    binary_mask = np.zeros((1000, 1000), dtype=np.int32)
    if regions_parent is None:
        return np.zeros((output_size, output_size), dtype=np.int32)

    element_idx = 1
    for region in regions_parent.findall("Region"):
        vertices_node = region.find("Vertices")
        if vertices_node is None:
            continue
        coords = []
        for vertex in vertices_node.findall("Vertex"):
            try:
                x = float(vertex.attrib["X"])
                y = float(vertex.attrib["Y"])
            except Exception:
                continue
            coords.append((x, y))
        if len(coords) < 3:
            continue
        canvas = Image.new("I", (1000, 1000), 0)
        drawer = ImageDraw.Draw(canvas)
        drawer.polygon(coords, outline=element_idx, fill=element_idx)
        region_mask = np.array(canvas, dtype=np.int32)
        binary_mask[region_mask > 0] = element_idx
        element_idx += 1

    resized_mask = np.array(
        Image.fromarray(binary_mask).resize(
            (output_size, output_size), resample=Image.Resampling.NEAREST
        )
    ).astype(np.int32)
    return resized_mask


def convert_image(img_path: Path, output_size: int = 1024) -> Image.Image:
    image = Image.open(img_path).convert("RGB")
    return image.resize((output_size, output_size), resample=Image.Resampling.LANCZOS)


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def prepare_split(
    ids: list[str],
    image_dir: Path,
    ann_dir: Path,
    out_dir: Path,
    output_size: int,
) -> None:
    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    ensure_clean_dir(images_out)
    ensure_clean_dir(labels_out)

    for image_id in ids:
        img_path = image_dir / f"{image_id}.tif"
        xml_path = ann_dir / f"{image_id}.xml"
        if not img_path.is_file():
            raise FileNotFoundError(f"Missing image: {img_path}")
        if not xml_path.is_file():
            raise FileNotFoundError(f"Missing annotation: {xml_path}")

        image = convert_image(img_path, output_size=output_size)
        mask = xml_to_instance_mask(xml_path, output_size=output_size)

        image.save(images_out / f"{image_id}.png")
        np.save(labels_out / f"{image_id}.npy", mask)


def write_dataset_config(output_root: Path) -> None:
    config = {
        "tissue_types": {"MoNuSeg": 0},
        "nuclei_types": {
            "background": 0,
            "nucleus": 1,
        },
    }
    with open(output_root / "dataset_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=Path, required=True)
    parser.add_argument("--test_root", type=Path, required=True)
    parser.add_argument("--split_json", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--output_size", type=int, default=1024)
    args = parser.parse_args()

    split = parse_split(args.split_json)
    output_root = args.output_root.resolve()
    ensure_clean_dir(output_root)

    train_image_dir = args.train_root / "Tissue Images"
    train_ann_dir = args.train_root / "Annotations"
    test_image_dir = args.test_root
    test_ann_dir = args.test_root

    prepare_split(
        split["train"],
        train_image_dir,
        train_ann_dir,
        output_root / "train",
        args.output_size,
    )
    prepare_split(
        split["val"],
        train_image_dir,
        train_ann_dir,
        output_root / "val",
        args.output_size,
    )
    prepare_split(
        split["test"],
        test_image_dir,
        test_ann_dir,
        output_root / "test",
        args.output_size,
    )
    write_dataset_config(output_root)

    metadata = {
        "split_json": str(args.split_json.resolve()),
        "train_root": str(args.train_root.resolve()),
        "test_root": str(args.test_root.resolve()),
        "output_size": args.output_size,
        "counts": {k: len(v) for k, v in split.items()},
    }
    (output_root / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
