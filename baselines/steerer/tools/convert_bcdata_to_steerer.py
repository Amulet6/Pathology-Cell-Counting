#!/usr/bin/env python3
"""Convert BCData into the processed layout expected by STEERER.

BCData layout:
  BCData/
    images/{train,validation,test}/*.png
    annotations/{train,validation,test}/{positive,negative}/*.h5

Output layout:
  BCData/
    images/*.png
    jsons/*.json
    train.txt
    val.txt
    test.txt
    train_gt_loc.txt
    val_gt_loc.txt
    test_gt_loc.txt
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import h5py
from PIL import Image


SPLIT_MAP = {
    "train": "train",
    "validation": "val",
    "test": "test",
}

SPLIT_ID_OFFSETS = {
    "train": 0,
    "validation": 100000,
    "test": 200000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to BCData.zip or extracted BCData directory")
    parser.add_argument("--output", required=True, help="Output ProcessedData/BCData directory")
    parser.add_argument(
        "--count-mode",
        choices=("all", "positive", "negative"),
        default="all",
        help="Which BCData annotations to count. Default counts all annotated cells.",
    )
    parser.add_argument("--box-size", type=int, default=16, help="Pseudo box size for point annotations")
    parser.add_argument("--category", type=int, default=0, help="STEERER scale category written to *_gt_loc.txt")
    parser.add_argument("--limit-per-split", type=int, default=0, help="Only convert first N images per split for smoke tests")
    return parser.parse_args()


class BCDataReader:
    def __init__(self, root: Path):
        self.root = root
        self.zip_file: zipfile.ZipFile | None = None
        if root.is_file():
            self.zip_file = zipfile.ZipFile(root)
            self.prefix = self._detect_zip_prefix()
        else:
            self.prefix = ""

    def close(self) -> None:
        if self.zip_file is not None:
            self.zip_file.close()

    def _detect_zip_prefix(self) -> str:
        assert self.zip_file is not None
        names = self.zip_file.namelist()
        return "BCData/" if any(name.startswith("BCData/") for name in names) else ""

    def list_images(self, split: str) -> list[str]:
        if self.zip_file is not None:
            stem = f"{self.prefix}images/{split}/"
            names = [
                name[len(stem) :]
                for name in self.zip_file.namelist()
                if name.startswith(stem) and name.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        else:
            image_dir = self.root / "images" / split
            names = [path.name for path in image_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
        return sorted(names, key=natural_key)

    def read_image(self, split: str, name: str) -> bytes:
        if self.zip_file is not None:
            return self.zip_file.read(f"{self.prefix}images/{split}/{name}")
        return (self.root / "images" / split / name).read_bytes()

    def image_size(self, split: str, name: str) -> tuple[int, int]:
        with Image.open(io.BytesIO(self.read_image(split, name))) as image:
            return image.size

    def read_coords(self, split: str, label: str, image_name: str) -> list[list[int]]:
        h5_name = Path(image_name).with_suffix(".h5").name
        if self.zip_file is not None:
            raw = self.zip_file.read(f"{self.prefix}annotations/{split}/{label}/{h5_name}")
            handle = io.BytesIO(raw)
            with h5py.File(handle, "r") as h5_file:
                return h5_file["coordinates"][:].astype(int).tolist()
        with h5py.File(self.root / "annotations" / split / label / h5_name, "r") as h5_file:
            return h5_file["coordinates"][:].astype(int).tolist()


def natural_key(name: str) -> tuple[int, str]:
    stem = Path(name).stem
    return (int(stem), name) if stem.isdigit() else (10**12, name)


def selected_labels(count_mode: str) -> Iterable[str]:
    if count_mode == "all":
        return ("positive", "negative")
    return (count_mode,)


def clamp_point(point: list[int], width: int, height: int) -> list[int]:
    x, y = int(point[0]), int(point[1])
    x = min(max(x, 0), width - 1)
    y = min(max(y, 0), height - 1)
    return [x, y]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    image_out = output_dir / "images"
    json_out = output_dir / "jsons"
    image_out.mkdir(parents=True, exist_ok=True)
    json_out.mkdir(parents=True, exist_ok=True)

    reader = BCDataReader(input_path)
    try:
        summary: dict[str, tuple[int, int]] = {}
        for source_split, target_split in SPLIT_MAP.items():
            image_names = reader.list_images(source_split)
            if args.limit_per_split > 0:
                image_names = image_names[: args.limit_per_split]

            ids: list[str] = []
            gt_lines: list[str] = []
            total_points = 0

            for index, image_name in enumerate(image_names):
                image_id = f"{SPLIT_ID_OFFSETS[source_split] + index:06d}"
                width, height = reader.image_size(source_split, image_name)

                points: list[list[int]] = []
                labels: list[str] = []
                for label in selected_labels(args.count_mode):
                    coords = reader.read_coords(source_split, label, image_name)
                    points.extend(clamp_point(point, width, height) for point in coords)
                    labels.extend([label] * len(coords))

                raw_image = reader.read_image(source_split, image_name)
                output_image_name = f"{image_id}{Path(image_name).suffix.lower()}"
                (image_out / output_image_name).write_bytes(raw_image)

                boxes = [
                    [point[0] - args.box_size // 2, point[1] - args.box_size // 2, point[0] + args.box_size // 2, point[1] + args.box_size // 2]
                    for point in points
                ]
                record = {
                    "img_id": output_image_name,
                    "human_num": len(points),
                    "points": points,
                    "boxes": boxes,
                    "labels": labels,
                }
                (json_out / f"{image_id}.json").write_text(json.dumps(record), encoding="utf-8")

                loc_items: list[str] = [image_id, str(len(points))]
                for x, y in points:
                    loc_items.extend([str(x), str(y), str(args.box_size), str(args.box_size), str(args.category)])
                gt_lines.append(" ".join(loc_items))
                ids.append(image_id)
                total_points += len(points)

            (output_dir / f"{target_split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
            (output_dir / f"{target_split}_gt_loc.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
            summary[target_split] = (len(ids), total_points)

        print("Converted BCData:")
        for split, (image_count, point_count) in summary.items():
            print(f"  {split}: {image_count} images, {point_count} points")
        print(f"Output: {output_dir}")
    finally:
        reader.close()


if __name__ == "__main__":
    main()
