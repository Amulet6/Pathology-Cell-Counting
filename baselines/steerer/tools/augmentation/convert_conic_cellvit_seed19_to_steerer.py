#!/usr/bin/env python3
"""Convert the CellViT-style CoNIC seed19 release to STEERER point format.

Input release layout:
  data/conic_cellvit_patient_x40_linear_withOverlap.zip
  data/conic_cellvit_patient.zip
  docs/conic_split_seed19.json

The label .npy files are object arrays containing:
  {"inst_map": HxW instance ids, "type_map": HxW type ids}
"""

from __future__ import annotations

import argparse
import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from label_to_centroids import instance_centroids as mask_to_centroids
from original_augment_common import apply_union_pixel_augment, sample_image_transform, transform_points


SPLIT_ID_OFFSETS = {"train": 0, "val": 100000, "test": 200000}
ZIP_NAMES = {
    "conic_cellvit_patient_x40_linear_withOverlap": "conic_cellvit_patient_x40_linear_withOverlap.zip",
    "conic_cellvit_patient": "conic_cellvit_patient.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", required=True, help="Path containing data/ and docs/")
    parser.add_argument("--output", required=True, help="Output STEERER ProcessedData directory")
    parser.add_argument("--augment", choices=("none", "unified"), default="none")
    parser.add_argument("--num-augments", type=int, default=1)
    parser.add_argument("--include-original", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patch-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=3035)
    parser.add_argument("--box-size", type=int, default=16)
    parser.add_argument("--scale-category", type=int, default=0)
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--scale-min", type=float, default=0.8)
    parser.add_argument("--scale-max", type=float, default=1.2)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--vflip-prob", type=float, default=0.5)
    parser.add_argument("--affine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--affine-prob", type=float, default=1.0)
    parser.add_argument("--rotate-deg", type=float, default=179.0)
    parser.add_argument("--translate-frac", type=float, default=0.01)
    parser.add_argument("--shear-deg", type=float, default=5.0)
    parser.add_argument("--affine-scale-min", type=float, default=0.8)
    parser.add_argument("--affine-scale-max", type=float, default=1.2)
    parser.add_argument("--pixel-aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--blur-noise-prob", type=float, default=1.0)
    parser.add_argument("--color-aug-prob", type=float, default=1.0)
    parser.add_argument("--pad-value", type=int, default=0)
    return parser.parse_args()


def resolve_release_root(path: Path) -> Path:
    path = path.resolve()
    if (path / "docs" / "conic_split_seed19.json").is_file():
        return path
    matches = list(path.rglob("docs/conic_split_seed19.json"))
    if not matches:
        raise FileNotFoundError(f"Cannot find docs/conic_split_seed19.json under {path}")
    return matches[0].parents[1]


def open_zips(release_root: Path) -> dict[str, zipfile.ZipFile]:
    zips = {}
    for key, name in ZIP_NAMES.items():
        path = release_root / "data" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        zips[key] = zipfile.ZipFile(path)
    return zips


def read_image(zips: dict[str, zipfile.ZipFile], relpath: str) -> Image.Image:
    zip_key = relpath.split("/", 1)[0]
    return Image.open(io.BytesIO(zips[zip_key].read(relpath))).convert("RGB")


def read_label(zips: dict[str, zipfile.ZipFile], relpath: str) -> dict[str, np.ndarray]:
    zip_key = relpath.split("/", 1)[0]
    return np.load(io.BytesIO(zips[zip_key].read(relpath)), allow_pickle=True).item()


def instance_centroids(label: dict[str, np.ndarray]) -> tuple[list[list[float]], list[int]]:
    inst_map = np.asarray(label["inst_map"])
    type_map = np.asarray(label.get("type_map", np.zeros_like(inst_map)))
    points = mask_to_centroids(inst_map)
    class_ids: list[int] = []

    instance_ids = np.unique(inst_map)
    instance_ids = instance_ids[instance_ids > 0]
    for instance_id in instance_ids:
        ys, xs = np.where(inst_map == instance_id)
        if len(xs) == 0:
            continue

        values, counts = np.unique(type_map[ys, xs], return_counts=True)
        keep = values > 0
        if keep.any():
            values = values[keep]
            counts = counts[keep]
        class_ids.append(int(values[np.argmax(counts)]) if len(values) else 0)
    return points, class_ids


def clamp_round_point(point: list[float], width: int, height: int) -> list[int]:
    x = min(max(int(round(point[0])), 0), width - 1)
    y = min(max(int(round(point[1])), 0), height - 1)
    return [x, y]


def write_sample(
    output_dir: Path,
    split: str,
    image_id: str,
    image: Image.Image,
    points: list[list[float]],
    labels: list[int],
    box_size: int,
    scale_category: int,
) -> tuple[str, int]:
    image_out = output_dir / "images"
    json_out = output_dir / "jsons"
    image_out.mkdir(parents=True, exist_ok=True)
    json_out.mkdir(parents=True, exist_ok=True)

    image_name = f"{image_id}.png"
    image.save(image_out / image_name)
    width, height = image.size
    int_points = [clamp_round_point(point, width, height) for point in points]
    boxes = [
        [x - box_size // 2, y - box_size // 2, x + box_size // 2, y + box_size // 2]
        for x, y in int_points
    ]
    record = {
        "img_id": image_name,
        "human_num": len(int_points),
        "points": points,
        "boxes": boxes,
        "labels": labels,
        "split": split,
    }
    (json_out / f"{image_id}.json").write_text(json.dumps(record), encoding="utf-8")

    loc_items = [image_id, str(len(int_points))]
    for x, y in int_points:
        loc_items.extend([str(x), str(y), str(box_size), str(box_size), str(scale_category)])
    return " ".join(loc_items), len(int_points)


def convert_original_item(
    zips: dict[str, zipfile.ZipFile],
    output_dir: Path,
    split: str,
    item: dict,
    image_id: str,
    args: argparse.Namespace,
) -> tuple[str, int]:
    image = read_image(zips, item["image_relpath"])
    label = read_label(zips, item["label_relpath"])
    points, labels = instance_centroids(label)
    return write_sample(
        output_dir, split, image_id, image, points, labels, args.box_size, args.scale_category
    )


def convert_augmented_item(
    zips: dict[str, zipfile.ZipFile],
    output_dir: Path,
    split: str,
    item: dict,
    image_id: str,
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[str, int]:
    image = read_image(zips, item["image_relpath"])
    label = read_label(zips, item["label_relpath"])
    points, labels = instance_centroids(label)

    patch, transform = sample_image_transform(
        image,
        rng=rng,
        patch_size=args.patch_size,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        hflip_prob=args.hflip_prob,
        vflip_prob=args.vflip_prob,
        pad_value=args.pad_value,
        affine_enabled=args.affine,
        affine_prob=args.affine_prob,
        rotate_deg=args.rotate_deg,
        translate_frac=args.translate_frac,
        shear_deg=args.shear_deg,
        affine_scale_min=args.affine_scale_min,
        affine_scale_max=args.affine_scale_max,
    )
    transformed_points: list[list[float]] = []
    transformed_labels: list[int] = []
    for point, label_id in zip(points, labels):
        new_points = transform_points([point], transform)
        if new_points:
            transformed_points.append(new_points[0])
            transformed_labels.append(label_id)

    patch = apply_union_pixel_augment(
        patch,
        rng=rng,
        pixel_aug_enabled=args.pixel_aug,
        blur_noise_prob=args.blur_noise_prob,
        color_aug_prob=args.color_aug_prob,
    )
    return write_sample(
        output_dir,
        split,
        image_id,
        patch,
        transformed_points,
        transformed_labels,
        args.box_size,
        args.scale_category,
    )


def main() -> None:
    args = parse_args()
    release_root = resolve_release_root(Path(args.release_root))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_json = json.loads((release_root / "docs" / "conic_split_seed19.json").read_text(encoding="utf-8"))
    zips = open_zips(release_root)
    rng = random.Random(args.seed)

    try:
        summary: dict[str, tuple[int, int]] = {}
        for split in ("train", "val", "test"):
            items = split_json["splits"][split]
            if args.limit_per_split > 0:
                items = items[: args.limit_per_split]

            ids: list[str] = []
            gt_lines: list[str] = []
            total_points = 0
            offset = SPLIT_ID_OFFSETS[split]
            output_index = 0

            for item in items:
                if split == "train" and args.augment == "unified":
                    if args.include_original:
                        image_id = f"{offset + output_index:06d}"
                        gt_line, point_count = convert_original_item(
                            zips, output_dir, split, item, image_id, args
                        )
                        ids.append(image_id)
                        gt_lines.append(gt_line)
                        total_points += point_count
                        output_index += 1
                    for _ in range(args.num_augments):
                        image_id = f"{offset + output_index:06d}"
                        gt_line, point_count = convert_augmented_item(
                            zips, output_dir, split, item, image_id, rng, args
                        )
                        ids.append(image_id)
                        gt_lines.append(gt_line)
                        total_points += point_count
                        output_index += 1
                else:
                    image_id = f"{offset + output_index:06d}"
                    gt_line, point_count = convert_original_item(
                        zips, output_dir, split, item, image_id, args
                    )
                    ids.append(image_id)
                    gt_lines.append(gt_line)
                    total_points += point_count
                    output_index += 1

            (output_dir / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
            (output_dir / f"{split}_gt_loc.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
            summary[split] = (len(ids), total_points)

        (output_dir / "conversion_summary.json").write_text(
            json.dumps(
                {
                    "release_root": str(release_root),
                    "source_split": "docs/conic_split_seed19.json",
                    "augment": args.augment,
                    "num_augments": args.num_augments,
                    "include_original": args.include_original,
                    "patch_size": args.patch_size,
                    "summary": summary,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("Converted CoNIC CellViT seed19 release:")
        for split, (image_count, point_count) in summary.items():
            print(f"  {split}: {image_count} images, {point_count} points")
        print(f"Output: {output_dir}")
    finally:
        for zf in zips.values():
            zf.close()


if __name__ == "__main__":
    main()
