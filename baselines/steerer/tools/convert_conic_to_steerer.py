#!/usr/bin/env python3
"""Convert CoNIC instance labels into the processed layout expected by STEERER.

Supported input layouts:
  CoNIC/
    images.npy
    labels.npy

or:
  CoNIC/
    train/images.npy
    train/labels.npy
    val/images.npy
    val/labels.npy
    test/images.npy
    test/labels.npy

CoNIC labels are expected to be either:
  H x W instance maps
  H x W x 2 maps where channel 0 is instance id and channel 1 is class id
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


TARGET_SPLITS = ("train", "val", "test")
SPLIT_ID_OFFSETS = {
    "train": 0,
    "val": 100000,
    "test": 200000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to extracted CoNIC directory")
    parser.add_argument("--output", required=True, help="Output ProcessedData/CoNIC_numeric directory")
    parser.add_argument("--images", default=None, help="Optional explicit images.npy path")
    parser.add_argument("--labels", default=None, help="Optional explicit labels.npy path")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3035)
    parser.add_argument("--box-size", type=int, default=16, help="Pseudo box size for point annotations")
    parser.add_argument("--scale-category", type=int, default=0, help="STEERER scale category written to *_gt_loc.txt")
    parser.add_argument("--limit-per-split", type=int, default=0, help="Only convert first N samples per split for smoke tests")
    return parser.parse_args()


def find_split_arrays(root: Path) -> dict[str, tuple[Path, Path]]:
    split_aliases = {
        "train": ("train", "training"),
        "val": ("val", "valid", "validation"),
        "test": ("test", "testing"),
    }
    split_arrays: dict[str, tuple[Path, Path]] = {}
    for target_split, aliases in split_aliases.items():
        for alias in aliases:
            candidates = [
                (root / alias / "images.npy", root / alias / "labels.npy"),
                (root / f"{alias}_images.npy", root / f"{alias}_labels.npy"),
            ]
            for image_path, label_path in candidates:
                if image_path.is_file() and label_path.is_file():
                    split_arrays[target_split] = (image_path, label_path)
                    break
            if target_split in split_arrays:
                break
    return split_arrays


def load_array(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False, mmap_mode="r")


def split_indices(sample_count: int, train_ratio: float, val_ratio: float, seed: int) -> dict[str, np.ndarray]:
    indices = np.arange(sample_count)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    train_end = int(round(sample_count * train_ratio))
    val_end = train_end + int(round(sample_count * val_ratio))
    return {
        "train": np.sort(indices[:train_end]),
        "val": np.sort(indices[train_end:val_end]),
        "test": np.sort(indices[val_end:]),
    }


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[-1] > 3:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        max_value = float(np.max(image)) if image.size else 0.0
        if max_value <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def split_label_channels(label: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    label = np.asarray(label)
    if label.ndim == 2:
        return label.astype(np.int32), None
    if label.ndim == 3 and label.shape[-1] >= 2:
        return label[:, :, 0].astype(np.int32), label[:, :, 1].astype(np.int32)
    if label.ndim == 3 and label.shape[0] >= 2:
        return label[0].astype(np.int32), label[1].astype(np.int32)
    raise ValueError(f"Unsupported CoNIC label shape: {label.shape}")


def instance_points(label: np.ndarray) -> tuple[list[list[int]], list[int]]:
    instance_map, class_map = split_label_channels(label)
    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids > 0]

    points: list[list[int]] = []
    class_ids: list[int] = []
    for instance_id in instance_ids:
        ys, xs = np.where(instance_map == instance_id)
        if len(xs) == 0:
            continue
        x = int(round(float(xs.mean())))
        y = int(round(float(ys.mean())))
        points.append([x, y])

        if class_map is None:
            class_ids.append(0)
        else:
            values, counts = np.unique(class_map[ys, xs], return_counts=True)
            keep = values > 0
            if keep.any():
                values = values[keep]
                counts = counts[keep]
            class_ids.append(int(values[np.argmax(counts)]) if len(values) else 0)

    return points, class_ids


def convert_sample(
    image: np.ndarray,
    label: np.ndarray,
    image_id: str,
    image_out: Path,
    json_out: Path,
    box_size: int,
    scale_category: int,
) -> tuple[str, int]:
    image = normalize_image(image)
    points, class_ids = instance_points(label)

    output_image_name = f"{image_id}.png"
    Image.fromarray(image).save(image_out / output_image_name)

    boxes = [
        [point[0] - box_size // 2, point[1] - box_size // 2, point[0] + box_size // 2, point[1] + box_size // 2]
        for point in points
    ]
    record = {
        "img_id": output_image_name,
        "human_num": len(points),
        "points": points,
        "boxes": boxes,
        "labels": class_ids,
    }
    (json_out / f"{image_id}.json").write_text(json.dumps(record), encoding="utf-8")

    loc_items = [image_id, str(len(points))]
    for x, y in points:
        loc_items.extend([str(x), str(y), str(box_size), str(box_size), str(scale_category)])
    return " ".join(loc_items), len(points)


def convert_split(
    split: str,
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    output_dir: Path,
    box_size: int,
    scale_category: int,
    limit_per_split: int,
) -> tuple[int, int]:
    image_out = output_dir / "images"
    json_out = output_dir / "jsons"
    image_out.mkdir(parents=True, exist_ok=True)
    json_out.mkdir(parents=True, exist_ok=True)

    if limit_per_split > 0:
        indices = indices[:limit_per_split]

    ids: list[str] = []
    gt_lines: list[str] = []
    total_points = 0
    offset = SPLIT_ID_OFFSETS[split]

    for output_index, source_index in enumerate(indices):
        image_id = f"{offset + output_index:06d}"
        gt_line, point_count = convert_sample(
            images[source_index],
            labels[source_index],
            image_id,
            image_out,
            json_out,
            box_size,
            scale_category,
        )
        ids.append(image_id)
        gt_lines.append(gt_line)
        total_points += point_count

    (output_dir / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    (output_dir / f"{split}_gt_loc.txt").write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    return len(ids), total_points


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, tuple[int, int]] = {}
    if args.images and args.labels:
        images = load_array(Path(args.images))
        labels = load_array(Path(args.labels))
        split_map = split_indices(len(images), args.train_ratio, args.val_ratio, args.seed)
        for split in TARGET_SPLITS:
            summary[split] = convert_split(
                split,
                images,
                labels,
                split_map[split],
                output_dir,
                args.box_size,
                args.scale_category,
                args.limit_per_split,
            )
    else:
        split_arrays = find_split_arrays(input_dir)
        if split_arrays:
            for split in TARGET_SPLITS:
                if split not in split_arrays:
                    continue
                images = load_array(split_arrays[split][0])
                labels = load_array(split_arrays[split][1])
                summary[split] = convert_split(
                    split,
                    images,
                    labels,
                    np.arange(len(images)),
                    output_dir,
                    args.box_size,
                    args.scale_category,
                    args.limit_per_split,
                )
        else:
            image_path = input_dir / "images.npy"
            label_path = input_dir / "labels.npy"
            if not image_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(
                    "Could not find CoNIC arrays. Expected images.npy/labels.npy, "
                    "split subdirectories, or explicit --images/--labels.")
            images = load_array(image_path)
            labels = load_array(label_path)
            split_map = split_indices(len(images), args.train_ratio, args.val_ratio, args.seed)
            for split in TARGET_SPLITS:
                summary[split] = convert_split(
                    split,
                    images,
                    labels,
                    split_map[split],
                    output_dir,
                    args.box_size,
                    args.scale_category,
                    args.limit_per_split,
                )

    print("Converted CoNIC:")
    for split in TARGET_SPLITS:
        if split in summary:
            image_count, point_count = summary[split]
            print(f"  {split}: {image_count} images, {point_count} points")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
