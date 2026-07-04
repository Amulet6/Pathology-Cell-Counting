#!/usr/bin/env python3
"""Augment BCData in its original image/H5-coordinate layout."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from original_augment_common import IMAGE_SUFFIXES, apply_union_pixel_augment, sample_image_transform, transform_points


SPLITS = ("train", "validation", "test")
LABELS = ("positive", "negative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Augment original BCData. Input layout: images/{split}/*.png and "
            "annotations/{split}/{positive,negative}/*.h5."
        )
    )
    parser.add_argument("--input", required=True, help="Path to extracted original BCData directory")
    parser.add_argument("--output", required=True, help="Output original-layout BCData directory")
    parser.add_argument("--split", default="train", choices=SPLITS, help="Split to augment")
    parser.add_argument("--patch-size", type=int, default=256, help="Output square patch size")
    parser.add_argument("--num-augments", type=int, default=1, help="Augmented copies per source image")
    parser.add_argument("--scale-min", type=float, default=0.8)
    parser.add_argument("--scale-max", type=float, default=1.2)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--vflip-prob", type=float, default=0.5)
    parser.add_argument("--affine", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--affine-prob", type=float, default=1.0)
    parser.add_argument("--rotate-deg", type=float, default=179.0)
    parser.add_argument("--translate-frac", type=float, default=0.01)
    parser.add_argument("--shear-deg", type=float, default=5.0)
    parser.add_argument("--affine-scale-min", type=float, default=0.8)
    parser.add_argument("--affine-scale-max", type=float, default=1.2)
    parser.add_argument("--pixel-aug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--blur-noise-prob", type=float, default=1.0)
    parser.add_argument("--color-aug-prob", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=3035)
    parser.add_argument("--include-original", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--copy-eval-splits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-points", type=int, default=0, help="Retry/skip crops with fewer total points")
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--pad-value", type=int, default=0)
    parser.add_argument("--image-ext", default=".png", help="Extension used for augmented images")
    return parser.parse_args()


def list_images(root: Path, split: str) -> list[Path]:
    image_dir = root / "images" / split
    if not image_dir.exists():
        return []
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def read_coords(root: Path, split: str, label: str, image_name: str) -> list[list[float]]:
    import h5py

    h5_path = root / "annotations" / split / label / Path(image_name).with_suffix(".h5").name
    if not h5_path.exists():
        return []
    with h5py.File(h5_path, "r") as h5_file:
        if "coordinates" not in h5_file:
            return []
        coords = h5_file["coordinates"][:]
    return np.asarray(coords, dtype=np.float32).reshape(-1, 2).tolist()


def write_coords(output_root: Path, split: str, label: str, image_name: str, coords: list[list[float]]) -> None:
    import h5py

    out_dir = output_root / "annotations" / split / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(image_name).with_suffix(".h5").name
    data = np.asarray(coords, dtype=np.float32).reshape(-1, 2)
    with h5py.File(out_path, "w") as h5_file:
        h5_file.create_dataset("coordinates", data=data)


def copy_original_sample(input_root: Path, output_root: Path, split: str, image_path: Path) -> int:
    out_image_dir = output_root / "images" / split
    out_image_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, out_image_dir / image_path.name)

    total = 0
    for label in LABELS:
        coords = read_coords(input_root, split, label, image_path.name)
        write_coords(output_root, split, label, image_path.name, coords)
        total += len(coords)
    return total


def augment_one(
    image: Image.Image,
    coords_by_label: dict[str, list[list[float]]],
    *,
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[Image.Image, dict[str, list[list[float]]]] | None:
    for _ in range(max(1, args.max_attempts)):
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
        transformed = {
            label: transform_points(points, transform)
            for label, points in coords_by_label.items()
        }
        total_points = sum(len(points) for points in transformed.values())
        if total_points >= args.min_points:
            return patch, transformed
    return None


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    if input_root.resolve() == output_root.resolve():
        raise ValueError("--output must be different from --input")
    if args.image_ext and not args.image_ext.startswith("."):
        args.image_ext = "." + args.image_ext

    rng = random.Random(args.seed)
    summary: dict[str, tuple[int, int]] = {}

    for split in SPLITS:
        image_paths = list_images(input_root, split)
        if not image_paths:
            continue

        image_count = 0
        point_count = 0
        should_copy_split = split != args.split and args.copy_eval_splits
        should_augment_split = split == args.split

        if should_copy_split or (should_augment_split and args.include_original):
            for image_path in image_paths:
                point_count += copy_original_sample(input_root, output_root, split, image_path)
                image_count += 1

        if should_augment_split:
            skipped = 0
            out_image_dir = output_root / "images" / split
            out_image_dir.mkdir(parents=True, exist_ok=True)
            for image_path in image_paths:
                image = Image.open(image_path).convert("RGB")
                coords_by_label = {
                    label: read_coords(input_root, split, label, image_path.name)
                    for label in LABELS
                }
                for aug_idx in range(args.num_augments):
                    result = augment_one(image, coords_by_label, rng=rng, args=args)
                    if result is None:
                        skipped += 1
                        continue
                    patch, transformed = result
                    patch = apply_union_pixel_augment(
                        patch,
                        rng=rng,
                        pixel_aug_enabled=args.pixel_aug,
                        blur_noise_prob=args.blur_noise_prob,
                        color_aug_prob=args.color_aug_prob,
                    )
                    aug_name = f"{image_path.stem}_aug{aug_idx:03d}{args.image_ext.lower()}"
                    patch.save(out_image_dir / aug_name)
                    for label in LABELS:
                        write_coords(output_root, split, label, aug_name, transformed[label])
                    point_count += sum(len(points) for points in transformed.values())
                    image_count += 1
            if skipped:
                print(f"BCData {split}: skipped {skipped} crops because of --min-points")

        if image_count:
            summary[split] = (image_count, point_count)

    print("Augmented BCData original layout:")
    for split, (num_images, num_points) in summary.items():
        print(f"  {split}: {num_images} images, {num_points} points")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
