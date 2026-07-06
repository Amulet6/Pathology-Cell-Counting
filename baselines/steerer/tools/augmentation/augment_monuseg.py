#!/usr/bin/env python3
"""Augment MoNuSeg in its original image/XML-polygon layout."""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from original_augment_common import (
    IMAGE_SUFFIXES,
    apply_union_pixel_augment,
    polygon_area,
    sample_image_transform,
    strip_namespace,
    transform_polygon,
)


SPLITS = ("train", "val", "test")
NON_IMAGE_HINTS = ("annotation", "annotations", "mask", "masks", "label", "labels", "groundtruth")


@dataclass(frozen=True)
class Sample:
    image_path: Path
    xml_path: Path
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Augment original MoNuSeg. The script searches recursively for image/XML pairs "
            "and writes Training/Validation/Testing folders with Tissue Images and Annotations."
        )
    )
    parser.add_argument("--input", required=True, help="Path to extracted original MoNuSeg directory")
    parser.add_argument("--output", required=True, help="Output original-layout MoNuSeg directory")
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
    parser.add_argument(
        "--resize-original-to-patch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "When --include-original is enabled for the augmented split, resize the "
            "original image to --patch-size and scale XML vertices accordingly instead "
            "of copying it unchanged."
        ),
    )
    parser.add_argument("--copy-eval-splits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-regions", type=int, default=0, help="Retry/skip crops with fewer polygons")
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--pad-value", type=int, default=255)
    parser.add_argument("--image-ext", default=".png", help="Extension used for augmented images")
    return parser.parse_args()


def split_from_path(path: Path) -> str:
    parts = [part.lower() for part in path.parts]
    if any(part in ("test", "testing") or "testing" in part for part in parts):
        return "test"
    if any(part in ("val", "valid", "validation") or "validation" in part for part in parts):
        return "val"
    if any(part in ("train", "training") or "training" in part for part in parts):
        return "train"
    return "train"


def discover_samples(root: Path) -> list[Sample]:
    xml_by_stem = {path.stem: path for path in root.rglob("*.xml")}
    samples: list[Sample] = []
    for image_path in root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        path_text = str(image_path.parent).lower()
        if any(hint in path_text for hint in NON_IMAGE_HINTS):
            continue
        xml_path = xml_by_stem.get(image_path.stem)
        if xml_path is None:
            continue
        samples.append(Sample(image_path=image_path, xml_path=xml_path, split=split_from_path(image_path)))
    return sorted(samples, key=lambda sample: (sample.split, sample.image_path.stem.lower()))


def split_dirs(output_root: Path, split: str) -> tuple[Path, Path]:
    folder = {"train": "Training", "val": "Validation", "test": "Testing"}[split]
    image_dir = output_root / folder / "Tissue Images"
    xml_dir = output_root / folder / "Annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, xml_dir


def get_vertices(region: ET.Element) -> tuple[ET.Element | None, list[ET.Element], list[tuple[float, float]]]:
    container = None
    for child in list(region):
        if strip_namespace(child.tag) == "vertices":
            container = child
            break
    search_root = container if container is not None else region
    vertex_nodes = [node for node in search_root.iter() if strip_namespace(node.tag) == "vertex"]
    vertices: list[tuple[float, float]] = []
    for node in vertex_nodes:
        x = node.attrib.get("X") or node.attrib.get("x")
        y = node.attrib.get("Y") or node.attrib.get("y")
        if x is not None and y is not None:
            vertices.append((float(x), float(y)))
    return container, vertex_nodes, vertices


def replace_vertices(container: ET.Element, vertex_nodes: list[ET.Element], vertices: list[tuple[float, float]]) -> None:
    vertex_tag = vertex_nodes[0].tag if vertex_nodes else "Vertex"
    for child in list(container):
        if strip_namespace(child.tag) == "vertex":
            container.remove(child)
    for x, y in vertices:
        ET.SubElement(container, vertex_tag, {"X": f"{x:.3f}", "Y": f"{y:.3f}", "Z": "0"})


def transform_xml(xml_path: Path, output_xml: Path, transform) -> int:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    parents = {child: parent for parent in root.iter() for child in parent}
    kept_regions = 0

    for region in list(root.iter()):
        if strip_namespace(region.tag) != "region":
            continue
        container, vertex_nodes, vertices = get_vertices(region)
        if len(vertices) < 3:
            parent = parents.get(region)
            if parent is not None:
                parent.remove(region)
            continue

        new_vertices = transform_polygon(vertices, transform)
        if len(new_vertices) < 3 or polygon_area(new_vertices) <= 1.0:
            parent = parents.get(region)
            if parent is not None:
                parent.remove(region)
            continue

        if container is None:
            container = ET.SubElement(region, "Vertices")
        replace_vertices(container, vertex_nodes, new_vertices)
        kept_regions += 1

    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    return kept_regions


def copy_original_sample(output_root: Path, sample: Sample) -> int:
    image_dir, xml_dir = split_dirs(output_root, sample.split)
    shutil.copy2(sample.image_path, image_dir / sample.image_path.name)
    shutil.copy2(sample.xml_path, xml_dir / sample.xml_path.name)
    tree = ET.parse(sample.xml_path)
    return sum(1 for node in tree.getroot().iter() if strip_namespace(node.tag) == "region")


def resize_original_sample(output_root: Path, sample: Sample, patch_size: int, image_ext: str) -> int:
    image_dir, xml_dir = split_dirs(output_root, sample.split)
    image = Image.open(sample.image_path).convert("RGB")
    scale_x = patch_size / image.width
    scale_y = patch_size / image.height
    image = image.resize((patch_size, patch_size), Image.BILINEAR)

    stem = f"{sample.image_path.stem}_orig"
    image.save(image_dir / f"{stem}{image_ext.lower()}")

    tree = ET.parse(sample.xml_path)
    root = tree.getroot()
    region_count = 0
    for region in root.iter():
        if strip_namespace(region.tag) != "region":
            continue
        region_count += 1
        for node in region.iter():
            if strip_namespace(node.tag) != "vertex":
                continue
            x = node.attrib.get("X") or node.attrib.get("x")
            y = node.attrib.get("Y") or node.attrib.get("y")
            if x is None or y is None:
                continue
            new_x = min(max(float(x) * scale_x, 0.0), patch_size - 1)
            new_y = min(max(float(y) * scale_y, 0.0), patch_size - 1)
            if "X" in node.attrib:
                node.attrib["X"] = f"{new_x:.3f}"
            else:
                node.attrib["x"] = f"{new_x:.3f}"
            if "Y" in node.attrib:
                node.attrib["Y"] = f"{new_y:.3f}"
            else:
                node.attrib["y"] = f"{new_y:.3f}"

    tree.write(xml_dir / f"{stem}.xml", encoding="utf-8", xml_declaration=True)
    return region_count


def augment_one(sample: Sample, output_root: Path, args: argparse.Namespace, rng: random.Random, aug_idx: int) -> int | None:
    image = Image.open(sample.image_path).convert("RGB")
    image_dir, xml_dir = split_dirs(output_root, sample.split)

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
        stem = f"{sample.image_path.stem}_aug{aug_idx:03d}"
        image_name = f"{stem}{args.image_ext.lower()}"
        xml_name = f"{stem}.xml"
        kept_regions = transform_xml(sample.xml_path, xml_dir / xml_name, transform)
        if kept_regions >= args.min_regions:
            patch = apply_union_pixel_augment(
                patch,
                rng=rng,
                pixel_aug_enabled=args.pixel_aug,
                blur_noise_prob=args.blur_noise_prob,
                color_aug_prob=args.color_aug_prob,
            )
            patch.save(image_dir / image_name)
            return kept_regions
        (xml_dir / xml_name).unlink(missing_ok=True)
    return None


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    output_root = Path(args.output)
    if input_root.resolve() == output_root.resolve():
        raise ValueError("--output must be different from --input")
    if args.image_ext and not args.image_ext.startswith("."):
        args.image_ext = "." + args.image_ext

    samples = discover_samples(input_root)
    rng = random.Random(args.seed)
    summary: dict[str, tuple[int, int]] = {}

    for split in SPLITS:
        split_samples = [sample for sample in samples if sample.split == split]
        if not split_samples:
            continue

        image_count = 0
        region_count = 0
        should_copy_split = split != args.split and args.copy_eval_splits
        should_augment_split = split == args.split

        if should_copy_split:
            for sample in split_samples:
                region_count += copy_original_sample(output_root, sample)
                image_count += 1

        if should_augment_split and args.include_original:
            for sample in split_samples:
                if args.resize_original_to_patch:
                    region_count += resize_original_sample(
                        output_root,
                        sample,
                        args.patch_size,
                        args.image_ext,
                    )
                else:
                    region_count += copy_original_sample(output_root, sample)
                image_count += 1

        if should_augment_split:
            skipped = 0
            for sample in split_samples:
                for aug_idx in range(args.num_augments):
                    kept = augment_one(sample, output_root, args, rng, aug_idx)
                    if kept is None:
                        skipped += 1
                        continue
                    region_count += kept
                    image_count += 1
            if skipped:
                print(f"MoNuSeg {split}: skipped {skipped} crops because of --min-regions")

        if image_count:
            summary[split] = (image_count, region_count)

    print("Augmented MoNuSeg original layout:")
    for split, (num_images, num_regions) in summary.items():
        print(f"  {split}: {num_images} images, {num_regions} XML regions")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()
