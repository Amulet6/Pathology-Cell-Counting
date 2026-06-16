#!/usr/bin/env python3
"""Convert MoNuSeg XML annotations into the processed layout expected by STEERER.

Supported common layout:
  MoNuSeg/
    ...Training.../Tissue Images/*.tif
    ...Training.../Annotations/*.xml
    ...Testing.../Tissue Images/*.tif
    ...Testing.../Annotations/*.xml

The converter searches recursively for image files and pairs each image with an
XML annotation file that has the same stem. Each annotated nucleus polygon is
converted to one centroid point.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath

import numpy as np
from PIL import Image, ImageOps


TARGET_SPLITS = ("train", "val", "test")
SPLIT_ID_OFFSETS = {
    "train": 0,
    "val": 100000,
    "test": 200000,
}

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
TRAIN_HINTS = ("train", "training")
VAL_HINTS = ("val", "valid", "validation")
TEST_HINTS = ("test", "testing")
NON_IMAGE_HINTS = ("annotation", "annotations", "mask", "masks", "label", "labels", "groundtruth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to extracted MoNuSeg directory, MoNuSeg zip, or a directory containing MoNuSeg zips")
    parser.add_argument("--output", required=True, help="Output ProcessedData/MoNuSeg_numeric directory")
    parser.add_argument("--train-zip", default=None, help="Optional explicit MoNuSeg training zip")
    parser.add_argument("--test-zip", default=None, help="Optional explicit MoNuSeg test zip")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Used when no official split folders are found")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation fraction split from train/all samples")
    parser.add_argument("--seed", type=int, default=3035)
    parser.add_argument("--box-size", type=int, default=16, help="Pseudo box size for point annotations")
    parser.add_argument("--scale-category", type=int, default=0, help="STEERER scale category written to *_gt_loc.txt")
    parser.add_argument("--limit-per-split", type=int, default=0, help="Only convert first N samples per split for smoke tests")
    return parser.parse_args()


@dataclass(frozen=True)
class SourceFile:
    origin: Path
    member: str | None = None

    @property
    def name(self) -> str:
        if self.member is not None:
            return PurePosixPath(self.member).name
        return self.origin.name

    @property
    def stem(self) -> str:
        if self.member is not None:
            return PurePosixPath(self.member).stem
        return self.origin.stem

    @property
    def suffix(self) -> str:
        if self.member is not None:
            return PurePosixPath(self.member).suffix
        return self.origin.suffix

    @property
    def parts(self) -> tuple[str, ...]:
        if self.member is not None:
            return self.origin.parts + PurePosixPath(self.member).parts
        return self.origin.parts

    def read_bytes(self) -> bytes:
        if self.member is None:
            return self.origin.read_bytes()
        with zipfile.ZipFile(self.origin) as zip_file:
            return zip_file.read(self.member)

    def __str__(self) -> str:
        if self.member is None:
            return str(self.origin)
        return f"{self.origin}!{self.member}"


def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def natural_key(path) -> tuple:
    key = []
    for part in path.stem.replace("_", "-").split("-"):
        key.append((0, int(part)) if part.isdigit() else (1, part.lower()))
    return tuple(key)


def path_has_hint(path, hints: tuple[str, ...]) -> bool:
    text = " ".join(part.lower() for part in path.parts)
    return any(hint in text for hint in hints)


def is_candidate_image(path) -> bool:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    stem = path.stem.lower()
    if any(hint in stem for hint in NON_IMAGE_HINTS):
        return False
    return not path_has_hint(path, NON_IMAGE_HINTS)


def split_hint(path) -> str | None:
    if path_has_hint(path, TEST_HINTS):
        return "test"
    if path_has_hint(path, VAL_HINTS):
        return "val"
    if path_has_hint(path, TRAIN_HINTS):
        return "train"
    return None


def is_ignored_zip_member(name: str) -> bool:
    path = PurePosixPath(name)
    if name.endswith("/"):
        return True
    parts = path.parts
    if "__MACOSX" in parts:
        return True
    if path.name.startswith("._") or path.name == ".DS_Store":
        return True
    return False


def iter_zip_sources(zip_path: Path) -> list[SourceFile]:
    sources = []
    with zipfile.ZipFile(zip_path) as zip_file:
        for name in zip_file.namelist():
            if is_ignored_zip_member(name):
                continue
            sources.append(SourceFile(zip_path, name))
    return sources


def find_monuseg_zips(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".zip":
        return [root]
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.rglob("*.zip")
        if "monuseg" in path.name.lower() or "monu" in path.name.lower()
    )


def iter_directory_sources(root: Path) -> list[SourceFile]:
    sources = []
    for path in root.rglob("*"):
        if path.is_file() and not path.name.startswith("._") and path.name != ".DS_Store":
            sources.append(SourceFile(path))
    return sources


def collect_source_files(root: Path, train_zip: str | None, test_zip: str | None) -> list[SourceFile]:
    explicit_zips = [Path(path) for path in (train_zip, test_zip) if path]
    if explicit_zips:
        sources = []
        for zip_path in explicit_zips:
            sources.extend(iter_zip_sources(zip_path))
        return sources

    zip_paths = find_monuseg_zips(root)
    if zip_paths:
        sources = []
        for zip_path in zip_paths:
            sources.extend(iter_zip_sources(zip_path))
        return sources

    if root.is_file() and root.suffix.lower() == ".zip":
        return iter_zip_sources(root)
    if root.is_dir():
        return iter_directory_sources(root)
    raise FileNotFoundError(f"Input path does not exist: {root}")


def polygon_centroid(vertices: list[tuple[float, float]]) -> tuple[float, float]:
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape[0] < 3:
        return float(points[:, 0].mean()), float(points[:, 1].mean())

    x = points[:, 0]
    y = points[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-8:
        return float(x.mean()), float(y.mean())

    cx = ((x + x_next) * cross).sum() / (6.0 * area)
    cy = ((y + y_next) * cross).sum() / (6.0 * area)
    return float(cx), float(cy)


def parse_xml_points(xml_path: SourceFile) -> list[list[float]]:
    root = ET.fromstring(xml_path.read_bytes())
    points = []
    for region in root.iter():
        if strip_namespace(region.tag) != "region":
            continue

        vertices = []
        for node in region.iter():
            if strip_namespace(node.tag) != "vertex":
                continue
            x = node.attrib.get("X") or node.attrib.get("x")
            y = node.attrib.get("Y") or node.attrib.get("y")
            if x is None or y is None:
                continue
            vertices.append((float(x), float(y)))

        if vertices:
            cx, cy = polygon_centroid(vertices)
            points.append([cx, cy])
    return points


def clamp_point(point: list[float], width: int, height: int) -> list[int]:
    x = int(round(float(point[0])))
    y = int(round(float(point[1])))
    x = min(max(x, 0), width - 1)
    y = min(max(y, 0), height - 1)
    return [x, y]


def read_image(path: SourceFile) -> Image.Image:
    image = Image.open(io.BytesIO(path.read_bytes()))
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def collect_samples(root: Path, train_zip: str | None, test_zip: str | None) -> dict[str, list[tuple[SourceFile, SourceFile]]]:
    sources = collect_source_files(root, train_zip, test_zip)
    xml_by_stem = {}
    for xml_path in sources:
        if xml_path.suffix.lower() == ".xml":
            xml_by_stem.setdefault(xml_path.stem.lower(), []).append(xml_path)

    grouped = {"train": [], "val": [], "test": [], "unknown": []}
    for image_path in sorted(sources, key=natural_key):
        if not is_candidate_image(image_path):
            continue
        xml_candidates = xml_by_stem.get(image_path.stem.lower())
        if not xml_candidates:
            continue

        xml_path = sorted(xml_candidates, key=lambda p: len(str(p)))[0]
        split = split_hint(image_path) or split_hint(xml_path) or "unknown"
        grouped[split].append((image_path, xml_path))

    total = sum(len(items) for items in grouped.values())
    if total == 0:
        raise FileNotFoundError(
            "No MoNuSeg image/XML pairs found. Expected MoNuSeg training/test "
            "zips or extracted image/XML files with matching file stems.")
    return grouped


def split_unknown(samples: list[tuple[SourceFile, SourceFile]], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[tuple[SourceFile, SourceFile]]]:
    samples = samples[:]
    random.Random(seed).shuffle(samples)
    n = len(samples)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n < 3:
        return {"train": samples, "val": [], "test": []}

    train_count = max(1, int(round(n * train_ratio)))
    val_count = max(1, int(round(n * val_ratio)))
    if train_count + val_count >= n:
        train_count = max(1, n - val_count - 1)

    return {
        "train": sorted(samples[:train_count], key=lambda item: natural_key(item[0])),
        "val": sorted(samples[train_count:train_count + val_count], key=lambda item: natural_key(item[0])),
        "test": sorted(samples[train_count + val_count:], key=lambda item: natural_key(item[0])),
    }


def split_train_val(samples: list[tuple[SourceFile, SourceFile]], val_ratio: float, seed: int) -> tuple[list[tuple[SourceFile, SourceFile]], list[tuple[SourceFile, SourceFile]]]:
    samples = samples[:]
    random.Random(seed).shuffle(samples)
    if len(samples) < 2:
        return samples, []

    val_count = max(1, int(round(len(samples) * val_ratio)))
    val_count = min(val_count, len(samples) - 1)
    train = sorted(samples[:-val_count], key=lambda item: natural_key(item[0]))
    val = sorted(samples[-val_count:], key=lambda item: natural_key(item[0]))
    return train, val


def resolve_splits(grouped: dict[str, list[tuple[SourceFile, SourceFile]]], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[tuple[SourceFile, SourceFile]]]:
    known_count = len(grouped["train"]) + len(grouped["val"]) + len(grouped["test"])
    if known_count == 0:
        return split_unknown(grouped["unknown"], train_ratio, val_ratio, seed)

    splits = {
        "train": grouped["train"] + grouped["unknown"],
        "val": grouped["val"],
        "test": grouped["test"],
    }
    if not splits["val"]:
        splits["train"], splits["val"] = split_train_val(splits["train"], val_ratio, seed)
    if not splits["test"]:
        all_samples = splits["train"] + splits["val"]
        return split_unknown(all_samples, train_ratio, val_ratio, seed)

    return {name: sorted(items, key=lambda item: natural_key(item[0])) for name, items in splits.items()}


def convert_sample(
    image_path: SourceFile,
    xml_path: SourceFile,
    image_id: str,
    image_out: Path,
    json_out: Path,
    box_size: int,
    scale_category: int,
) -> tuple[str, int]:
    image = read_image(image_path)
    width, height = image.size
    points = [clamp_point(point, width, height) for point in parse_xml_points(xml_path)]

    output_image_name = f"{image_id}.png"
    image.save(image_out / output_image_name)

    boxes = [
        [point[0] - box_size // 2, point[1] - box_size // 2, point[0] + box_size // 2, point[1] + box_size // 2]
        for point in points
    ]
    record = {
        "img_id": output_image_name,
        "human_num": len(points),
        "points": points,
        "boxes": boxes,
        "labels": [0 for _ in points],
        "source_image": str(image_path),
        "source_annotation": str(xml_path),
    }
    (json_out / f"{image_id}.json").write_text(json.dumps(record), encoding="utf-8")

    loc_items = [image_id, str(len(points))]
    for x, y in points:
        loc_items.extend([str(x), str(y), str(box_size), str(box_size), str(scale_category)])
    return " ".join(loc_items), len(points)


def convert_split(
    split: str,
    samples: list[tuple[SourceFile, SourceFile]],
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
        samples = samples[:limit_per_split]

    ids = []
    gt_lines = []
    total_points = 0
    offset = SPLIT_ID_OFFSETS[split]

    for output_index, (image_path, xml_path) in enumerate(samples):
        image_id = f"{offset + output_index:06d}"
        gt_line, point_count = convert_sample(
            image_path,
            xml_path,
            image_id,
            image_out,
            json_out,
            box_size,
            scale_category,
        )
        ids.append(image_id)
        gt_lines.append(gt_line)
        total_points += point_count

    (output_dir / f"{split}.txt").write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
    (output_dir / f"{split}_gt_loc.txt").write_text("\n".join(gt_lines) + ("\n" if gt_lines else ""), encoding="utf-8")
    return len(ids), total_points


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = collect_samples(input_dir, args.train_zip, args.test_zip)
    splits = resolve_splits(grouped, args.train_ratio, args.val_ratio, args.seed)

    summary = {}
    for split in TARGET_SPLITS:
        summary[split] = convert_split(
            split,
            splits.get(split, []),
            output_dir,
            args.box_size,
            args.scale_category,
            args.limit_per_split,
        )

    print("Converted MoNuSeg:")
    for split in TARGET_SPLITS:
        image_count, point_count = summary[split]
        print(f"  {split}: {image_count} images, {point_count} points")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
