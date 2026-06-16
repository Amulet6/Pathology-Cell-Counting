#!/usr/bin/env python3
"""Create a strict MoNuSeg train/val/test directory from monuseg_split.json."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
SPLIT_DIRS = {
    "train": "Training",
    "val": "Validation",
    "test": "Testing",
}
NON_IMAGE_HINTS = ("annotation", "annotations", "mask", "masks", "label", "labels", "groundtruth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Extracted original MoNuSeg root")
    parser.add_argument("--split-json", required=True, help="monuseg_split.json")
    parser.add_argument("--output", required=True, help="Output original-layout split root")
    return parser.parse_args()


def is_candidate_image(path: Path) -> bool:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    text = str(path.parent).lower()
    return not any(hint in text for hint in NON_IMAGE_HINTS)


def item_id(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("id", "stem", "name", "image_id"):
            if key in item:
                return str(item[key])
    raise ValueError(f"Cannot resolve split item id from {item!r}")


def build_index(root: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    images: dict[str, list[Path]] = {}
    xmls: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.name.startswith("._") or path.name == ".DS_Store":
            continue
        if is_candidate_image(path):
            images.setdefault(path.stem, []).append(path)
        elif path.suffix.lower() == ".xml":
            xmls.setdefault(path.stem, []).append(path)
    return images, xmls


def choose(paths: list[Path], sample_id: str, role: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"No {role} file found for {sample_id}")
    return sorted(paths, key=lambda path: (len(str(path)), str(path)))[0]


def split_output_dirs(output_root: Path, split: str) -> tuple[Path, Path]:
    folder = SPLIT_DIRS[split]
    image_dir = output_root / folder / "Tissue Images"
    xml_dir = output_root / folder / "Annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, xml_dir


def main() -> None:
    args = parse_args()
    input_root = Path(args.input)
    split_data = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    images, xmls = build_index(input_root)
    summary = {}
    missing = []
    for split in ("train", "val", "test"):
        image_dir, xml_dir = split_output_dirs(output_root, split)
        ids = [item_id(item) for item in split_data[split]]
        copied = []
        for sid in ids:
            try:
                image_path = choose(images.get(sid, []), sid, "image")
                xml_path = choose(xmls.get(sid, []), sid, "xml")
            except FileNotFoundError as exc:
                missing.append(str(exc))
                continue
            shutil.copy2(image_path, image_dir / image_path.name)
            shutil.copy2(xml_path, xml_dir / xml_path.name)
            copied.append({
                "id": sid,
                "image": str(image_path),
                "xml": str(xml_path),
            })
        summary[split] = {
            "requested": len(ids),
            "copied": len(copied),
            "items": copied,
        }

    report = {
        "source_root": str(input_root),
        "split_json": str(Path(args.split_json)),
        "summary": summary,
        "missing": missing,
    }
    (output_root / "split_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_root}")
    for split in ("train", "val", "test"):
        print(f"{split}: {summary[split]['copied']} / {summary[split]['requested']}")
    if missing:
        print("Missing files:")
        for item in missing[:20]:
            print("  " + item)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
