from __future__ import annotations

import json
import zipfile
from pathlib import Path

import torch
from natsort import natsorted
from torch.utils.data import random_split


SEED = 19
VAL_RATIO = 0.2


def list_zip_image_stems(zip_path: Path, prefix: str) -> list[str]:
    with zipfile.ZipFile(zip_path) as zf:
        names = [
            Path(name).stem
            for name in zf.namelist()
            if name.startswith(prefix) and name.endswith(".png")
        ]
    return natsorted(names)


def parse_overlap_stem(stem: str) -> dict:
    return {
        "stem": stem,
        "image_relpath": f"conic_cellvit_patient_x40_linear_withOverlap/fold0/images/{stem}.png",
        "label_relpath": f"conic_cellvit_patient_x40_linear_withOverlap/fold0/labels/{stem}.npy",
    }


def parse_base_stem(stem: str) -> dict:
    return {
        "stem": stem,
        "image_relpath": f"conic_cellvit_patient/fold1/images/{stem}.png",
        "label_relpath": f"conic_cellvit_patient/fold1/labels/{stem}.npy",
    }


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    docs_dir = root / "docs"

    overlap_zip = data_dir / "conic_cellvit_patient_x40_linear_withOverlap.zip"
    base_zip = data_dir / "conic_cellvit_patient.zip"
    output_json = docs_dir / "conic_split_seed19.json"

    overlap_stems = list_zip_image_stems(
        overlap_zip,
        "conic_cellvit_patient_x40_linear_withOverlap/fold0/images/",
    )
    base_test_stems = list_zip_image_stems(
        base_zip,
        "conic_cellvit_patient/fold1/images/",
    )

    generator = torch.Generator().manual_seed(SEED)
    full_dataset = list(range(len(overlap_stems)))
    train_subset, val_subset = random_split(
        full_dataset,
        lengths=[1 - VAL_RATIO, VAL_RATIO],
        generator=generator,
    )

    train_items = [parse_overlap_stem(overlap_stems[idx]) for idx in train_subset.indices]
    val_items = [parse_overlap_stem(overlap_stems[idx]) for idx in val_subset.indices]
    test_items = [parse_base_stem(stem) for stem in base_test_stems]

    result = {
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "counts": {
            "train": len(train_items),
            "val": len(val_items),
            "test": len(test_items),
        },
        "splits": {
            "train": train_items,
            "val": val_items,
            "test": test_items,
        },
    }

    assert result["counts"] == {"train": 12768, "val": 3192, "test": 991}

    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_json)


if __name__ == "__main__":
    main()
