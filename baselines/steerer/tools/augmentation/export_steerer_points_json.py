#!/usr/bin/env python3
"""Export STEERER point text files to the shared predictions.json format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="STEERER *_gt_loc.txt or pred_points.txt")
    parser.add_argument("--output", required=True, help="Output predictions.json")
    parser.add_argument("--dataset", required=True, choices=("BCData", "CoNIC", "MoNuSeg"))
    parser.add_argument("--method", required=True, help="Method name, e.g. STEERER or ground_truth")
    parser.add_argument("--role", required=True, choices=("gt", "pred"))
    parser.add_argument("--extraction-method", required=True)
    parser.add_argument("--id-prefix", default="", help="Optional prefix added to every sample id")
    parser.add_argument("--thresholds", nargs="+", type=int, default=[6, 12, 24])
    return parser.parse_args()


def read_points(path: Path) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_no} has fewer than two columns")
            sample_id = parts[0]
            point_num = int(parts[1])
            values = [float(item) for item in parts[2:]]

            if point_num == 0:
                if values:
                    raise ValueError(f"{path}:{line_no} declares 0 points but has values")
                points = []
            elif len(values) == point_num * 2:
                points = [
                    [values[index], values[index + 1]]
                    for index in range(0, len(values), 2)
                ]
            elif len(values) == point_num * 5:
                points = [
                    [values[index], values[index + 1]]
                    for index in range(0, len(values), 5)
                ]
            else:
                raise ValueError(
                    f"{path}:{line_no} expected {point_num * 2} pred values "
                    f"or {point_num * 5} gt values, got {len(values)}")

            samples.append({"id": sample_id, "points": points})
    return samples


def main() -> None:
    args = parse_args()
    samples = read_points(Path(args.input))
    if args.id_prefix:
        for sample in samples:
            sample["id"] = args.id_prefix + sample["id"]

    output = {
        "metadata": {
            "dataset": args.dataset,
            "method": args.method,
            "role": args.role,
            "extraction_method": args.extraction_method,
            "coordinate_order": "xy",
            "coordinate_unit": "pixel",
            "matching_thresholds_px": args.thresholds,
            "source_file": str(Path(args.input)),
        },
        "samples": samples,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Samples: {len(samples)}")
    print(f"Total points: {sum(len(sample['points']) for sample in samples)}")


if __name__ == "__main__":
    main()
