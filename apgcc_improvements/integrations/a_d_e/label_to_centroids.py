#!/usr/bin/env python3
"""
Convert pathology cell/nucleus labels to center-point coordinates.

Supported formats:
  - CoNIC: labels.npy, using labels[..., 0] as the instance-id map.
  - MoNuSeg: XML polygon annotations, one polygon/Region per nucleus.

Output coordinate order is [x, y] in pixel units.
This script is model-agnostic and does not depend on any training code.
"""

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def polygon_area_centroid(vertices):
    """Return area-weighted polygon centroid as (x, y)."""
    if len(vertices) == 0:
        return None
    if len(vertices) < 3:
        xs, ys = zip(*vertices)
        return float(np.mean(xs)), float(np.mean(ys))

    area2 = 0.0
    cx_num = 0.0
    cy_num = 0.0
    n = len(vertices)
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        cross = x0 * y1 - x1 * y0
        area2 += cross
        cx_num += (x0 + x1) * cross
        cy_num += (y0 + y1) * cross

    if abs(area2) < 1e-8:
        xs, ys = zip(*vertices)
        return float(np.mean(xs)), float(np.mean(ys))

    cx = cx_num / (3.0 * area2)
    cy = cy_num / (3.0 * area2)
    return float(cx), float(cy)


def vertex_mean_centroid(vertices):
    """Return arithmetic mean of polygon vertices as (x, y)."""
    if len(vertices) == 0:
        return None
    xs, ys = zip(*vertices)
    return float(np.mean(xs)), float(np.mean(ys))


def read_monuseg_xml(xml_path, centroid_method):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    points = []

    for region in root.findall(".//Region"):
        vertices = []
        for vertex in region.findall(".//Vertex"):
            x = vertex.get("X")
            y = vertex.get("Y")
            if x is None or y is None:
                continue
            vertices.append((float(x), float(y)))

        if centroid_method == "polygon_centroid":
            point = polygon_area_centroid(vertices)
        elif centroid_method == "vertex_mean":
            point = vertex_mean_centroid(vertices)
        else:
            raise ValueError(f"Unknown centroid method: {centroid_method}")

        if point is not None:
            points.append([point[0], point[1]])

    return points


def convert_monuseg(args):
    xml_root = Path(args.xml_root)
    xml_paths = sorted(xml_root.rglob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No XML files found under {xml_root}")

    samples = []
    for xml_path in xml_paths:
        points = read_monuseg_xml(xml_path, args.centroid_method)
        samples.append({
            "id": xml_path.stem,
            "label_path": str(xml_path),
            "points": points,
            "count": len(points)
        })

    metadata = {
        "dataset": "MoNuSeg",
        "source_format": "XML polygon annotations",
        "coordinate_order": "[x, y]",
        "coordinate_unit": "pixel",
        "centroid_method": args.centroid_method,
        "num_samples": len(samples),
        "total_points": int(sum(sample["count"] for sample in samples))
    }
    write_outputs(args.output, metadata, samples)


def instance_centroids(instance_map):
    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids > 0]

    points = []
    for instance_id in instance_ids:
        ys, xs = np.where(instance_map == instance_id)
        if len(xs) == 0:
            continue
        points.append([float(xs.mean()), float(ys.mean())])
    return points


def convert_conic(args):
    labels_path = Path(args.labels_npy)
    labels = np.load(labels_path, mmap_mode="r")
    if labels.ndim == 4:
        instance_maps = labels[..., 0]
    elif labels.ndim == 3:
        instance_maps = labels
    else:
        raise ValueError(
            "CoNIC labels must have shape [N,H,W] or [N,H,W,C], "
            f"but got {labels.shape}"
        )

    samples = []
    for index in range(instance_maps.shape[0]):
        points = instance_centroids(instance_maps[index])
        samples.append({
            "id": f"conic_{index:05d}",
            "label_index": index,
            "points": points,
            "count": len(points)
        })

    metadata = {
        "dataset": "CoNIC",
        "source_format": "NumPy instance-id masks",
        "label_path": str(labels_path),
        "instance_channel": 0 if labels.ndim == 4 else None,
        "coordinate_order": "[x, y]",
        "coordinate_unit": "pixel",
        "centroid_method": "pixel_mean_of_instance_mask",
        "num_samples": len(samples),
        "total_points": int(sum(sample["count"] for sample in samples))
    }
    write_outputs(args.output, metadata, samples)


def write_outputs(output_path, metadata, samples):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "metadata": metadata,
        "samples": samples
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "point_index", "x", "y"])
        for sample in samples:
            for point_index, (x, y) in enumerate(sample["points"]):
                writer.writerow([
                    sample["id"],
                    point_index,
                    format_float(x),
                    format_float(y)
                ])

    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Wrote JSON: {output_path}")
    print(f"Wrote CSV:  {csv_path}")
    print(f"Wrote summary: {summary_path}")


def format_float(value):
    if not math.isfinite(value):
        return str(value)
    return f"{value:.6f}".rstrip("0").rstrip(".")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert label annotations to cell/nucleus centroid coordinates."
    )
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    monuseg = subparsers.add_parser(
        "monuseg",
        help="Convert MoNuSeg XML polygon annotations to centroids."
    )
    monuseg.add_argument(
        "--xml_root",
        required=True,
        help="Directory containing MoNuSeg XML annotation files."
    )
    monuseg.add_argument(
        "--output",
        required=True,
        help="Output JSON path. A CSV and summary JSON will also be written."
    )
    monuseg.add_argument(
        "--centroid_method",
        choices=["polygon_centroid", "vertex_mean"],
        default="polygon_centroid",
        help=(
            "polygon_centroid uses the area-weighted polygon centroid; "
            "vertex_mean uses the arithmetic mean of annotation vertices."
        )
    )
    monuseg.set_defaults(func=convert_monuseg)

    conic = subparsers.add_parser(
        "conic",
        help="Convert CoNIC instance-id masks from labels.npy to centroids."
    )
    conic.add_argument(
        "--labels_npy",
        required=True,
        help="Path to CoNIC labels.npy."
    )
    conic.add_argument(
        "--output",
        required=True,
        help="Output JSON path. A CSV and summary JSON will also be written."
    )
    conic.set_defaults(func=convert_conic)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
