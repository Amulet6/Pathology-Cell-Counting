#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ROOT_DIR / "repos" / "CellVTA"
SHARE_DIR = ROOT_DIR / "share"

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(SHARE_DIR) not in sys.path:
    sys.path.insert(0, str(SHARE_DIR))

from label_to_centroids import instance_centroids  # type: ignore
from cell_segmentation.inference.inference_cellvit_experiment_pannuke import (  # noqa: E402
    InferenceCellViT,
)
from cell_segmentation.inference.inference_cellvit_upscale import (  # noqa: E402
    InferenceCellViTUpscale,
)
from cell_segmentation.utils.tools import pair_coordinates  # noqa: E402


def sample_id_from_name(name: str) -> str:
    return Path(name).stem


def centroids_from_instance_dict(instance_dict: dict[str, Any]) -> list[list[float]]:
    points: list[list[float]] = []
    for _, spec in sorted(instance_dict.items(), key=lambda item: int(item[0])):
        centroid = spec.get("centroid")
        if centroid is None:
            continue
        points.append([float(centroid[0]), float(centroid[1])])
    return points


def centroids_from_cell_frame(cell_frame: Any) -> list[list[float]]:
    if cell_frame is None:
        return []
    if hasattr(cell_frame, "empty") and cell_frame.empty:
        return []
    points: list[list[float]] = []
    for centroid in cell_frame["centroid"].tolist():
        centroid = np.asarray(centroid, dtype=float).reshape(-1)
        points.append([float(centroid[0]), float(centroid[1])])
    return points


def centroids_from_cell_frame_scaled(cell_frame: Any, scale: float) -> list[list[float]]:
    """Convert CellVTA global centroids back to local patch coordinates."""
    if cell_frame is None:
        return []
    if hasattr(cell_frame, "empty") and cell_frame.empty:
        return []
    points: list[list[float]] = []
    for centroid in cell_frame["centroid"].tolist():
        centroid = np.asarray(centroid, dtype=float).reshape(-1)
        points.append([float(centroid[0] * scale), float(centroid[1] * scale)])
    return points


def build_predictions_json(
    dataset: str,
    method: str,
    role: str,
    extraction_method: str,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "metadata": {
            "dataset": dataset,
            "method": method,
            "role": role,
            "extraction_method": extraction_method,
            "coordinate_order": "xy",
            "coordinate_unit": "pixel",
            "matching_thresholds_px": [6, 12, 24],
        },
        "samples": samples,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def make_monuseg_infer_config(run_dir: Path, output_dir: Path) -> Path:
    """Build a temporary inference config from a training run directory."""
    with open(run_dir / "config.yaml", "r", encoding="utf-8") as f:
        run_conf = yaml.safe_load(f)

    run_conf = dict(run_conf)
    run_conf.setdefault("model", {})
    run_conf["model"]["path"] = str(run_dir / "checkpoints" / "model_best.pth")
    run_conf.setdefault("data", {})
    run_conf["data"].setdefault("overlap", 32)
    run_conf["data"].setdefault("with_padding", False)

    tmp_dir = Path(tempfile.mkdtemp(prefix="cellvta_monuseg_infer_", dir=str(output_dir)))
    infer_config_path = tmp_dir / "config.yaml"
    infer_config_path.write_text(
        yaml.safe_dump(run_conf, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return infer_config_path


@contextmanager
def pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def extract_gt_and_pred_conic(run_dir: Path, gpu: int) -> tuple[dict[str, Any], dict[str, Any]]:
    with pushd(REPO_DIR):
        inference = InferenceCellViT(
            run_dir=run_dir,
            gpu=gpu,
            checkpoint_name="model_best.pth",
            magnification=40,
        )
        model, dataloader, _ = inference.setup_patch_inference()
        model.to(inference.device)
        model.eval()

        gt_samples = []
        pred_samples = []

        with torch.no_grad():
            for batch in dataloader:
                imgs = batch[0].to(inference.device)
                masks = batch[1]
                tissue_types = list(batch[2])
                image_names = list(batch[3])

                if inference.mixed_precision:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        predictions_raw = model.forward(imgs)
                else:
                    predictions_raw = model.forward(imgs)

                predictions = inference.unpack_predictions(predictions_raw, model)
                gt = inference.unpack_masks(masks=masks, tissue_types=tissue_types, model=model)

                for i, image_name in enumerate(image_names):
                    sample_id = sample_id_from_name(image_name)
                    gt_samples.append(
                        {
                            "id": sample_id,
                            "points": centroids_from_instance_dict(gt.instance_types[i]),
                        }
                    )
                    pred_samples.append(
                        {
                            "id": sample_id,
                            "points": centroids_from_instance_dict(predictions.instance_types[i]),
                        }
                    )

    gt_json = build_predictions_json(
        dataset="CoNIC",
        method="ground_truth",
        role="gt",
        extraction_method="pixel_mean_of_instance_mask",
        samples=gt_samples,
    )
    pred_json = build_predictions_json(
        dataset="CoNIC",
        method="CellVTA",
        role="pred",
        extraction_method="model_instance_centroids",
        samples=pred_samples,
    )
    return gt_json, pred_json


def extract_gt_and_pred_monuseg(run_dir: Path, gpu: int, outdir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with pushd(REPO_DIR):
        infer_config = make_monuseg_infer_config(run_dir, outdir)
        inference = InferenceCellViTUpscale(
            config_path=str(infer_config),
            outdir=str(outdir),
            gpu=gpu,
            magnification=40,
        )
        model = inference.model
        model.to(inference.device)
        model.eval()

        gt_samples = []
        pred_samples = []

        with torch.no_grad():
            for batch in inference.inference_dataloader:
                imgs = batch[0]
                masks = batch[1]
                tissue_types = list(batch[2])
                image_names = list(batch[3])

                flattened_upscaled_imgs, upscaled_size, position_order = inference.upscale_imgs(
                    imgs,
                    grid=(2, 2),
                    patch_size=inference.run_conf["data"]["input_shape"],
                    overlap=inference.run_conf["data"]["overlap"],
                    with_padding=inference.run_conf["data"]["with_padding"],
                )
                flattened_upscaled_imgs = flattened_upscaled_imgs.to(inference.device)

                if inference.mixed_precision:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        predictions_flattened = model.forward(flattened_upscaled_imgs)
                else:
                    predictions_flattened = model.forward(flattened_upscaled_imgs)

                cell_dict_list = inference.extract_unflattened_cell_dict(
                    predictions_flattened,
                    position_order=position_order,
                    overlap=inference.run_conf["data"]["overlap"],
                    with_padding=inference.run_conf["data"]["with_padding"],
                    generate_plots=False,
                )
                gt = inference.unpack_masks(masks=masks, tissue_types=tissue_types)
                local_scale = float(inference.run_conf["data"]["input_shape"]) / float(
                    upscaled_size
                )

                for i, image_name in enumerate(image_names):
                    sample_id = sample_id_from_name(image_name)
                    gt_samples.append(
                        {
                            "id": sample_id,
                            "points": centroids_from_instance_dict(gt.instance_types[i]),
                        }
                    )
                    pred_samples.append(
                        {
                            "id": sample_id,
                            "points": centroids_from_cell_frame_scaled(
                                cell_dict_list[i], local_scale
                            ),
                        }
                    )

    gt_json = build_predictions_json(
        dataset="MoNuSeg",
        method="ground_truth",
        role="gt",
        extraction_method="pixel_mean_of_instance_mask",
        samples=gt_samples,
    )
    pred_json = build_predictions_json(
        dataset="MoNuSeg",
        method="CellVTA",
        role="pred",
        extraction_method="model_instance_centroids",
        samples=pred_samples,
    )
    return gt_json, pred_json


def pair_stats(gt_points: np.ndarray, pred_points: np.ndarray, threshold: float) -> tuple[int, int, int]:
    if gt_points.size == 0 and pred_points.size == 0:
        return 0, 0, 0
    if gt_points.size == 0:
        return 0, int(pred_points.shape[0]), 0
    if pred_points.size == 0:
        return 0, 0, int(gt_points.shape[0])
    paired, unpaired_true, unpaired_pred = pair_coordinates(gt_points, pred_points, threshold)
    return int(paired.shape[0]), int(unpaired_pred.shape[0]), int(unpaired_true.shape[0])


def evaluate_predictions(gt_json: dict[str, Any], pred_json: dict[str, Any], thresholds: Iterable[int]) -> dict[str, Any]:
    gt_map = {sample["id"]: sample.get("points", []) for sample in gt_json["samples"]}
    pred_map = {sample["id"]: sample.get("points", []) for sample in pred_json["samples"]}
    all_ids = sorted(set(gt_map) | set(pred_map))

    abs_errors: list[float] = []
    sq_errors: list[float] = []
    total_gt = 0
    total_pred = 0

    threshold_totals = {
        int(threshold): {"tp": 0, "fp": 0, "fn": 0} for threshold in thresholds
    }

    for sample_id in all_ids:
        gt_points = np.asarray(gt_map.get(sample_id, []), dtype=float).reshape(-1, 2)
        pred_points = np.asarray(pred_map.get(sample_id, []), dtype=float).reshape(-1, 2)
        gt_count = int(gt_points.shape[0])
        pred_count = int(pred_points.shape[0])
        err = pred_count - gt_count
        abs_errors.append(abs(err))
        sq_errors.append(err * err)
        total_gt += gt_count
        total_pred += pred_count

        for threshold in threshold_totals:
            tp, fp, fn = pair_stats(gt_points, pred_points, threshold)
            threshold_totals[threshold]["tp"] += tp
            threshold_totals[threshold]["fp"] += fp
            threshold_totals[threshold]["fn"] += fn

    mae = float(np.mean(abs_errors)) if abs_errors else 0.0
    mse = float(np.mean(sq_errors)) if sq_errors else 0.0
    rmse = float(math.sqrt(mse))
    total_err_pct = float(100.0 * sum(abs_errors) / max(total_gt, 1))

    localization: dict[str, Any] = {}
    for threshold, stats in threshold_totals.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == fp == fn == 0 else 0.0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if tp == fp == fn == 0 else 0.0)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        localization[f"{threshold}px"] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

    return {
        "counting": {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "total_gt": int(total_gt),
            "total_pred": int(total_pred),
            "total_err_pct": total_err_pct,
        },
        "localization": localization,
    }


def run_one(dataset_name: str, run_dir: Path, gpu: int, output_root: Path) -> dict[str, Any]:
    outdir = output_root / dataset_name
    outdir.mkdir(parents=True, exist_ok=True)

    if dataset_name == "conic":
        gt_json, pred_json = extract_gt_and_pred_conic(run_dir, gpu)
    elif dataset_name == "monuseg":
        gt_json, pred_json = extract_gt_and_pred_monuseg(run_dir, gpu, outdir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    gt_path = outdir / "gt_predictions.json"
    pred_path = outdir / "pred_predictions.json"
    eval_path = outdir / "centroid_eval.json"

    write_json(gt_path, gt_json)
    write_json(pred_path, pred_json)
    metrics = evaluate_predictions(gt_json, pred_json, [6, 12, 24])
    write_json(eval_path, metrics)

    return {
        "dataset": dataset_name,
        "gt": str(gt_path),
        "pred": str(pred_path),
        "eval": str(eval_path),
        "metrics": metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CellVTA centroid evaluation")
    parser.add_argument("--conic-run", type=Path, default=None, help="CoNIC run directory")
    parser.add_argument("--monuseg-run", type=Path, default=None, help="MoNuSeg run directory")
    parser.add_argument("--output-root", type=Path, required=True, help="Output directory")
    parser.add_argument("--gpu", type=int, default=1, help="GPU id")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    summary: dict[str, Any] = {}
    if args.conic_run is not None:
        summary["conic"] = run_one("conic", args.conic_run, args.gpu, args.output_root)
    if args.monuseg_run is not None:
        summary["monuseg"] = run_one("monuseg", args.monuseg_run, args.gpu, args.output_root)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
