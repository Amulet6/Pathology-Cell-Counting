#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import albumentations as A
import numpy as np
import torch
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ROOT_DIR / "repos" / "CellVTA"

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from models.segmentation.cell_segmentation.cellvit import (  # noqa: E402
    CellViT,
    CellViTUNIAdapter,
)


def load_run_conf(run_dir: Path) -> dict[str, Any]:
    with open(run_dir / "config.yaml", "r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f))


def build_normalizer(run_conf: dict[str, Any]) -> A.Compose:
    transform_settings = run_conf.get("transformations", {})
    normalize_cfg = transform_settings.get("normalize", {})
    mean = normalize_cfg.get("mean", (0.5, 0.5, 0.5))
    std = normalize_cfg.get("std", (0.5, 0.5, 0.5))
    return A.Compose([A.Normalize(mean=mean, std=std)])


def instantiate_model(run_conf: dict[str, Any], checkpoint: dict[str, Any]) -> torch.nn.Module:
    arch = checkpoint["arch"]
    if arch == "CellViTUNIAdapter":
        model = CellViTUNIAdapter(
            num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=run_conf["data"]["num_tissue_classes"],
            drop_rate=0,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=8,
            drop_path_rate=0.4,
            interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
            with_cffn=True,
            cffn_ratio=0.25,
            deform_ratio=0.5,
            add_vit_feature=True,
        )
    elif arch == "CellViT":
        model = CellViT(
            num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=run_conf["data"]["num_tissue_classes"],
            embed_dim=run_conf["model"]["embed_dim"],
            input_channels=run_conf["model"].get("input_channels", 3),
            depth=run_conf["model"]["depth"],
            num_heads=run_conf["model"]["num_heads"],
            extract_layers=run_conf["model"]["extract_layers"],
            regression_loss=run_conf["model"].get("regression_loss", False),
        )
    else:
        raise NotImplementedError(f"Unsupported arch: {arch}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def preprocess_uint8(raw_image: np.ndarray, normalizer: A.Compose) -> torch.Tensor:
    image = normalizer(image=raw_image)["image"]
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0)


def count_params(model: torch.nn.Module) -> float:
    return float(sum(p.numel() for p in model.parameters()) / 1e6)


def extract_flops_from_profiler(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> tuple[float, str]:
    try:
        from torch.profiler import ProfilerActivity, profile

        with torch.inference_mode():
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_flops=True,
                record_shapes=False,
                profile_memory=False,
            ) as prof:
                _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

        total_flops = 0.0
        for evt in prof.key_averages():
            total_flops += float(getattr(evt, "flops", 0.0) or 0.0)
        if total_flops > 0:
            return total_flops / 1e9, "torch_profiler"
    except Exception:
        pass

    return float("nan"), "unavailable"


def fallback_flops_from_log(run_dir: Path) -> float:
    log_path = run_dir / "logs.log"
    if not log_path.is_file():
        return float("nan")
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Total mult-adds \(G\):\s*([0-9.]+)", text)
    if match is None:
        return float("nan")
    return float(match.group(1))


def benchmark_latency(
    model: torch.nn.Module,
    normalizer: A.Compose,
    device: torch.device,
    input_size: int,
    batch_size: int,
    warmup: int,
    iters: int,
) -> list[float]:
    raw_image = np.random.randint(
        0, 256, size=(input_size, input_size, 3), dtype=np.uint8
    )

    times_ms: list[float] = []
    total = warmup + iters

    with torch.inference_mode():
        for step in range(total):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter()

            batch = [preprocess_uint8(raw_image, normalizer) for _ in range(batch_size)]
            x = torch.cat(batch, dim=0).to(device, non_blocking=True)
            _ = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (perf_counter() - start) * 1000.0
            if step >= warmup:
                times_ms.append(elapsed_ms)

    return times_ms


def benchmark_run(
    dataset_name: str,
    run_dir: Path,
    output_root: Path,
    gpu: int,
    input_size: int,
    batch_size: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device(f"cuda:{gpu}")
    run_conf = load_run_conf(run_dir)
    checkpoint_path = run_dir / "checkpoints" / "model_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = instantiate_model(run_conf, checkpoint).to(device)

    params_m = count_params(model)
    normalizer = build_normalizer(run_conf)

    profile_input = preprocess_uint8(
        np.random.randint(0, 256, size=(input_size, input_size, 3), dtype=np.uint8),
        normalizer,
    ).to(device)
    flops_g, flops_source = extract_flops_from_profiler(model, profile_input, device)
    if not math.isfinite(flops_g) or flops_g <= 0:
        flops_g = fallback_flops_from_log(run_dir)
        flops_source = "logs_mult_adds"

    times_ms = benchmark_latency(
        model=model,
        normalizer=normalizer,
        device=device,
        input_size=input_size,
        batch_size=batch_size,
        warmup=warmup,
        iters=iters,
    )

    if not times_ms:
        raise RuntimeError("Latency benchmark produced no samples.")

    result = {
        "dataset": dataset_name,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "params_M": round(params_m, 6),
        "flops_G": round(flops_g, 6) if math.isfinite(flops_g) else None,
        "flops_source": flops_source,
        "latency_ms": round(float(np.mean(times_ms)), 6),
        "latency_std": round(float(np.std(times_ms, ddof=0)), 6),
        "gpu_model": torch.cuda.get_device_name(device),
        "pytorch_ver": torch.__version__,
        "cuda_ver": torch.version.cuda,
        "batch_size": batch_size,
        "input_size": input_size,
        "warmup": warmup,
        "iters": iters,
    }

    outdir = output_root / dataset_name
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "efficiency.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CellVTA efficiency benchmark")
    parser.add_argument("--conic-run", type=Path, default=None, help="CoNIC run directory")
    parser.add_argument("--monuseg-run", type=Path, default=None, help="MoNuSeg run directory")
    parser.add_argument("--output-root", type=Path, required=True, help="Output directory")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument("--input-size", type=int, default=256, help="Input spatial size")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Measured iterations")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary: dict[str, Any] = {}

    if args.conic_run is not None:
        summary["conic"] = benchmark_run(
            "conic",
            args.conic_run,
            args.output_root,
            args.gpu,
            args.input_size,
            args.batch_size,
            args.warmup,
            args.iters,
        )
    if args.monuseg_run is not None:
        summary["monuseg"] = benchmark_run(
            "monuseg",
            args.monuseg_run,
            args.output_root,
            args.gpu,
            args.input_size,
            args.batch_size,
            args.warmup,
            args.iters,
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
