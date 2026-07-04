#!/usr/bin/env python3
"""Measure STEERER efficiency following the shared PyTorch protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count
from mmcv import Config

import _init_paths
from lib.models.build_counter import Baseline_Counter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def load_model(config: Config, checkpoint: str, device: torch.device) -> torch.nn.Module:
    model = Baseline_Counter(
        config.network,
        config.dataset.den_factor,
        route_size=config.train.route_size,
        device=device,
    )
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def preprocess(raw_uint8: torch.Tensor, device: torch.device) -> torch.Tensor:
    # raw_uint8: NHWC, RGB, uint8. Output: NCHW, float32, ImageNet normalized.
    x = raw_uint8.to(device=device, dtype=torch.float32)
    x = x.permute(0, 3, 1, 2).contiguous() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def main() -> None:
    args = parse_args()
    config = Config.fromfile(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, args.checkpoint, device)

    raw = torch.randint(
        0,
        256,
        (1, args.height, args.width, 3),
        dtype=torch.uint8,
        device="cpu",
    )

    float_input = preprocess(raw, device)
    params = parameter_count(model)[""]
    try:
        flops = FlopCountAnalysis(model, float_input).total()
    except Exception as exc:
        flops = None
        flops_error = repr(exc)
    else:
        flops_error = None

    latencies = []
    with torch.no_grad():
        for _ in range(args.warmup):
            x = preprocess(raw, device)
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        for _ in range(args.iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            x = preprocess(raw, device)
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)

    result = {
        "input_size": [args.height, args.width, 3],
        "input_dtype": "uint8",
        "batch_size": 1,
        "warmup": args.warmup,
        "repeat": args.iters,
        "params_M": params / 1e6,
        "flops_G": None if flops is None else flops / 1e9,
        "latency_ms": float(np.mean(latencies)),
        "latency_std": float(np.std(latencies, ddof=1)) if len(latencies) > 1 else 0.0,
        "fps": float(1000.0 / np.mean(latencies)),
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "pytorch_ver": torch.__version__,
        "cuda_ver": torch.version.cuda,
        "timing_scope": "raw uint8 tensor -> preprocessing -> model output, with torch.cuda.synchronize",
        "flops_error": flops_error,
    }

    print("=" * 60)
    print("EFFICIENCY METRICS")
    print("=" * 60)
    for key in (
        "params_M",
        "flops_G",
        "latency_ms",
        "latency_std",
        "fps",
        "gpu_model",
        "pytorch_ver",
        "cuda_ver",
    ):
        print(f"{key}: {result[key]}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
