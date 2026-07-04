#!/usr/bin/env python3
"""Benchmark STEERER efficiency with the shared PyTorch protocol.

The timing range is:
  raw uint8 tensor -> normalization/preprocessing -> model output

This script is adapted from the CellVTA benchmark style but loads STEERER
configs/checkpoints directly.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch


STEERER_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = STEERER_ROOT / "tools"
for path in (STEERER_ROOT, TOOLS_DIR):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified STEERER efficiency benchmark")
    parser.add_argument("--cfg", required=True, type=Path, help="STEERER config file")
    parser.add_argument("--checkpoint", required=True, type=Path, help="STEERER .pth checkpoint")
    parser.add_argument("--output-root", type=Path, default=None, help="Directory for efficiency.json")
    parser.add_argument("--output", type=Path, default=None, help="Optional explicit output JSON path")
    parser.add_argument("--dataset-name", default="steerer", help="Subdirectory name under --output-root")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument("--input-size", type=int, default=256, help="Square input size")
    parser.add_argument("--height", type=int, default=None, help="Input height; overrides --input-size")
    parser.add_argument("--width", type=int, default=None, help="Input width; overrides --input-size")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def load_model(config: Any, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    import _init_paths  # noqa: F401
    from lib.models.build_counter import Baseline_Counter

    model = Baseline_Counter(
        config.network,
        config.dataset.den_factor,
        route_size=config.train.route_size,
        device=device,
    )

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(checkpoint)}")

    checkpoint = {
        key.replace("module.", "", 1): value
        for key, value in checkpoint.items()
    }
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()
    model._steerer_missing_keys = list(missing)  # type: ignore[attr-defined]
    model._steerer_unexpected_keys = list(unexpected)  # type: ignore[attr-defined]
    return model


def preprocess_uint8(raw: torch.Tensor, device: torch.device) -> torch.Tensor:
    # raw: NHWC uint8 on CPU. Output: NCHW float32 normalized on device.
    x = raw.to(device=device, dtype=torch.float32, non_blocking=True)
    x = x.permute(0, 3, 1, 2).contiguous() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def count_params_m(model: torch.nn.Module) -> float:
    return float(sum(param.numel() for param in model.parameters()) / 1e6)


def flops_with_torch_profiler(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> tuple[float, str]:
    try:
        from torch.profiler import ProfilerActivity, profile

        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        with torch.inference_mode():
            with profile(
                activities=activities,
                with_flops=True,
                record_shapes=False,
                profile_memory=False,
            ) as prof:
                _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        total_flops = 0.0
        for evt in prof.key_averages():
            total_flops += float(getattr(evt, "flops", 0.0) or 0.0)
        if total_flops > 0:
            return total_flops / 1e9, "torch_profiler"
    except Exception:
        pass

    return float("nan"), "unavailable"


def flops_with_fvcore(model: torch.nn.Module, x: torch.Tensor) -> tuple[float, str]:
    try:
        from fvcore.nn import FlopCountAnalysis

        flops = FlopCountAnalysis(model, x).total()
        if flops > 0:
            return float(flops / 1e9), "fvcore"
    except Exception:
        pass
    return float("nan"), "unavailable"


def measure_latency_ms(
    model: torch.nn.Module,
    raw: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
) -> list[float]:
    times_ms: list[float] = []
    total_iters = warmup + iters

    with torch.inference_mode():
        for step in range(total_iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = perf_counter()

            x = preprocess_uint8(raw, device)
            _ = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = (perf_counter() - start) * 1000.0
            if step >= warmup:
                times_ms.append(elapsed)

    return times_ms


def make_output_path(args: argparse.Namespace, height: int, width: int) -> Path | None:
    if args.output is not None:
        return args.output
    if args.output_root is None:
        return None
    out_dir = args.output_root / args.dataset_name
    return out_dir / f"efficiency_{height}x{width}.json"


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    height = args.height if args.height is not None else args.input_size
    width = args.width if args.width is not None else args.input_size
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    from mmcv import Config

    config = Config.fromfile(str(args.cfg))
    model = load_model(config, args.checkpoint, device)

    raw = torch.randint(
        0,
        256,
        (args.batch_size, height, width, 3),
        dtype=torch.uint8,
        device="cpu",
    )
    profile_input = preprocess_uint8(raw, device)

    flops_g, flops_source = flops_with_torch_profiler(model, profile_input, device)
    if not math.isfinite(flops_g) or flops_g <= 0:
        flops_g, flops_source = flops_with_fvcore(model, profile_input)

    times_ms = measure_latency_ms(model, raw, device, args.warmup, args.iters)
    if not times_ms:
        raise RuntimeError("Latency benchmark produced no samples.")

    mean_ms = float(np.mean(times_ms))
    result = {
        "method": "STEERER",
        "dataset": args.dataset_name,
        "cfg": str(args.cfg),
        "checkpoint": str(args.checkpoint),
        "params_M": round(count_params_m(model), 6),
        "flops_G": round(float(flops_g), 6) if math.isfinite(flops_g) else None,
        "flops_source": flops_source,
        "latency_ms": round(mean_ms, 6),
        "latency_std": round(float(np.std(times_ms, ddof=0)), 6),
        "fps": round(float(args.batch_size * 1000.0 / mean_ms), 6),
        "gpu_model": torch.cuda.get_device_name(device),
        "pytorch_ver": torch.__version__,
        "cuda_ver": torch.version.cuda,
        "batch_size": args.batch_size,
        "input_size": [height, width, 3],
        "input_dtype": "uint8",
        "warmup": args.warmup,
        "iters": args.iters,
        "timing_scope": "raw uint8 tensor -> preprocessing -> model output, with torch.cuda.synchronize",
        "missing_keys": getattr(model, "_steerer_missing_keys", []),
        "unexpected_keys": getattr(model, "_steerer_unexpected_keys", []),
    }

    output_path = make_output_path(args, height, width)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        result["output"] = str(output_path)

    return result


def main() -> None:
    args = parse_args()
    result = benchmark(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
