#!/usr/bin/env python3
"""HoVerNet 效率测量脚本。

支持两种模型结构：
  - CoNIC 分支 (HoVerNetExt): --num-types 7
  - official 分支 (HoVerNet): --mode original 或不传 --num-types

测量指标：参数量、FLOPs、推理延迟（256×256, batch=1, warmup=10, 100 次平均）。

用法：
    # CoNIC 模型
    python benchmark_hovernet_efficiency.py \
        --ckpt ...baseline_unified/00/model/01/net_epoch=37.tar \
        --num-types 7 --output-dir results/efficiency/

    # MoNuSeg / CoNSeP 模型
    python benchmark_hovernet_efficiency.py \
        --ckpt ...monuseg/01/net_epoch=24.tar \
        --mode original --output-dir results/efficiency/
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

REPO_DIR = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = REPO_DIR / "official"
CONIC_DIR = REPO_DIR / "conic_branch"


def load_hovernet_model_official(
    ckpt_path: str, mode: str = "original", device: torch.device = torch.device("cpu")
) -> torch.nn.Module:
    """加载 official 分支 HoVerNet checkpoint (MoNuSeg / CoNSeP)。"""
    if str(OFFICIAL_DIR) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_DIR))

    from models.hovernet.net_desc import create_model
    from run_utils.utils import convert_pytorch_checkpoint

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "desc" in ckpt:
        state_dict = ckpt["desc"]
        state_dict = convert_pytorch_checkpoint(state_dict)
    else:
        state_dict = ckpt

    # 从 state_dict 推断 nr_types：找 tp decoder 最后一层 conv weight shape[0]
    nr_types = None
    for k, v in state_dict.items():
        if "decoder.tp" in k and "conv.weight" in k and "u0" in k:
            nr_types = v.shape[0]
            break

    model = create_model(mode=mode, nr_types=nr_types)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def load_hovernet_model_conic(
    ckpt_path: str, num_types: int = 7, device: torch.device = torch.device("cpu")
) -> torch.nn.Module:
    """加载 CoNIC 分支 HoVerNetExt checkpoint。"""
    if str(CONIC_DIR) not in sys.path:
        sys.path.insert(0, str(CONIC_DIR))

    from models.hovernet.net_desc import HoVerNetExt
    from run_utils.utils import convert_pytorch_checkpoint

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "desc" in ckpt:
        state_dict = ckpt["desc"]
        state_dict = convert_pytorch_checkpoint(state_dict)
    else:
        state_dict = ckpt

    # 传已有 pretrained_backbone 路径避免 None 导致的 bug
    pretrained_path = CONIC_DIR / "exp_output/local/[ImageNet]resnet50-0676ba61.pth"
    model = HoVerNetExt(
        num_types=num_types,
        freeze=False,
        pretrained_backbone=str(pretrained_path) if pretrained_path.exists() else "",
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def count_params(model: torch.nn.Module) -> float:
    """返回参数量（百万）。"""
    return float(sum(p.numel() for p in model.parameters()) / 1e6)


def extract_flops_from_profiler(
    model: torch.nn.Module, x: torch.Tensor, device: torch.device
) -> tuple[float, str]:
    """使用 torch.profiler 提取 FLOPs（G）。"""
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


def benchmark_latency(
    model: torch.nn.Module,
    device: torch.device,
    input_size: int = 256,
    batch_size: int = 1,
    warmup: int = 10,
    iters: int = 100,
) -> list[float]:
    """测量推理延迟（ms）。输入 uint8 图像，模型内部 /255 归一化。"""
    raw_image = np.random.randint(0, 256, size=(input_size, input_size, 3), dtype=np.uint8)
    x = torch.from_numpy(raw_image).permute(2, 0, 1).unsqueeze(0)
    x = x.expand(batch_size, -1, -1, -1).to(device)

    times_ms: list[float] = []
    total = warmup + iters

    with torch.inference_mode():
        for step in range(total):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter()

            _ = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (perf_counter() - start) * 1000.0
            if step >= warmup:
                times_ms.append(elapsed_ms)

    return times_ms


def run_benchmark(
    ckpt_path: str,
    output_dir: Path,
    gpu: int = 0,
    num_types: int | None = None,
    mode: str | None = None,
    input_size: int = 256,
    batch_size: int = 1,
    warmup: int = 10,
    iters: int = 100,
) -> dict[str, Any]:
    """执行完整效率测量。"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device(f"cuda:{gpu}")

    # 自动选择模型加载方式
    if num_types is not None:
        model_type = "HoVerNetExt (CoNIC)"
        model = load_hovernet_model_conic(ckpt_path, num_types=num_types, device=device)
    elif mode is not None:
        model_type = f"HoVerNet ({mode})"
        model = load_hovernet_model_official(ckpt_path, mode=mode, device=device)
    else:
        model_type = "HoVerNet (original)"
        model = load_hovernet_model_official(ckpt_path, mode="original", device=device)

    params_m = count_params(model)

    # FLOPs — 用 uint8 输入
    profile_input = torch.randint(
        0, 256, size=(batch_size, 3, input_size, input_size), dtype=torch.uint8, device=device
    )
    flops_g, flops_source = extract_flops_from_profiler(model, profile_input, device)

    # 延迟
    times_ms = benchmark_latency(
        model=model, device=device, input_size=input_size,
        batch_size=batch_size, warmup=warmup, iters=iters,
    )

    if not times_ms:
        raise RuntimeError("No latency samples collected.")

    result: dict[str, Any] = {
        "model": model_type,
        "checkpoint": str(ckpt_path),
        "params_M": round(params_m, 3),
        "flops_G": round(flops_g, 4) if math.isfinite(flops_g) else None,
        "flops_source": flops_source,
        "latency_ms": round(float(np.mean(times_ms)), 3),
        "latency_std": round(float(np.std(times_ms, ddof=0)), 3),
        "latency_min": round(float(np.min(times_ms)), 3),
        "latency_max": round(float(np.max(times_ms)), 3),
        "gpu_model": torch.cuda.get_device_name(device),
        "pytorch_ver": torch.__version__,
        "cuda_ver": torch.version.cuda,
        "batch_size": batch_size,
        "input_size": input_size,
        "warmup": warmup,
        "iters": iters,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "efficiency.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HoVerNet efficiency benchmark")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint .tar 路径")
    parser.add_argument(
        "--num-types", type=int, default=None,
        help="细胞类型数。CoNIC=7。不传则使用 official 分支模型",
    )
    parser.add_argument(
        "--mode", type=str, default="original",
        help="official 分支的模式: original / fast。CoNIC 模型忽略此参数",
    )
    parser.add_argument("--output-dir", type=str, default="results/efficiency")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_benchmark(
        ckpt_path=args.ckpt,
        output_dir=Path(args.output_dir),
        gpu=args.gpu,
        num_types=args.num_types,
        mode=args.mode if args.num_types is None else None,
        input_size=args.input_size,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
