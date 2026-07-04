#!/usr/bin/env python3
"""APGCC efficiency benchmark, mirroring benchmark_cellvta_efficiency.py:
  - params_M       : #parameters / 1e6
  - flops_G        : forward GFLOPs via torch.profiler (with_flops)
  - latency_ms     : mean +/- std forward latency over `iters` (after `warmup`)
on a synthetic input_size x input_size RGB input (forward-only, input pre-loaded
to device). The APGCC architecture is identical across the 3 datasets, so
params/FLOPs are dataset-independent; latency is measured per given config.

Example:
  python benchmark_efficiency.py --config ./configs/CoNIC_finetune.yml \
      --weight ./output/CoNIC_finetune/best.pth --gpu 2 \
      --input-size 256 --batch-size 1 --warmup 10 --iters 100 \
      --out ./output/CoNIC_finetune/efficiency.json
"""
import argparse
import json
import math
import os
from time import perf_counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weight", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--input-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import numpy as np
    import torch

    from config import cfg, merge_from_file
    from models import build_model
    cfg = merge_from_file(cfg, args.config)
    cfg.config_file = args.config

    device = torch.device("cuda")
    model = build_model(cfg, training=False).to(device).eval()
    if args.weight:
        sd = torch.load(args.weight, map_location="cpu")
        sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        model.load_state_dict(sd, strict=False)

    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    x = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)

    # FLOPs via torch.profiler
    flops_g, flops_source = float("nan"), "unavailable"
    try:
        from torch.profiler import ProfilerActivity, profile
        with torch.inference_mode():
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         with_flops=True) as prof:
                _ = model(x)
                torch.cuda.synchronize(device)
        total = sum(float(getattr(e, "flops", 0.0) or 0.0) for e in prof.key_averages())
        if total > 0:
            flops_g, flops_source = total / 1e9, "torch_profiler"
    except Exception as e:
        flops_source = "error:%s" % type(e).__name__

    # latency (forward-only, input already on device)
    times_ms = []
    with torch.inference_mode():
        for step in range(args.warmup + args.iters):
            torch.cuda.synchronize(device)
            t0 = perf_counter()
            _ = model(x)
            torch.cuda.synchronize(device)
            if step >= args.warmup:
                times_ms.append((perf_counter() - t0) * 1000.0)

    result = {
        "params_M": round(params_m, 6),
        "flops_G": round(flops_g, 6) if math.isfinite(flops_g) else None,
        "flops_source": flops_source,
        "latency_ms": round(float(np.mean(times_ms)), 6),
        "latency_std": round(float(np.std(times_ms)), 6),
        "throughput_fps": round(args.batch_size * 1000.0 / float(np.mean(times_ms)), 4),
        "gpu_model": torch.cuda.get_device_name(device),
        "pytorch_ver": torch.__version__,
        "cuda_ver": torch.version.cuda,
        "batch_size": args.batch_size,
        "input_size": args.input_size,
        "warmup": args.warmup,
        "iters": args.iters,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
