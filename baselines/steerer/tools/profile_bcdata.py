"""Measure STEERER parameters, FLOPs and synchronized forward latency.

Synthetic input isolates model inference from data I/O and point post-processing.
The historical filename mentions BCData, but any compatible config is accepted.
"""

import argparse
import time
import torch
from mmcv import Config
from fvcore.nn import FlopCountAnalysis, parameter_count_table

import _init_paths
from lib.models.build_counter import Baseline_Counter


def main():
    parser = argparse.ArgumentParser(description="Profile a trained STEERER model.")
    parser.add_argument("--cfg", default="configs/BCData_train.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    config = Config.fromfile(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Baseline_Counter(
        config.network,
        config.dataset.den_factor,
        route_size=config.train.route_size,
        device=device,
    )
    # Accept raw, wrapped and DistributedDataParallel checkpoint formats.
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {key.replace("module.", "", 1): value for key, value in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    # Batch size one matches the cross-baseline efficiency protocol.
    x = torch.randn(1, 3, args.height, args.width).to(device)

    print("Parameter count:")
    print(parameter_count_table(model))

    try:
        flops = FlopCountAnalysis(model, x)
        print("FLOPs:")
        print(f"{flops.total() / 1e9:.4f} GFLOPs")
    except Exception as exc:
        print("FLOPs calculation failed:", repr(exc))

    with torch.no_grad():
        # Warm-up excludes CUDA initialization; synchronization below is
        # required because GPU kernels otherwise launch asynchronously.
        for _ in range(args.warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(args.iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

    avg_ms = elapsed / args.iters * 1000
    fps = 1000 / avg_ms
    print("Inference time:")
    print(f"{avg_ms:.3f} ms/image")
    print(f"{fps:.2f} FPS")


if __name__ == "__main__":
    main()
