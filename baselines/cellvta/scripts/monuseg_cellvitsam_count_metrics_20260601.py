import json
import math
import time
import numpy as np
import torch
from tqdm import tqdm
from einops import rearrange
from cellvit.training.evaluate.inference_cellvit_experiment_monuseg import MoNuSegInference

model_path = "/data/zju-151/liyixin/project/pathology-cell-counting/downloads/gdrive_retry/SAM/CellViT-SAM-H-x40-AMP.redownload.pth"
dataset_path = "/data/zju-151/liyixin/project/pathology-cell-counting/datasets/processed/monuseg_full_1024"
outdir = "/data/zju-151/liyixin/project/pathology-cell-counting/results/monuseg_cellvitsam_1024"

inf = MoNuSegInference(
    model_path=model_path,
    dataset_path=dataset_path,
    outdir=outdir,
    gpu=0,
    patching=True,
    overlap=64,
    magnification=40,
)

loader = inf.inference_dataloader
abs_err = []
sq_err = []
times = []

for idx, batch in enumerate(tqdm(loader, total=len(loader))):
    img = batch[0]
    mask = batch[1]
    name = batch[2][0] if isinstance(batch[2], (list, tuple)) else str(batch[2])
    if len(img.shape) > 4:
        img = img[0]
        img = rearrange(img, "c i j w h -> (i j) c w h")
    img = img.to(inf.device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        if inf.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds = inf.model.forward(img)
        else:
            preds = inf.model.forward(img)
        cell_list = inf.post_process_patching_overlap(preds, overlap=64)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000.0

    pred_count = int(len(cell_list))
    gt_count = int(mask["instance_map"].max().item())
    ae = abs(pred_count - gt_count)
    se = (pred_count - gt_count) ** 2

    abs_err.append(ae)
    sq_err.append(se)
    times.append(dt)

    print(json.dumps({
        "index": idx,
        "name": name,
        "pred_count": pred_count,
        "gt_count": gt_count,
        "abs_error": float(ae),
        "sq_error": float(se),
        "time_ms": float(dt),
    }, ensure_ascii=False), flush=True)

result = {
    "num_images": len(abs_err),
    "mae": float(np.mean(abs_err)),
    "mse": float(math.sqrt(np.mean(sq_err))),
    "mean_time_ms": float(np.mean(times)),
    "median_time_ms": float(np.median(times)),
    "fps": float(1000.0 / np.mean(times)),
}
print("FINAL_RESULTS=" + json.dumps(result, ensure_ascii=False), flush=True)
