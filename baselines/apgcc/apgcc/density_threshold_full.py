#!/usr/bin/env python3
"""Density-adaptive threshold (2-fold parity CV) — SAME assignment as density_threshold.py,
but also caches predicted-point COORDINATES so we can report MSE / Recall / F1@12 (Hungarian),
not just counting MAE. Used to fully populate the +density-adaptive row of tab:A-ablation."""
import argparse, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--eval-list", default="test.list")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ref", type=float, default=0.3)
    ap.add_argument("--nbins", type=int, default=3)
    ap.add_argument("--radius", type=float, default=12.0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch, torch.nn.functional as F
    from scipy.optimize import linear_sum_assignment
    from config import cfg, merge_from_file
    cfg = merge_from_file(cfg, args.config)
    cfg.DATASETS.DATA_ROOT = args.data_root
    cfg.config_file = args.config
    from datasets import build_dataset
    from models import build_model

    _, dl = build_dataset(cfg, eval_list=args.eval_list)
    model = build_model(cfg, training=False); model.cuda(); model.eval()
    sd = torch.load(args.weight, map_location="cpu"); sd = sd.get("model", sd)
    md = model.state_dict(); md.update({k: v for k, v in sd.items() if k in md}); model.load_state_dict(md)

    scores, coords, gtpts, gts = [], [], [], []
    with torch.no_grad():
        for samples, targets in dl:
            out = model(samples.cuda())
            s = F.softmax(out["pred_logits"], -1)[:, :, 1][0].cpu().numpy()
            p = out["pred_points"][0].cpu().numpy()
            g = targets[0]["point"].cpu().numpy()
            scores.append(s); coords.append(p); gtpts.append(g); gts.append(len(g))
    gts = np.array(gts, float); n = len(gts)

    grid = np.round(np.arange(0.10, 0.605, 0.05), 2)
    pc = np.array([[(s > t).sum() for t in grid] for s in scores], float)
    dens = np.array([(s > args.ref).sum() for s in scores], float)
    abserr = np.abs(pc - gts[:, None])

    def density_assign(train_idx, test_idx):
        edges = np.quantile(dens[train_idx], np.linspace(0, 1, args.nbins + 1)[1:-1])
        def binof(v): return int(np.searchsorted(edges, v))
        bin_thr = {}
        for b in range(args.nbins):
            idx = np.array(train_idx)[[binof(dens[i]) == b for i in train_idx]]
            bin_thr[b] = grid[np.argmin(abserr[idx].mean(0))] if len(idx) else grid[np.argmin(abserr[train_idx].mean(0))]
        return {i: float(bin_thr[binof(dens[i])]) for i in test_idx}

    fa = [i for i in range(n) if i % 2 == 0]
    fb = [i for i in range(n) if i % 2 == 1]
    assign = {}
    assign.update(density_assign(fb, fa))   # learn on B, apply to A
    assign.update(density_assign(fa, fb))   # learn on A, apply to B

    # --- self-check: reproduce the count-only CV MAEs from density_threshold.py ---
    def global_cv(tr, te):
        ti = np.argmin(np.abs(grid - grid[np.argmin(abserr[tr].mean(0))]))
        return abserr[te, ti].sum()
    g_cv = (global_cv(fb, fa) + global_cv(fa, fb)) / n
    d_cv = sum(abserr[i, np.argmin(np.abs(grid - assign[i]))] for i in range(n)) / n
    print(f"[self-check] GLOBAL-CV MAE={g_cv:.3f}  DENSITY-CV MAE={d_cv:.3f}  (json: 10.781 / 10.581)")

    # dump density-adaptive predictions + GT in unified format, then call the canonical
    # centroid_eval.py (so MSE/Recall/F1 match every other row in the report exactly).
    import subprocess, sys, tempfile
    from pathlib import Path
    gt_s, pred_s = [], []
    for i in range(n):
        keep = scores[i] > assign[i]
        gt_s.append({"id": str(i), "points": gtpts[i].tolist()})
        pred_s.append({"id": str(i), "points": coords[i][keep].tolist()})
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "gt.json").write_text(json.dumps({"samples": gt_s}))
    (out / "pred.json").write_text(json.dumps({"samples": pred_s}))
    repo_eval = Path(__file__).resolve().parents[3] / "centroid_eval.py"
    print("=" * 60)
    print("DENSITY-ADAPTIVE (2-fold parity CV)  n=%d ref=%.2f  -> canonical centroid_eval" % (n, args.ref))
    print("=" * 60)
    subprocess.run([sys.executable, str(repo_eval), "--gt", str(out / "gt.json"),
                    "--pred", str(out / "pred.json"), "--thresholds", "6", "12", "24"], check=True)


if __name__ == "__main__":
    main()
