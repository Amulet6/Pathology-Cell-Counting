#!/usr/bin/env python3
"""Dump per-point predictions WITH class-1 scores (no thresholding) so a threshold
scan / re-thresholding can be done post-hoc. Same format as eval_centroid but points
are [x, y, score]. GT dumped as [x, y].
  python dump_scores.py --config ... --weight ... --data-root ... --gpu N --out FILE.json
"""
import argparse, json, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--eval-list", default="test.list")
    ap.add_argument("--min-score", type=float, default=0.01,
                    help="drop points below this to keep files small")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch, torch.nn.functional as F
    from config import cfg, merge_from_file
    cfg = merge_from_file(cfg, args.config)
    cfg.DATASETS.DATA_ROOT = args.data_root
    cfg.config_file = args.config
    from datasets import build_dataset
    from models import build_model

    _, dl = build_dataset(cfg, eval_list=args.eval_list)
    model = build_model(cfg, training=False); model.cuda(); model.eval()
    sd = torch.load(args.weight, map_location="cpu"); sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    md = model.state_dict(); md.update({k: v for k, v in sd.items() if k in md}); model.load_state_dict(md)

    samples_out = []
    with torch.no_grad():
        for samples, targets in dl:
            out = model(samples.cuda())
            score = F.softmax(out["pred_logits"], -1)[:, :, 1][0].cpu().numpy()
            pts = out["pred_points"][0].cpu().numpy()
            keep = score > args.min_score
            pp = [[float(x), float(y), float(s)] for (x, y), s in zip(pts[keep], score[keep])]
            gt = targets[0]["point"].cpu().numpy().tolist()
            sid = os.path.splitext(targets[0]["name"])[0]
            samples_out.append({"id": sid, "points": pp, "gt": gt})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"samples": samples_out}))
    print("wrote", args.out, len(samples_out), "samples")


if __name__ == "__main__":
    main()
