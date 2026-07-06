#!/usr/bin/env python3
"""Phase 0 — no-retraining diagnosis on an existing checkpoint.

Decision gate for direction A2: is CoNIC's undercount caused by
  (a) UNDERCONFIDENT proposals  -> a proposal exists near the missed cell but its
      score sits below the 0.50 cutoff  => EOS_COEF / focal / threshold work helps; OR
  (b) ABSENT proposals          -> no proposal lands near the missed cell at all
      => classifier/threshold tricks cannot recover it.

Runs the model ONCE over a split, keeps ALL proposals (no score filtering), then:
  1) Coverage / FN-score analysis: for each GT, nearest proposal (<=match_px) and its
     score -> classify GT as covered_high (TP@0.50) / covered_low (recoverable) /
     no_proposal (absent). Reported overall + per subset.
  2) Score histogram of the nearest-proposal score for currently-missed GTs.
  3) Density signal: per-image pred_count@low vs per-image oracle threshold, to judge
     whether an inference-time density-adaptive threshold could help (correlation).

Example:
  python phase0_analysis.py --config ./configs/CoNIC_unified.yml \
      --weight ./output/CoNIC_unified/best.pth --data-root /data1/llx/CoNICdata \
      --eval-list test.list --gpu 3 --out-dir output/CoNIC_unified/phase0
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def subset_of(sid):
    for sep in ("_", "-"):
        if sep in sid:
            sid = sid.split(sep, 1)[0]
    return sid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--eval-list", default="test.list")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--match-px", type=float, default=12.0,
                    help="a GT is 'covered' if a proposal lies within this px")
    ap.add_argument("--cutoff", type=float, default=0.50,
                    help="reference score cutoff that defines 'currently kept'")
    ap.add_argument("--low", type=float, default=0.15,
                    help="low score used for density signal pred_count@low")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scipy.spatial.distance import cdist

    from config import cfg, merge_from_file
    cfg = merge_from_file(cfg, args.config)
    cfg.DATASETS.DATA_ROOT = args.data_root
    cfg.config_file = args.config
    from datasets import build_dataset
    from models import build_model

    _, dl = build_dataset(cfg, eval_list=args.eval_list)
    model = build_model(cfg, training=False)
    model.cuda(); model.eval()
    sd = torch.load(args.weight, map_location="cpu")
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    md = model.state_dict(); md.update({k: v for k, v in sd.items() if k in md})
    model.load_state_dict(md)

    samples_rec = []  # per image: id, gt(Nx2), pts(Mx2), scores(M)
    with torch.no_grad():
        for samples, targets in dl:
            samples = samples.cuda()
            out = model(samples)
            scores = F.softmax(out["pred_logits"], -1)[:, :, 1][0].cpu().numpy()
            pts = out["pred_points"][0].cpu().numpy()
            gt = targets[0]["point"].cpu().numpy()
            sid = os.path.splitext(targets[0]["name"])[0]
            samples_rec.append({"id": sid, "gt": gt, "pts": pts, "scores": scores})

    # ---- 1) Coverage / FN-score analysis ----
    cats = ("covered_high", "covered_low", "no_proposal")
    overall = {c: 0 for c in cats}
    by_sub = defaultdict(lambda: {c: 0 for c in cats})
    missed_scores = []          # nearest-proposal score of GTs not covered_high
    no_prop_total = 0
    gt_total = 0
    for r in samples_rec:
        gt, pts, sc = r["gt"], r["pts"], r["scores"]
        sub = subset_of(r["id"])
        gt_total += len(gt)
        if len(gt) == 0:
            continue
        if len(pts) == 0:
            overall["no_proposal"] += len(gt)
            by_sub[sub]["no_proposal"] += len(gt)
            no_prop_total += len(gt)
            continue
        D = cdist(gt, pts)                 # [n_gt, n_prop]
        within = D <= args.match_px        # [n_gt, n_prop] bool
        for i in range(len(gt)):
            near = within[i]
            if not near.any():
                cat = "no_proposal"; no_prop_total += 1
                overall[cat] += 1; by_sub[sub][cat] += 1
                missed_scores.append(-1.0)
                continue
            max_s = float(sc[near].max())  # best proposal trying to detect this cell
            if max_s > args.cutoff:
                cat = "covered_high"
            else:
                cat = "covered_low"
            overall[cat] += 1; by_sub[sub][cat] += 1
            if cat != "covered_high":
                missed_scores.append(max_s)

    def pct(d):
        t = sum(d.values()) or 1
        return {k: (v, 100.0 * v / t) for k, v in d.items()}

    print("=" * 74)
    print("COVERAGE / FN-SOURCE ANALYSIS  (match<=%gpx, cutoff=%.2f)" %
          (args.match_px, args.cutoff))
    print("=" * 74)
    o = pct(overall)
    print(f"GT total = {gt_total}")
    for c in cats:
        print(f"  {c:>13}: {o[c][0]:>7d}  ({o[c][1]:5.1f}%)")
    rec = [s for s in missed_scores if s >= 0]
    print(f"\n  currently-missed GT with a nearby proposal (recoverable) = {len(rec)}")
    print(f"  currently-missed GT with NO nearby proposal (absent)      = "
          f"{sum(1 for s in missed_scores if s < 0)}")

    print("\nPer-subset (covered_high / covered_low / no_proposal %):")
    print(f"{'subset':>10} {'n_gt':>7} {'cov_hi%':>8} {'cov_lo%':>8} {'noprop%':>8}")
    for sub in sorted(by_sub):
        p = pct(by_sub[sub]); n = sum(by_sub[sub].values())
        print(f"{sub:>10} {n:>7d} {p['covered_high'][1]:>8.1f} "
              f"{p['covered_low'][1]:>8.1f} {p['no_proposal'][1]:>8.1f}")

    # ---- 2) histogram of recoverable missed-GT nearest-proposal scores ----
    rec_arr = np.array(rec)
    edges = np.arange(0.0, 0.55, 0.05)
    hist, _ = np.histogram(rec_arr, bins=edges)
    print("\nScore histogram of recoverable missed GT (nearest-proposal score):")
    for i in range(len(hist)):
        bar = "#" * int(40 * hist[i] / (hist.max() + 1e-9))
        print(f"  [{edges[i]:.2f},{edges[i+1]:.2f}) {hist[i]:>6d} {bar}")

    # ---- 3) density signal vs per-image oracle threshold ----
    thr_grid = np.round(np.arange(0.15, 0.51, 0.05), 2)
    dens, oracle_thr = [], []
    for r in samples_rec:
        sc, g = r["scores"], len(r["gt"])
        dens.append(int((sc > args.low).sum()))
        best_t, best_e = thr_grid[0], 1e18
        for t in thr_grid:
            e = abs(int((sc > t).sum()) - g)
            if e < best_e:
                best_e, best_t = e, t
        oracle_thr.append(float(best_t))
    dens = np.array(dens); oracle_thr = np.array(oracle_thr)
    corr = float(np.corrcoef(dens, oracle_thr)[0, 1]) if len(dens) > 1 else 0.0
    print("\n" + "=" * 74)
    print("DENSITY SIGNAL  (pred_count@%.2f  vs  per-image oracle threshold)" % args.low)
    print("=" * 74)
    print(f"  corr(pred_count@low, oracle_thr) = {corr:+.3f}  "
          f"(negative => denser images want LOWER threshold)")
    for lo, hi in [(0, 80), (80, 140), (140, 10**9)]:
        m = (dens >= lo) & (dens < hi)
        if m.sum():
            name = "sparse" if hi == 80 else ("medium" if hi == 140 else "dense")
            print(f"  {name:>7} (cnt@low in [{lo},{hi})): n={m.sum():>4d}  "
                  f"mean oracle thr={oracle_thr[m].mean():.3f}")

    # ---- save ----
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    result = {
        "config": args.config, "eval_list": args.eval_list,
        "match_px": args.match_px, "cutoff": args.cutoff,
        "gt_total": gt_total, "coverage_overall": overall,
        "coverage_by_subset": {k: dict(v) for k, v in by_sub.items()},
        "recoverable_missed": len(rec), "absent_missed": int((np.array(missed_scores) < 0).sum()),
        "score_hist_edges": edges.tolist(), "score_hist_counts": hist.tolist(),
        "density_corr": corr,
    }
    (out / "phase0.json").write_text(json.dumps(result, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].bar(edges[:-1], hist, width=0.045, align="edge")
        ax[0].axvline(args.cutoff, color="r", ls="--", label=f"cutoff {args.cutoff}")
        ax[0].set_title("Recoverable missed-GT nearest-proposal score")
        ax[0].set_xlabel("score"); ax[0].set_ylabel("# missed GT"); ax[0].legend()
        ax[1].scatter(dens, oracle_thr, s=6, alpha=0.3)
        ax[1].set_title(f"density vs oracle thr (corr={corr:+.2f})")
        ax[1].set_xlabel("pred_count@%.2f" % args.low); ax[1].set_ylabel("oracle thr")
        fig.tight_layout(); fig.savefig(out / "phase0.png", dpi=120)
        print(f"\nsaved {out/'phase0.png'}")
    except Exception as e:
        print("plot skipped:", e)
    print(f"saved {out/'phase0.json'}")


if __name__ == "__main__":
    main()
