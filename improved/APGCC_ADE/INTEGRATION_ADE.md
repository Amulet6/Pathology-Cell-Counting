# A + D + E integration on CoNIC

Combines three directions on the APGCC (VGG16-bn + IFI) base, built **on top of Direction-D's
network** (`improved/APGCC_D` → copied here):

| Dir | What it adds | Where |
|-----|--------------|-------|
| **D** | DCNv2 (conv5 last 2 layers) + Edge Ignore (band 16, weight 0.1) | `models/dcnv2.py`, `models/Encoder.py`, `models/APGCC.py`, `engine.py`, `util/misc.py` |
| **E** | Pathology / stain augmentation (train-only) | `datasets/dataset.py` (`pathology_augment`, `get_domain_label`), `config.py` (`DATALOADER.PATHOLOGY_AUG/STAIN_JITTER/BLUR_NOISE/DOMAIN_NAMES`) |
| **A** | EOS↓ + τ↑ + softmax-focal + density-adaptive threshold | `config.py` (`MODEL.FOCAL_GAMMA`), `models/APGCC.py` (focal in `loss_labels`), `density_threshold.py`, `scan_threshold.py` |

Config: **`configs/CoNIC_finetune_dcnv2_edge_stain.yml`**
(`EOS_COEF=0.10`, `SET_COST_POINT=0.10`, `FOCAL_GAMMA=2.0`, `DCNV2_ENABLED=True`, `DCNV2_LAYERS=2`,
`EDGE_BAND=16`, `EDGE_WEIGHT=0.1`, `PATHOLOGY_AUG/STAIN_JITTER/BLUR_NOISE=True`).

## Important note on A's recipe

A's `EOS0.10/τ0.10/focal` recipe was tuned on A's own base. Here, `loss_ce` follows **Direction-D's
normalization** (`sum / num_points`) so it composes cleanly with the edge mask — not A's
`sum / alpha.sum()`. Treat the EOS/τ/γ values as a **strong starting point**, then re-verify on this
combined base with `scan_threshold.py` (counting-threshold sweep on val) before reporting.

## Pipeline (env: `conda activate apgcc`, data `/data1/llx/CoNICdata`)

```bash
cd improved/APGCC_ADE/apgcc
# 0. SHHA pretrained weight is symlinked at ./output/SHHA_best.pth

# 1. Train (A+D+E)
python main.py -c ./configs/CoNIC_finetune_dcnv2_edge_stain.yml

# 2. Baseline eval at fixed threshold 0.5 (D's wrapper; --edge-band is a no-op on CoNIC full patches)
python eval_centroid.py --config ./configs/CoNIC_finetune_dcnv2_edge_stain.yml \
   --weight ./output/CoNIC_ADE_dcnv2_edge_stain/best.pth \
   --data-root /data1/llx/CoNICdata --eval-list test.list --gpu 3

# 3a. A's threshold path — pick global threshold on val, then density-adaptive (CV-honest) on test
python scan_threshold.py   --config ... --weight .../best.pth --data-root /data1/llx/CoNICdata \
   --eval-list val.list --gpu 3 --out ./output/CoNIC_ADE_dcnv2_edge_stain/val_scan/scan.json
python density_threshold.py --config ... --weight .../best.pth --data-root /data1/llx/CoNICdata \
   --eval-list test.list --gpu 3 --ref 0.15 \
   --out ./output/CoNIC_ADE_dcnv2_edge_stain/density_thr.json

# 3b. E's threshold path (for the A-vs-E comparison) — per-source-domain MAE-optimal threshold
python eval_domain_threshold.py --config ... --weight .../best.pth \
   --data-root /data1/llx/CoNICdata --val-list val.list --test-list test.list \
   --out-dir ./output/CoNIC_ADE_dcnv2_edge_stain/domain_threshold_eval_mae \
   --select-metric mae --score-candidates 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 \
   --thresholds 6 12 24 --gpu 3
```

## A-vs-E threshold comparison (the deliverable)

Both run post-hoc on the **same** trained A+D+E weights; select on `val.list`, report once on `test.list`:

- **A — density-adaptive** (`density_threshold.py`): per-image threshold from a test-time density
  signal (pred-count @ low ref), chosen by 2-fold CV → reports GLOBAL-CV / DENSITY-CV / PERIMG-ORACLE.
- **E — DomainTh-MAE** (`eval_domain_threshold.py`): one MAE-optimal threshold per source domain
  (crag/dpath/glas/pannuke/consep), fixed from val.

Per the team plan, **A's method is the adopted threshold**; E's DomainTh-MAE is the comparison baseline.

## Validation done

- `dcnv2.py` self-check passes (zero-init identity, backward NaN-free, ckpt load).
- Full build from the combined config + 1 train sample + forward/backward on GPU:
  - target carries both `crop_size` (D) and `domain` (E); stain aug runs.
  - 2 `ModulatedDeformConv2d` injected; criterion `focal_gamma=2.0`, `edge_band=16`.
  - `loss_ce`(focal+edge)+`loss_points` finite; DCNv2 offset grads finite; edge mask `(32,1024)`, ramp 0.21→1.0.
