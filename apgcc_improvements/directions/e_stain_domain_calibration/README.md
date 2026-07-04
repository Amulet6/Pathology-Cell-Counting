# APGCC_E: Stain Augmentation + Domain-aware Threshold Calibration

Author/signature: **suqiseven**

This folder contains the Direction E APGCC improvement code for pathology cell counting/localization. It is organized so other teammates can copy the useful parts into their own improved APGCC branch.

## What Is Most Useful

The recommended integration is:

```text
APGCC + pathology stain augmentation + Domain-aware Threshold selected by val MAE
```

The most useful code for teammates is:

| Purpose | File in this package | Where to copy in APGCC |
|---|---|---|
| Stain/pathology augmentation preprocessing | `apgcc_patch/datasets/dataset.py` | `apgcc/datasets/dataset.py` |
| New config flags for stain aug / GRL / hard reweight | `apgcc_patch/config.py` | `apgcc/config.py` |
| Main stain training config | `apgcc_patch/configs/CoNIC_finetune_stain.yml` | `apgcc/configs/CoNIC_finetune_stain.yml` |
| Domain-aware threshold post-processing | `apgcc_patch/eval_domain_threshold.py` | `apgcc/eval_domain_threshold.py` |
| Domain + density threshold ablation | `apgcc_patch/eval_domain_density_threshold.py` | `apgcc/eval_domain_density_threshold.py` |

If a teammate only wants to reuse the strongest parts, copy:

```text
apgcc_patch/config.py
apgcc_patch/datasets/dataset.py
apgcc_patch/eval_domain_threshold.py
apgcc_patch/configs/CoNIC_finetune_stain.yml
```

Then train with stain augmentation and evaluate with domain-aware threshold calibration.

## One-command Install Into Another APGCC Checkout

From this folder:

```bash
bash install_apgcc_e.sh /path/to/Pathology-Cell-Counting/baselines/apgcc/apgcc
```

The script copies the patched APGCC files and configs into the target `apgcc/` directory.

## Main Method

### 1. Training: pathology stain augmentation

Implemented in:

```text
apgcc_patch/datasets/dataset.py
```

Core function:

```python
pathology_augment(img, stain_jitter=True, blur_noise=True)
```

It applies light pathology-specific image perturbations:

- brightness jitter;
- contrast jitter;
- color jitter;
- RGB channel scale/bias;
- optional light blur/noise.

The main config enabling it is:

```text
apgcc_patch/configs/CoNIC_finetune_stain.yml
```

Important config fields:

```yaml
DATALOADER:
  PATHOLOGY_AUG: True
  STAIN_JITTER: True
  BLUR_NOISE: True
```

Train:

```bash
cd apgcc
python main.py -c ./configs/CoNIC_finetune_stain.yml
```

### 2. Post-processing: Domain-aware Threshold by validation MAE

Implemented in:

```text
apgcc_patch/eval_domain_threshold.py
```

This script scans score thresholds on `val.list` separately for each CoNIC source domain:

```text
consep / crag / dpath / glas / pannuke
```

It then fixes those thresholds and applies them once to `test.list`.

Recommended command:

```bash
python eval_domain_threshold.py \
  --config ./configs/CoNIC_finetune_stain.yml \
  --weight ./output/CoNIC_finetune_stain/best.pth \
  --data-root /path/to/CoNIC_APGCC_seed19 \
  --val-list val.list \
  --test-list test.list \
  --out-dir ./output/CoNIC_finetune_stain/domain_threshold_eval_mae \
  --select-metric mae \
  --score-candidates 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 \
  --thresholds 6 12 24 \
  --gpu 0
```

Selected thresholds in our run:

| Domain | Threshold |
|---|---:|
| consep | 0.25 |
| crag | 0.45 |
| dpath | 0.35 |
| glas | 0.55 |
| pannuke | 0.45 |

## Results Summary

CoNIC overlap test set, `test.list`, 991 samples, 112545 GT cells.

| Method | MAE | MSE | RMSE | P@12 | R@12 | F1@12 | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| APGCC baseline | 12.9273 | 468.8628 | 21.6532 | 0.8161 | 0.7709 | 0.7929 | native baseline |
| Stain Aug | 12.4581 | 449.6448 | 21.2048 | 0.8376 | 0.7718 | 0.8034 | useful training preprocessing |
| Stain Aug + DomainTh-MAE | **11.4844** | **409.1877** | **20.2284** | 0.8263 | **0.7783** | 0.8016 | recommended final method |
| Domain + Density-aware Threshold | 11.5661 | 409.9637 | 20.2476 | 0.8246 | 0.7791 | 0.8012 | ablation, not better |
| Weak Warmup GRL | 16.2260 | 716.4541 | 26.7667 | 0.8524 | 0.7517 | 0.7989 | higher precision, worse recall |
| Hard-density Reweight | 13.2735 | 420.3532 | 20.5025 | 0.8270 | 0.7629 | 0.7936 | improves glas/pannuke MSE but hurts overall |

Small result JSON files are included under:

```text
results/
```

Large files such as model weights, `pred.json`, `gt.json`, tensorboard logs, and training outputs are intentionally not included.

## Other Ablations Included

These are included for completeness but are not the recommended integration path.

### GRL domain adaptation

Files:

```text
apgcc_patch/models/APGCC.py
apgcc_patch/models/__init__.py
apgcc_patch/configs/CoNIC_finetune_grl.yml
apgcc_patch/configs/CoNIC_finetune_stain_grl.yml
apgcc_patch/configs/CoNIC_finetune_stain_grl_warmup.yml
```

Conclusion: GRL is theoretically reasonable for stain/domain robustness, but CoNIC source domains also encode morphology and density differences. Direct domain-invariant learning reduced recall or increased MSE, so it is kept as ablation only.

### Hard-density/domain reweight training

Files:

```text
apgcc_patch/engine.py
apgcc_patch/configs/CoNIC_finetune_stain_hard_density.yml
```

Conclusion: it helped hard domains such as `glas` and `pannuke` in MSE, but worsened the overall MAE/F1. It is useful as analysis code, not the final method.

## Integration Advice For Teammates

For another improved APGCC model, the cleanest integration is:

1. Keep your model architecture changes.
2. Copy `pathology_augment` and the `PATHOLOGY_AUG/STAIN_JITTER/BLUR_NOISE` config fields.
3. Train your improved model with stain augmentation.
4. Run `eval_domain_threshold.py --select-metric mae` on your model output.
5. Report the normal fixed threshold result and the DomainTh-MAE calibrated result.

Do not tune thresholds on `test.list`; use `val.list` only, then evaluate once on `test.list`.
