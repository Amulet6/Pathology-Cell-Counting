# Direction A: confidence and matching calibration

Direction A calibrates APGCC confidence and matching behavior for dense CoNIC
patches. The baseline-compatible implementation lives in `../../../baselines/apgcc/apgcc/`;
the same thresholding utilities are also used by the A+D+E integration under
`../../integrations/a_d_e/apgcc/`.

## Main components

- `MODEL.FOCAL_GAMMA`: optional softmax-focal modulation of the APGCC classification loss.
- `MATCHER.SET_COST_POINT`: tau in the Hungarian matching cost, used for geometry-vs-confidence calibration.
- `MODEL.EOS_COEF`: no-object class weight, used for confidence calibration sweeps.
- `phase0_analysis.py`: no-training diagnosis of coverage, missed GT score distribution, and density-threshold correlation.
- `scan_threshold.py`: validation-set global threshold sweep.
- `density_threshold.py`: density-adaptive test-time thresholding.
- `density_threshold_full.py`: density-adaptive thresholding with canonical centroid metrics.

## Report-facing configs

- Final native CoNIC setting: `../../../baselines/apgcc/apgcc/configs/CoNIC_A_native.yml`
- Controlled unified diagnostic setting: `../../../baselines/apgcc/apgcc/configs/CoNIC_A_unified_diagnostic.yml`
- K=8 negative-result configs:
  - `../../../baselines/apgcc/apgcc/configs/CoNIC_finetune_k8.yml`
  - `../../../baselines/apgcc/apgcc/configs/CoNIC_unified_K8.yml`
- APG self-reimplementation config: `../../../baselines/apgcc/apgcc/configs/CoNIC_apg.yml`

Thresholds must be selected on `val.list`; `test.list` is evaluated once with the
selected rule.
