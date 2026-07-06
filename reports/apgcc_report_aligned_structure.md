# APGCC report-aligned code map

This branch organizes the APGCC work around the report sections instead of raw
experiment history.

## 1. APGCC reproduction

- Code: `baselines/apgcc/apgcc/`
- Reproduction guide: `baselines/apgcc/REPRODUCE.md`
- Workflow and commands: `baselines/apgcc/WORKFLOW.md`
- Dataset converters: `baselines/apgcc/apgcc/datasets/prepare_*.py`
- Baseline configs:
  - `baselines/apgcc/apgcc/configs/BCData_{finetune,unified}.yml`
  - `baselines/apgcc/apgcc/configs/MoNuSeg_{finetune,unified}.yml`
  - `baselines/apgcc/apgcc/configs/CoNIC_{finetune,unified}.yml`

## 2. Direction A: confidence and matching calibration

Direction A is implemented as non-breaking options in the APGCC reproduction fork:

- `MODEL.FOCAL_GAMMA` in `baselines/apgcc/apgcc/config.py`
- softmax-focal classification loss in `baselines/apgcc/apgcc/models/APGCC.py`
- optional APG auxiliary reconstruction in `baselines/apgcc/apgcc/models/Decoder.py`
- subset breakdown support in `baselines/apgcc/apgcc/eval_centroid.py`
- diagnosis / calibration tools:
  - `baselines/apgcc/apgcc/phase0_analysis.py`
  - `baselines/apgcc/apgcc/scan_threshold.py`
  - `baselines/apgcc/apgcc/density_threshold.py`
  - `baselines/apgcc/apgcc/density_threshold_full.py`

Report-facing configs:

- Final native CoNIC setting: `baselines/apgcc/apgcc/configs/CoNIC_A_native.yml`
- Controlled unified diagnostic setting: `baselines/apgcc/apgcc/configs/CoNIC_A_unified_diagnostic.yml`
- K=8 negative-result configs:
  - `baselines/apgcc/apgcc/configs/CoNIC_finetune_k8.yml`
  - `baselines/apgcc/apgcc/configs/CoNIC_unified_K8.yml`
- APG self-reimplementation config: `baselines/apgcc/apgcc/configs/CoNIC_apg.yml`

Condensed Direction-A notes are in `progress.md`; report-ready tables and figures are
under `baogao/`.

## 3. Direction E and A + D + E integration

- Direction E minimal integration guide: `apgcc_improvements/directions/e_stain_domain_calibration/MINIMAL_INTEGRATION.md`
- A+D+E code: `apgcc_improvements/integrations/a_d_e/apgcc/`
- Integration guide: `apgcc_improvements/integrations/a_d_e/INTEGRATION_ADE.md`
- Result summary: `apgcc_improvements/integrations/a_d_e/RESULTS_ADE.md`
- Main CoNIC config: `apgcc_improvements/integrations/a_d_e/apgcc/configs/CoNIC_DE_stain_plain.yml`
- Combined A+D+E configs:
  - `apgcc_improvements/integrations/a_d_e/apgcc/configs/CoNIC_finetune_dcnv2_edge_stain.yml`
  - `apgcc_improvements/integrations/a_d_e/apgcc/configs/CoNIC_finetune_dcnv2_edge_stain_eos0.25.yml`
  - `apgcc_improvements/integrations/a_d_e/apgcc/configs/BCData_finetune_dcnv2_edge_stain.yml`
  - `apgcc_improvements/integrations/a_d_e/apgcc/configs/MoNuSeg_finetune_dcnv2_edge_stain.yml`

The adopted ADE report result is D+E clean training plus A density-adaptive
post-processing. Large local training outputs and weights are intentionally ignored.
