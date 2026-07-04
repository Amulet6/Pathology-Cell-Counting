# Baseline reproduction

This module contains the training, inference, dataset-adapter, and result-summary
code for the five reproduced baselines.

| Method | Directory | Main task |
| --- | --- | --- |
| PET | `pet/` | Point-supervised localization and counting |
| STEERER | `steerer/` | Density-regression counting and localization |
| APGCC | `apgcc/` | Point-supervised localization and counting |
| HoVer-Net | `hovernet/` | Instance segmentation converted to center points |
| CellVTA | `cellvta/` | Instance segmentation converted to center points |

Each baseline keeps its own environment and commands because their framework and
CUDA requirements are not mutually compatible. Shared data conversion, split,
coordinate, and evaluation conventions live in `../data_conversion/` and
`../evaluation/`.

STEERER is preserved exactly as the `steerer-dev` integration overlay. Its
training scripts import upstream `lib/` and `lib_cls/` packages that were not
tracked on that branch; see `steerer/README.md` before attempting reproduction.
