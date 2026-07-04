# APGCC improvements

This module maps the implementation directly to report Directions A--E.

| Direction | Implementation | Purpose |
| --- | --- | --- |
| A | `directions/a_threshold_calibration/` | Score and density-aware threshold calibration |
| B | `directions/b_dense_auxiliary_supervision/` | Dense auxiliary supervision |
| C | `directions/c_adaptive_nms/` | Adaptive NMS and score calibration |
| D | `directions/d_dcnv2_edge_ignore/` | DCNv2 and edge-ignore training |
| E | `directions/e_stain_domain_calibration/` | Stain augmentation and domain-aware calibration |

Runnable combined experiments are retained under `integrations/`:

- `a_d_e/`: CoNIC integration for threshold calibration, DCNv2/edge ignore,
  stain augmentation, and domain-aware calibration.
- `b_d/`: dense auxiliary supervision with DCNv2/edge ignore.
- `b_d_c/`: the preceding model plus Adaptive NMS for MoNuSeg.

Large checkpoints and generated training artifacts are intentionally excluded.
