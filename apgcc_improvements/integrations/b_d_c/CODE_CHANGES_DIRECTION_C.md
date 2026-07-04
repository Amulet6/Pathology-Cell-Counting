# Direction C: Adaptive NMS + Score Threshold Calibration

## Summary

Direction C adds a post-processing path for APGCC point predictions:

- score threshold filtering
- optional Direction D edge-aware filtering
- adaptive point NMS

It also keeps Repulsion Loss as an optional training ablation. Repulsion is disabled by default because earlier experiments showed weak counting benefit.

## Changed Files

### `apgcc/config.py`

Added:

```yaml
MODEL.REPULSION:
  ENABLED: false
  WEIGHT: 0.05
  RADIUS: 8.0
  GT_SAFE_DISTANCE: 6.0

TEST.ADAPTIVE_NMS:
  ENABLED: false
  DENSE_RADIUS: 4.0
  SPARSE_RADIUS: 8.0
  DENSE_KNN_DIST: 10.0
```

### `apgcc/engine.py`

Added:

- `adaptive_point_nms(points, scores, cfg)`
- `filter_prediction_points(...)`

The unified filter first applies the score threshold, then optional edge filtering, then optional adaptive NMS.

### `apgcc/eval_centroid.py`

Uses `filter_prediction_points()` so offline centroid evaluation can test C with:

```bash
python eval_centroid.py \
  --config ./configs/MoNuSeg_finetune_dcnv2_edge_dense_nms.yml \
  --weight ./output/MoNuSeg_finetune_dcnv2_edge_dense/best.pth \
  --data-root /data1/llx/MoNuSegdata \
  --eval-list test.list \
  --score-threshold 0.57 \
  --thresholds 6 12 24 \
  --out-dir ./output/MoNuSeg_finetune_dcnv2_edge_dense_nms/centroid_eval_t057 \
  --gpu 0
```

### `apgcc/models/APGCC.py`

Added optional `loss_repulsion`, controlled by `MODEL.REPULSION.ENABLED`.

### `apgcc/models/__init__.py`

Adds `loss_repulsion` to `weight_dict` only when enabled.

## Notes

Adaptive NMS adapts the spatial suppression radius, not the score threshold.
The score threshold is calibrated per dataset by sweep.

