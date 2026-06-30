# Run Direction B + D + C on MoNuSeg

Run from:

```bash
cd improved/APGCC_B+D+C/apgcc
```

## Train B+D base

```bash
python main.py -c ./configs/MoNuSeg_finetune_dcnv2_edge_dense.yml \
  GPU_ID 0 \
  DATASETS.DATA_ROOT /data1/llx/MoNuSegdata \
  RESUME_PATH ./output/SHHA_best.pth
```

## Evaluate with Direction C

Use the trained B+D checkpoint and enable Adaptive NMS in the evaluation config.

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

Threshold sweep:

```bash
for t in 0.52 0.54 0.55 0.56 0.57 0.58; do
  tag=${t/./}
  python eval_centroid.py \
    --config ./configs/MoNuSeg_finetune_dcnv2_edge_dense_nms.yml \
    --weight ./output/MoNuSeg_finetune_dcnv2_edge_dense/best.pth \
    --data-root /data1/llx/MoNuSegdata \
    --eval-list test.list \
    --score-threshold "$t" \
    --thresholds 6 12 24 \
    --out-dir ./output/MoNuSeg_finetune_dcnv2_edge_dense_nms/centroid_eval_t${tag} \
    --gpu 0
done
```

## Optional Repulsion Ablation

Use `MoNuSeg_finetune_dcnv2_edge_dense_repulsion.yml` only for ablation. It changes training loss and should not replace the main B+D+C post-processing result unless it improves MAE/MSE.

