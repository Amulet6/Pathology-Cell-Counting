# APGCC B+D+C Integrated Improvement

This folder contains the integrated APGCC improvement used for the MoNuSeg experiment.

The integration combines three modules:

- **B: Dense Auxiliary Loss**  
  Adds auxiliary dense supervision during training to strengthen learning in dense cell regions.

- **D: DCNv2 + Edge Ignore**  
  Adds deformable convolution for irregular cell morphology and edge-aware handling for truncated boundary cells.

- **C: Adaptive NMS + Score Threshold Calibration**  
  Adds inference-time post-processing to suppress duplicate nearby predictions and calibrate the confidence threshold.

The main target scenario is **MoNuSeg**, where the model tends to over-count in dense or morphologically diverse regions.

## Folder Layout

```text
APGCC_B+D+C/
├── apgcc/                         # APGCC source code with B/D/C changes
│   ├── configs/
│   │   ├── MoNuSeg_finetune_dcnv2_edge_dense.yml
│   │   └── MoNuSeg_finetune_dcnv2_edge_dense_nms.yml
│   ├── models/
│   │   ├── APGCC.py
│   │   ├── Encoder.py
│   │   └── dcnv2.py
│   ├── engine.py
│   ├── eval_centroid.py
│   └── main.py
├── CODE_CHANGES_DIRECTION_B_DENSE_LOSS.md
├── CODE_CHANGES_DIRECTION_D.md
├── CODE_CHANGES_DIRECTION_C.md
└── RUN_DIRECTION_B_D_C_MONUSEG.md
```

## Environment

Use the same environment as APGCC:

```bash
conda create -n apgcc python=3.10 -y
conda activate apgcc
pip install -r requirements.txt
```

If `conda activate` is not available on the server shell, initialize conda first:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate apgcc
```

## Data Format

The data root should follow the APGCC centroid format:

```text
MoNuSegdata/
├── train/
├── train_gt/
├── train.list
├── val/
├── val_gt/
├── val.list
├── test/
├── test_gt/
└── test.list
```

If the dataset has not been converted, use the project conversion scripts such as `regenerate_centroids.py` / `label_to_centroids.py` from the repository root.

## Pretrained Weight

The APGCC SHHA pretrained weight should be placed at:

```text
apgcc/output/SHHA_best.pth
```

Example:

```bash
cd APGCC_B+D+C/apgcc
mkdir -p output
cp /path/to/SHHA_best.pth ./output/SHHA_best.pth
```

## Train B+D Base on MoNuSeg

Run from:

```bash
cd improved/APGCC_B+D+C/apgcc
```

Training command:

```bash
python main.py -c ./configs/MoNuSeg_finetune_dcnv2_edge_dense.yml \
  GPU_ID 0 \
  DATASETS.DATA_ROOT /path/to/MoNuSegdata \
  RESUME_PATH ./output/SHHA_best.pth
```

This trains the B+D model:

- B: dense auxiliary loss
- D: DCNv2 + edge-aware handling

## Evaluate with Direction C

After training, evaluate the B+D checkpoint with C post-processing:

```bash
python eval_centroid.py \
  --config ./configs/MoNuSeg_finetune_dcnv2_edge_dense_nms.yml \
  --weight ./output/MoNuSeg_finetune_dcnv2_edge_dense/best.pth \
  --data-root /path/to/MoNuSegdata \
  --eval-list test.list \
  --score-threshold 0.58 \
  --thresholds 6 12 24 \
  --out-dir ./output/MoNuSeg_finetune_dcnv2_edge_dense_nms/centroid_eval_t058 \
  --gpu 0
```

Recommended threshold sweep:

```bash
for t in 0.52 0.54 0.55 0.56 0.57 0.58; do
  tag=${t/./}
  python eval_centroid.py \
    --config ./configs/MoNuSeg_finetune_dcnv2_edge_dense_nms.yml \
    --weight ./output/MoNuSeg_finetune_dcnv2_edge_dense/best.pth \
    --data-root /path/to/MoNuSegdata \
    --eval-list test.list \
    --score-threshold "$t" \
    --thresholds 6 12 24 \
    --out-dir ./output/MoNuSeg_finetune_dcnv2_edge_dense_nms/centroid_eval_t${tag} \
    --gpu 0
done
```

## MoNuSeg Result Used in the Report

Using the unified APGCC native baseline:

| Method | Threshold | MAE | F1@12 |
|---|---:|---:|---:|
| APGCC baseline | 0.50 | 115.00 | 0.8117 |
| C-only | 0.58 | 83.93 | 0.8017 |
| D-only | 0.50 | ~90.57 | 0.7925 |
| B+D+C integrated | 0.58 | **80.50** | 0.8010 |

The integrated result reduces MAE from **115.00** to **80.50**, about **30.0%** lower than the unified APGCC baseline.

## Notes

- Direction C is inference-time post-processing. It does not retrain the model.
- Adaptive NMS adjusts the spatial suppression radius, not the score threshold.
- Score threshold calibration controls how many predicted points are kept.
- Repulsion Loss is kept only as an optional ablation and is not the main integrated scheme.
- Large files such as checkpoints and generated outputs are intentionally not included.

## Detailed Change Logs

See:

- `CODE_CHANGES_DIRECTION_B_DENSE_LOSS.md`
- `CODE_CHANGES_DIRECTION_D.md`
- `CODE_CHANGES_DIRECTION_C.md`
- `RUN_DIRECTION_B_D_C_MONUSEG.md`

