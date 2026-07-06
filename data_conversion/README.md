# Data conversion

This is the canonical conversion module for BCData, CoNIC, and MoNuSeg. It
normalizes raw annotations to image-level center-point labels and exports shared
dataset splits.

## Canonical output

```text
<output>/
├── train/
│   ├── images/<sample>.png
│   └── points/<sample>.npy
├── val/
│   ├── images/<sample>.png
│   └── points/<sample>.npy
├── test/
│   ├── images/<sample>.png
│   └── points/<sample>.npy
└── {train,val,test}.csv
```

Each point file is an `N x 2` float array in `(y, x)` order. See
`coordinate_protocol.json` for the cross-method convention.

## Commands

```bash
python data_conversion/convert_to_points.py --dataset bcdata \
  --src_root /path/to/BCData --out_root data/BCData_points

python data_conversion/convert_to_points.py --dataset conic \
  --src_root /path/to/CoNIC --out_root data/CoNIC_points

python data_conversion/convert_to_points.py --dataset monuseg \
  --src_root /path/to/MoNuSeg --out_root data/MoNuSeg_points
```

Baseline-specific adapters remain inside each baseline when their native tensor,
mask, or density-map format differs from this canonical representation.
