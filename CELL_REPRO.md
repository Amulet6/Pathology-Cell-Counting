# PET pathology-cell reproduction notes

## Environment

Use the separate conda environment created for this project:

```powershell
conda activate pet
```

The current machine has already passed a CUDA smoke test with:

- Python 3.8
- torch 1.12.1+cu113
- torchvision 0.13.1+cu113
- RTX 4060 Laptop GPU

## Unified data format

PET now supports `--dataset_file CELL`, `BCData`, `CoNIC`, and `MoNuSeg`.
All of them use the same point-label format:

```text
data/CoNIC_pet/
  train/
    images/*.png
    points/*.npy
  val/
    images/*.png
    points/*.npy
  train.csv
  val.csv
```

Each `.npy` file stores an `N x 2` array in PET's internal point order:

```text
y, x
```

## Convert datasets

CoNIC, when unpacked with `images.npy` and `labels.npy`:

```powershell
conda run -n pet python prepare_cell_dataset.py --dataset conic --src_root path\to\conic --out_root data\CoNIC_pet
```

MoNuSeg, when images and XML annotations are under one root:

```powershell
conda run -n pet python prepare_cell_dataset.py --dataset monuseg --src_root path\to\MoNuSeg --out_root data\MoNuSeg_pet
```

BCData, using the downloaded structure with `images/{train,validation,test}` and
`annotations/{train,validation,test}/{positive,negative}/*.h5`:

```powershell
conda run -n pet python prepare_cell_dataset.py --dataset bcdata --src_root ..\..\data_raw\BCData --out_root data\BCData_pet
```

The converter merges positive and negative points for total cell counting and swaps BCData's `x, y` HDF5 coordinates into PET's `y, x` order.

Generic point-folder data:

```text
raw_bcdata/
  images/*.png
  points/*.npy
```

Then:

```powershell
conda run -n pet python prepare_cell_dataset.py --dataset point_folders --src_root raw_bcdata --out_root data\BCData_pet
```

The point files for `point_folders` must already be in `y, x` order. If a downloaded annotation is in `x, y`, swap columns before conversion.

## Quick PET commands

Small local training run:

```powershell
conda run -n pet python main.py --dataset_file CoNIC --data_path data\CoNIC_pet --batch_size 2 --epochs 20 --eval_freq 5 --output_dir conic_debug
```

Evaluation with visualizations:

```powershell
conda run -n pet python eval.py --dataset_file CoNIC --data_path data\CoNIC_pet --resume outputs\CoNIC\conic_debug\best_checkpoint.pth --vis_dir vis_conic
```

The evaluation output now includes MAE, MSE, Precision, Recall, and F1. The localization threshold is currently 8 pixels in `engine.py`.
