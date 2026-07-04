# Shared Dataset Splits

这些文件用于小组内统一 train/val/test 划分，避免不同模型使用不同随机划分导致结果不可比。

## 当前冻结划分

当前 `data_conversion/splits/` 下导出的是 PET 已经使用过的划分，因此不会要求 PET 重新训练：

- `BCData`
  - 使用官方 `train / validation / test`
  - `validation` 在 PET 中记为 `val`
- `CoNIC`
  - 使用当前 PET 实验的固定随机划分
  - `train`: 3985 patches
  - `val`: 996 patches
  - 当前没有单独 test；如果用于最终表格，建议把这份 `val` 统一作为 test/validation set，并在报告中说明
- `MoNuSeg`
  - 使用当前 PET 实验的固定随机 8:2 划分
  - 注意：如果后续改用官方 test set，严格来说需要所有方法统一切换并重新评估/训练

每个数据集目录中：

- `*_ids.txt`：每行一个样本 id，最适合给其他模型复用
- `*.csv`：PET 格式的 image/points 对应关系
- `summary.json`：样本数和 SHA1，用来检查大家拿到的是同一份 split

## 给其他同学怎么用

优先给他们 `*_ids.txt`。

例如 CoNIC：

```text
data_conversion/splits/CoNIC/train_ids.txt
data_conversion/splits/CoNIC/val_ids.txt
```

每一行是 `images.npy / labels.npy` 的 patch index。

例如 MoNuSeg：

```text
data_conversion/splits/MoNuSeg/train_ids.txt
data_conversion/splits/MoNuSeg/val_ids.txt
```

每一行是病例图像的 stem，例如 `TCGA-18-5592-01Z-00-DX1`。

## 如果要重新生成 CoNIC 70/15/15

这会改变当前 PET 实验协议，不建议在不重训的情况下替换当前结果。若全组决定统一新划分，可运行：

```bash
python data_conversion/export_splits.py make-conic --src_root path/to/CoNIC --out_dir splits_new --ratios 0.7,0.15,0.15 --seed 42
```
