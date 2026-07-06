# PET 病理细胞计数课程设计说明

本仓库基于 PET（Point-Query Quadtree）实现，针对病理细胞计数与定位任务做了数据适配、训练评估流程整理和统一指标输出。

## 上传范围

建议上传到 GitHub 的内容包括：

- `datasets/`：PET 数据集读取逻辑。
- `models/`：PET 模型、backbone、transformer 等代码。
- `util/`：训练与分布式辅助函数。
- `main.py`：训练入口。
- `eval.py`：测试/评估入口。
- `engine.py`：训练与评估循环。
- `data_conversion/convert_to_points.py`：BCData、CoNIC、MoNuSeg 等数据转换脚本。
- `data_conversion/export_splits.py`：统一划分导出脚本。
- `data_conversion/splits/`：小组统一划分文件。
- `baselines/pet/REPRODUCTION.md`、`baselines/pet/SERVER_COMMANDS.md`：复现实验说明与服务器运行命令。
- `data_conversion/coordinate_protocol.json`、`data_conversion/protocols/pet_augmentation.json`：坐标与增强协议。

不建议上传到 GitHub 的内容包括：

- `data/`：原始数据和转换后的图片/点标注。
- `outputs/`：训练输出、可视化和 checkpoint。
- `pretrained/`：预训练权重。
- `*.pth`：所有模型权重文件。

这些内容已经在 `.gitignore` 中排除。

## 数据格式

转换后的 PET 细胞数据格式如下：

```text
data/CoNIC_pet_split/
  train/
    images/*.png
    points/*.npy
  val/
    images/*.png
    points/*.npy
  test/
    images/*.png
    points/*.npy
  train.csv
  val.csv
  test.csv
```

其中 `.npy` 点标注为 `N x 2` 数组，内部顺序采用 PET 代码使用的：

```text
y, x
```

## 数据转换示例

BCData：

```bash
python data_conversion/convert_to_points.py --dataset bcdata \
  --src_root /path/to/BCData \
  --out_root data/BCData_pet_official
```

MoNuSeg：

```bash
python data_conversion/convert_to_points.py --dataset monuseg \
  --src_root /path/to/MoNuSeg \
  --out_root data/MoNuSeg_pet_split
```

CoNIC 统一划分：

```bash
python data_conversion/convert_to_points.py --dataset conic_split \
  --split_json /path/to/conic_split_seed19.json \
  --zip_root /path/to/cellvta_conic_release/data \
  --out_root data/CoNIC_pet_split
```

## 训练示例

```bash
python baselines/pet/main.py --dataset_file CoNIC \
  --data_path data/CoNIC_pet_split \
  --device cuda \
  --num_workers 2 \
  --batch_size 16 \
  --epochs 100 \
  --eval_freq 5 \
  --output_dir conic_unified_100ep_bs16 \
  --aug_mode unified
```

## 测试示例

```bash
python baselines/pet/eval.py --dataset_file CoNIC \
  --data_path data/CoNIC_pet_split \
  --device cuda \
  --num_workers 2 \
  --resume outputs/CoNIC/conic_unified_100ep_bs16/best_checkpoint.pth \
  --image_set test \
  --loc_radius 12
```

## 已汇总 PET 结果

主表采用 12px 定位匹配阈值。

| Dataset | Augmentation | MAE | MSE | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| BCData | native | 18.09 | 584.91 | 0.8259 | 0.8019 | 0.8137 |
| BCData | unified | 19.33 | 627.51 | 0.8062 | 0.7652 | 0.7852 |
| CoNIC | native | 37.20 | 3023.76 | 0.8583 | 0.5798 | 0.6921 |
| CoNIC | unified | 24.10 | 1176.24 | 0.8691 | 0.6910 | 0.7699 |
| MoNuSeg | native | 82.64 | 8435.79 | 0.6603 | 0.7744 | 0.7128 |
| MoNuSeg | unified | 116.71 | 14967.00 | 0.6747 | 0.8393 | 0.7481 |

## 说明

本仓库只保存可复现代码、划分文件和实验说明。由于数据集和模型权重体积较大，且可能涉及数据许可，未随仓库上传；复现实验时需要按上述命令在本地或服务器重新准备数据与权重。
