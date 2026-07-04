# APGCC 方向 B：Dense Loss-Only 改进

在 APGCC（VGG16-bn + IFI decoder）基础上，新增 Count Anything 风格的 **dense auxiliary branch**，仅在训练阶段使用密集监督，测试阶段保持原有推理流程不变。

## 改动摘要

| 项目 | 说明 |
|---|---|
| 新增模块 | `dense_head`：从编码器 level-3 特征输出单通道 dense map |
| 新增损失 | `loss_dense`：sigmoid(dense_map) 与高斯密度 target 的 MSE |
| 关键参数 | `DENSE_AUX_EN=True`, `DENSE_AUX_LEVEL=3`, `DENSE_SIGMA=2.0`, `loss_dense=0.1` |
| 推理 | 测试时 dense 分支不参与，输出与 baseline 完全一致 |

代码改动详见 [`docs/CODE_CHANGES_DIRECTION_B_DENSE_LOSS.md`](docs/CODE_CHANGES_DIRECTION_B_DENSE_LOSS.md)。

## 环境

```bash
conda create -n apgcc python=3.8 -y
conda activate apgcc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install h5py
```

## 预训练权重

```bash
cd apgcc
mkdir -p output
wget --no-check-certificate \
  'https://docs.google.com/uc?export=download&id=1pEvn5RrvmDqVJUDZ4c9-rCJcl2I7bRhu' \
  -O ./output/SHHA_best.pth
```

## 训练（dense loss-only）

在 `apgcc/` 目录下运行，配置中已开启 `DENSE_AUX_EN: True`：

```bash
python main.py -c ./configs/BCData_finetune.yml  GPU_ID 0  DATASETS.DATA_ROOT /path/to/BCData
python main.py -c ./configs/MoNuSeg_finetune.yml GPU_ID 0  DATASETS.DATA_ROOT /path/to/MoNuSeg
python main.py -c ./configs/CoNIC_finetune.yml   GPU_ID 0  DATASETS.DATA_ROOT /path/to/CoNIC
```

## 评测

```bash
python eval_centroid.py \
  --config   ./configs/BCData_finetune.yml \
  --weight   ./output/BCData_finetune/best.pth \
  --data-root /path/to/BCData \
  --eval-list test.list \
  --score-threshold 0.5 \
  --thresholds 6 12 24 \
  --out-dir  ./output/BCData_finetune/centroid_eval \
  --gpu 0
```

## 实验结果

主表（F1@12px，native 增强）：

| 数据集 | 方法 | MAE | F1@12 | 计数偏差 |
|---|---|---:|---:|---|
| BCData | baseline | 18.27 | 0.8145 | −3.67% |
| BCData | + dense loss-only | **17.51** | **0.8247** | −3.16% |
| MoNuSeg | baseline | 111.71 | 0.8032 | +23.35% |
| MoNuSeg | + dense loss-only | **110.36** | **0.8101** | +23.07% |
| CoNIC | baseline | **11.99** | 0.7944 | **−3.08%** |
| CoNIC | + dense loss-only | 13.69 | **0.8005** | −7.10% |

完整分析与 fusion 消融见 [`docs/APGCC_direction_B_dense_loss_report.md`](docs/APGCC_direction_B_dense_loss_report.md)。

## 结果文件

`results/` 目录保存了各实验的日志、配置副本和 centroid 评测 JSON（不含 `.pth` 权重）：

- `results/BCData_finetune_dense_only/` — 方向 B 正式版本
- `results/MoNuSeg_finetune_dense_only/`
- `results/CoNIC_finetune_dense_only/`
- `results/BCData_finetune_dense_fusion/` — fusion v1 消融
- `results/BCData_finetune_dense_fusion_v2/` — fusion v2 消融
- `results/baseline/` — APGCC baseline 三数据集结果

## 文档

| 文件 | 说明 |
|---|---|
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | APGCC 三数据集完整复现流程 |
| [`docs/APGCC_direction_B_dense_loss_report.md`](docs/APGCC_direction_B_dense_loss_report.md) | 方向 B 实验报告 |
| [`docs/CODE_CHANGES_DIRECTION_B_DENSE_LOSS.md`](docs/CODE_CHANGES_DIRECTION_B_DENSE_LOSS.md) | 代码改动清单 |
| [`docs/report_direction_B_snippet.md`](docs/report_direction_B_snippet.md) | 课程报告 LaTeX 片段 |
| [`data_augmentation_protocol.md`](data_augmentation_protocol.md) | 统一增强协议 |
