# APGCC 方向 D：DCNv2 + Edge Ignore

## 改进概述

在 APGCC（VGG16-bn + IFI decoder）上实现两个改进：

| 改进 | 说明 | 新增文件 | 修改文件 |
|---|---|---|---|
| **DCNv2** | VGG16-bn conv5 最后 2 层替换为可变形卷积，增强对不规则细胞形态的建模能力 | `models/dcnv2.py` | `Encoder.py`, `APGCC.py`, `__init__.py`, `config.py` |
| **Edge Ignore** | 训练时对图像边缘 16px 带的 anchor 降低 loss 权重，减轻边缘截断细胞的假阳性 | — | `APGCC.py`, `engine.py`, `util/misc.py`, `config.py`, `dataset.py` |
| **区域分析** | 4×4 网格区域级误差诊断 | `analyze_region_error.py` | — |
| **统一评测** | 路径修复 + 完整指标输出 | — | `eval_centroid.py` |

参数增量：+0.25M（+1.4%）。向后兼容：所有新 feature 默认关闭，旧 checkpoint 可直接加载。

## 消融结果摘要（native 基线）

| 数据集 | 最优方案 | MAE Δ | F1@12px Δ |
|---|---|---|---|
| MoNuSeg（大图密集） | DCNv2 + Edge | **−21.2%** | −0.019 |
| BCData（稀疏近全图） | DCNv2 + Edge | **−5.3%** | −0.001 |
| CoNIC（小 patch） | Edge only | **−8.9%** | −0.003 |

三个数据集组合方案均优于 baseline。

## 快速开始

### 1. 环境
```bash
conda create -n apgcc python=3.10 -y && conda activate apgcc
pip install -r requirements.txt
```

### 2. 预训练权重

下载 APGCC 官方 SHHA 预训练权重（Google Drive）：

```bash
cd apgcc/output
# Google Drive 下载（约 68MB）
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pEvn5RrvmDqVJUDZ4c9-rCJcl2I7bRhu' -O SHHA_best.pth
cd ../..
```

> 此权重来自 [APGCC 官方仓库](https://github.com/AaronCIH/APGCC)，在 ShanghaiTech Part A 上预训练的 VGG16-bn + IFI 模型。

### 3. 准备数据
参考 `apgcc/datasets/prepare_*.py` 将数据转换为 APGCC 格式。

### 4. 修改配置
编辑对应 yml 中的 `GPU_ID` 和 `DATA_ROOT`。

### 5. 训练

```bash
cd apgcc

# Baseline 复现
python main.py -c ./configs/MoNuSeg_finetune.yml GPU_ID 0 DATASETS.DATA_ROOT /path/to/data

# DCNv2 消融
python main.py -c ./configs/MoNuSeg_finetune_dcnv2.yml GPU_ID 0 DATASETS.DATA_ROOT /path/to/data

# Edge Ignore 消融
python main.py -c ./configs/MoNuSeg_finetune_edge.yml GPU_ID 0 DATASETS.DATA_ROOT /path/to/data

# 组合消融
python main.py -c ./configs/MoNuSeg_finetune_dcnv2_edge.yml GPU_ID 0 DATASETS.DATA_ROOT /path/to/data

# ⚠️ BCData / CoNIC 必须加 EVAL_LIST，否则默认退化为 test.list
python main.py -c ./configs/BCData_finetune_dcnv2_edge.yml GPU_ID 0 \
  DATASETS.DATA_ROOT /path/to/data DATASETS.EVAL_LIST val.list
```

### 6. 评测
```bash
python eval_centroid.py \
  --config ./configs/MoNuSeg_finetune_dcnv2_edge.yml \
  --weight ./output/MoNuSeg_finetune_dcnv2_edge/best.pth \
  --data-root /path/to/data \
  --eval-list test.list \
  --score-threshold 0.5 \
  --thresholds 6 12 24 \
  --out-dir ./output/MoNuSeg_finetune_dcnv2_edge/centroid_eval \
  --gpu 0
```

### 7. 区域分析
```bash
python analyze_region_error.py \
  --gt ./output/MoNuSeg_finetune_dcnv2_edge/centroid_eval/gt.json \
  --pred ./output/MoNuSeg_finetune_dcnv2_edge/centroid_eval/pred_centroid_eval.json \
  --grid 4 4
```

## 使用决策树

```
CROP_SIZE ≥ 256 + UPPER_BOUNDER = −1 → DCNv2 + Edge（如 MoNuSeg）
CROP_SIZE ≥ 256 + UPPER_BOUNDER 有约束 → 仅 DCNv2
CROP_SIZE < 256 + UPPER_BOUNDER = −1 → 仅 Edge（如 CoNIC）
其他 → 两个都不建议
```

## 配置参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `DCNV2_EN` | `False` | 启用可变形卷积（conv5 最后 2 层） |
| `EDGE_IGNORE` | `False` | 启用训练期边缘忽略 |
| `EDGE_BAND` | `16` | 边缘带宽度（像素） |

在对应 yml 中设置即可，无需改代码。

## 报告

详细消融报告见项目 docs 目录：
- `docs/RESULTS_DIRECTION_D_MoNuSeg.md`
- `docs/RESULTS_DIRECTION_D_BCData.md`
- `docs/RESULTS_DIRECTION_D_CoNIC.md`
- `docs/RESULTS_DIRECTION_D_CROSS_DATASET.md`
- `docs/CODE_CHANGES_DIRECTION_D.md`

---

*基于 [APGCC (ECCV 2024)](https://github.com/AaronCIH/APGCC)*
