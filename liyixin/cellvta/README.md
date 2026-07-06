# CellVTA 复现

CellVTA 在病理细胞计数数据集（BCData / MoNuSeg / CoNIC）上的复现实验交付包。

## 目录说明

| 目录 | 内容 |
|---|---|
| `code/` | CellVTA 源码、训练/推理配置 |
| `scripts/` | 数据转换、评测、效率 benchmark 脚本 |
| `share/` | 统一数据划分、增强协议、centroid 评估工具 |
| `logs/` | 关键训练/推理日志（CoNIC 官方复现、统一增强等） |
| `reports/` | 指标汇总、效率结果、阶段报告 |
| `manifest/` | 交付清单说明 |

## 主要结果

详见 [`reports/cellvta_results_summary.md`](reports/cellvta_results_summary.md)。

| 数据集 | 增强 | MAE | F1@12 |
|---|---|---:|---:|
| CoNIC | 官方 | 2.76 | 0.814 |
| CoNIC | unified | 2.72 | 0.804 |
| MoNuSeg | unified | 13.11 | 0.645 |

## 复现流程概览

1. 准备 CellVTA 环境与依赖（见 `code/repos/CellVTA/`）
2. 使用 `scripts/prepare_*.py` 转换数据集格式
3. 训练：`code/repos/CellVTA/cell_segmentation/run_cellvit.py`
4. 评测：`scripts/evaluate_cellvta_centroids.py`
5. 效率：`scripts/benchmark_cellvta_efficiency.py`

CoNIC 官方对齐复现可参考 `scripts/run_cellvta_conic_official.sh` 与 `logs/paper/CellVTA_CoNIC/`。

## 不包含

- 原始数据集
- 训练 checkpoint / 权重
- conda 环境
