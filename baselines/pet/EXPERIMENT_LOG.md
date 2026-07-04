# PET Experiment Log

记录 PET 在各数据集上的训练配置、评估结果和备注，方便最后汇总报告。

## 统一说明

- 模型：PET
- Backbone：VGG16-BN
- 输入 patch：256 x 256
- 定位指标：预测点与真实点一对一最近邻匹配
- 当前定位半径：24 pixels
- 主要指标：MAE、MSE、Precision、Recall、F1-score

## BCData

### Run: `bcdata_100ep_bs16`

| 项目 | 内容 |
|---|---|
| 训练平台 | AutoDL RTX 4090D |
| 数据路径 | `data/BCData_pet` |
| batch size | 16 |
| epochs | 100 |
| eval freq | 10 |
| 输出目录 | `outputs/BCData/bcdata_100ep_bs16` |
| checkpoint | `best_checkpoint.pth` |
| best epoch | 50 |

| 指标 | 结果 |
|---|---:|
| MAE | 19.8797 |
| MSE | 26.2807 |
| Precision | 0.8634 |
| Recall | 0.8551 |
| F1-score | 0.8528 |

备注：

- BCData 原始标注为 positive/negative 两类 H5 点标注。
- 当前实验将 positive 和 negative 合并为总细胞点进行计数与定位。
- 评估时 `--loc_radius 24`。
- 可视化文件名示例：`0_gt74_pred55.jpg`、`100_gt98_pred103.jpg`。

## MoNuSeg

### Run: `monuseg_100ep_bs4`

本地/服务器数据概况：

| split | 图像数 | 点数范围 | 平均点数 |
|---|---:|---:|---:|
| train | 30 | 294-1863 | 677.03 |
| val | 7 | 354-1076 | 547.00 |

| 项目 | 内容 |
|---|---|
| 训练平台 | AutoDL RTX 4090D |
| 数据路径 | `data/MoNuSeg_pet` |
| batch size | 4 |
| epochs | 100 |
| eval freq | 10 |
| 输出目录 | `outputs/MoNuSeg/monuseg_100ep_bs4` |
| checkpoint | `best_checkpoint.pth` |
| best epoch | 90 |

| 指标 | 结果 |
|---|---:|
| MAE | 107.0000 |
| MSE | 124.7346 |
| Precision | 0.7873 |
| Recall | 0.6927 |
| F1-score | 0.7215 |

备注：

- MoNuSeg 原始标注为 XML 多边形实例标注。
- 当前实验从每个实例多边形提取质心，作为 PET 的点监督标签。
- 验证集图像数较少，且单图细胞数较高，因此 MAE 波动会比 BCData/CoNIC 明显。

## CoNIC

### Run: `conic_100ep_bs24`

| 项目 | 内容 |
|---|---|
| 训练平台 | AutoDL RTX 4090D |
| 数据路径 | `data/CoNIC_pet` |
| batch size | 24 |
| epochs | 100 |
| eval freq | 10 |
| 输出目录 | `outputs/CoNIC/conic_100ep_bs24` |
| checkpoint | `best_checkpoint.pth` |
| best epoch | 30 |

| 指标 | 结果 |
|---|---:|
| MAE | 7.0733 |
| MSE | 9.5996 |
| Precision | 0.8414 |
| Recall | 0.8743 |
| F1-score | 0.8552 |

备注：

- CoNIC 原始数据为 `images.npy` 和 `labels.npy`。
- 当前实验使用 `labels[..., 0]` 实例 ID 通道提取细胞中心点，只做总细胞计数与定位。
- 训练时跳过空标注 patch，验证时保留。
