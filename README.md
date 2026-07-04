# Pathology Cell Counting

面向 BCData、CoNIC 和 MoNuSeg 的病理细胞计数与定位项目。仓库按报告中的四个功能模块组织：数据转换、Baseline 复现、统一评价和 APGCC 改进。

## 仓库结构

```text
Pathology-Cell-Counting/
├── data_conversion/                 # 原始标注转统一中心点、固定数据划分
│   ├── convert_to_points.py
│   ├── export_splits.py
│   ├── coordinate_protocol.json
│   ├── protocols/
│   └── splits/{BCData,CoNIC,MoNuSeg}/
├── baselines/                       # 五种方法的训练、推理与结果整理
│   ├── pet/
│   ├── steerer/
│   ├── apgcc/
│   ├── hovernet/
│   └── cellvta/
├── evaluation/                      # 跨方法统一中心点评价
│   ├── centroid_eval.py
│   └── formats/predictions_json.md
├── apgcc_improvements/              # APGCC 方向 A--E 与组合实验
│   ├── directions/
│   │   ├── a_threshold_calibration/
│   │   ├── b_dense_auxiliary_supervision/
│   │   ├── c_adaptive_nms/
│   │   ├── d_dcnv2_edge_ignore/
│   │   └── e_stain_domain_calibration/
│   └── integrations/{a_d_e,b_d,b_d_c}/
├── docs/                             # 项目记录、阶段报告与论文资料
├── report/                           # 外部课程报告子模块
├── LICENSE
└── README.md
```

## 1. 数据转换模块

`data_conversion/convert_to_points.py` 将三种数据集统一为图像和 `N x 2` 中心点数组。内部坐标顺序为 `(y, x)`；跨方法 JSON 交换格式使用 `(x, y)`，详见 `data_conversion/coordinate_protocol.json`。

```bash
python data_conversion/convert_to_points.py --dataset bcdata \
  --src_root /path/to/BCData --out_root data/BCData_points

python data_conversion/convert_to_points.py --dataset conic \
  --src_root /path/to/CoNIC --out_root data/CoNIC_points

python data_conversion/convert_to_points.py --dataset monuseg \
  --src_root /path/to/MoNuSeg --out_root data/MoNuSeg_points
```

共享的训练、验证和测试划分位于 `data_conversion/splits/`。各模型需要特殊张量、实例图或密度图时，其适配器保留在对应 baseline 内。

## 2. Baseline 复现模块

| 方法 | 目录 | 方法类型 |
| --- | --- | --- |
| PET | `baselines/pet/` | 点监督定位匹配 |
| STEERER | `baselines/steerer/` | 密度图回归 |
| APGCC | `baselines/apgcc/` | 点监督定位匹配 |
| HoVer-Net | `baselines/hovernet/` | 实例分割 |
| CellVTA | `baselines/cellvta/` | 实例分割 |

各方法依赖的 PyTorch/CUDA 版本不同，请进入对应目录，按照该目录的 README 和 requirements/environment 文件创建独立环境。STEERER 与 HoVer-Net 的分支文件已归入统一 baseline 目录；其中 `steerer-dev` 是叠加包，运行前仍需补齐其 README 所列的上游 `lib/` 与 `lib_cls/` 核心依赖。

## 3. 统一评价模块

统一脚本基于预测中心点和 GT 中心点计算 MAE、MSE、RMSE、Precision、Recall 和 F1-score，并支持多个匹配距离阈值。

```bash
python evaluation/centroid_eval.py \
  --gt path/to/gt.json \
  --pred path/to/pred.json \
  --thresholds 6 12 24
```

输入格式见 `evaluation/formats/predictions_json.md`。所有 baseline 的最终横向对比应使用该脚本和相同阈值。

## 4. APGCC 改进模块

| 方向 | 内容 | 代码位置 |
| --- | --- | --- |
| A | 阈值扫描与密度自适应校准 | `apgcc_improvements/directions/a_threshold_calibration/` |
| B | 密集区域辅助监督 | `apgcc_improvements/directions/b_dense_auxiliary_supervision/` |
| C | Adaptive NMS 与置信度校准 | `apgcc_improvements/directions/c_adaptive_nms/` |
| D | DCNv2 与 Edge Ignore | `apgcc_improvements/directions/d_dcnv2_edge_ignore/` |
| E | Stain Aug 与域感知阈值校准 | `apgcc_improvements/directions/e_stain_domain_calibration/` |

可运行的组合版本位于 `apgcc_improvements/integrations/`：CoNIC 使用 A+D+E，MoNuSeg 使用 B+D+C，另保留 B+D 消融版本。

## 数据与模型文件

仓库只跟踪代码、小型评价结果、固定划分和复现文档。原始数据、转换后图像、checkpoint、预训练权重、TensorBoard 日志及大规模预测结果由 `.gitignore` 排除，需要在本地准备。

## License

项目级代码使用 [MIT License](LICENSE)。各 baseline 中的上游实现仍遵循其目录内的原始许可证。
