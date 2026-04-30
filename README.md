# Pathology Image Cell Counting (病理图像细胞计数)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目简介 (Introduction)

病理图像分析是临床诊断的基础，而精准的细胞计数是其核心任务。本项目旨在解决病理图像中细胞形态极不规则、密集堆叠重叠、以及不同实验室染色工艺差异导致的视觉域偏移等挑战。

基于真实公开的病理图像数据集，本项目实现了多种先进的细胞计数与定位估计方法，包括密度图回归、点监督定位匹配以及实例分割，并对模型在复杂病理环境下的高效性与精准性进行了综合评估与改进。


## 项目目标 (Objectives)

1.  **多方法复现与对比**：在三个公开数据集上复现并对比至少三种不同类型的细胞计数方法：
    *   **密度图回归 (Density Map Regression)**: 如 Steerer [4]
    *   **点监督定位匹配 (Point Supervision)**: 如 PET [5]
    *   **实例分割 (Instance Segmentation)**: 如 HoVer-Net [6]
2.  **多维度评估**：
    *   **宏观定量分析**：MAE, MSE
    *   **微观定位分析**：Precision, Recall, F1-score
    *   **效率指标**：模型参数量、计算量 (FLOPs)、推理时间
3.  **模型改进与可视化**：基于综合效果最好的模型，针对细胞形态不规则、密集堆叠导致边界粘连等挑战进行改进，并输出可视化的预测位置/密度分布（散点图/热力图）。

## 数据集 (Datasets)

本项目使用以下三个公开病理细胞切片数据集：

| 数据集 | 全称 | 特点 | 引用 |
| :--- | :--- | :--- | :--- |
| **BCData** | Breast Cancer Cell Dataset | 专为点标注设计，包含大量严重重叠和形态不规则的乳腺癌细胞图像。 | [1] |
| **CoNIC** | Colon Nuclei Identification and Counting | 结肠核识别和计数挑战赛数据，包含极其复杂的细胞形态和多类别细胞标签。 | [2] |
| **MoNuSeg** | Multi-organ Nucleus Segmentation | 多器官细胞核分割与计数，涵盖多种器官的不同组织形态，染色差异极大。 | [3] |

## 项目结构 (project structure)

<details>
<summary>点击展开项目结构</summary>
```bash
Pathology-Cell-Counting/
├── README.md                    # 项目说明文档（最重要！）
├── requirements.txt             # Python依赖包列表
├── environment.yml             # Conda环境配置（可选）
├── .gitignore                  # Git忽略文件配置
├── LICENSE                     # 开源许可证（MIT/Apache 2.0）
│
├── configs/                    # 配置文件目录
│   ├── config.py              # 主配置文件（路径、超参数等）
│   ├── dataset_config.py      # 数据集配置
│   └── model_config.py        # 模型配置
│
├── datasets/                   # 数据集相关
│   ├── __init__.py
│   ├── bcdata_dataset.py      # BCData数据集加载器
│   ├── conic_dataset.py       # CoNIC数据集加载器
│   ├── monuseg_dataset.py     # MoNuSeg数据集加载器
│   └── transforms.py          # 数据增强和预处理
│
├── models/                     # 模型定义
│   ├── __init__.py
│   ├── hovernet.py            # HoVer-Net实现
│   ├── steerer.py             # STEERER实现
│   ├── pet.py                 # PET实现
│   ├── density_map_net.py     # 密度图回归方法
│   └── losses.py              # 损失函数定义
│
├── utils/                      # 工具函数
│   ├── __init__.py
│   ├── metrics.py             # 评估指标（MAE, MSE, Precision等）
│   ├── visualization.py       # 可视化函数
│   ├── logger.py              # 日志记录
│   └── misc.py                # 其他辅助函数
│
├── train/                      # 训练相关
│   ├── train.py               # 主训练脚本
│   ├── trainer.py             # Trainer类
│   └── train_scheduler.py     # 学习率调度
│
├── eval/                       # 测试/评估
│   ├── evaluate.py            # 评估脚本
│   └── test.py                # 测试脚本
│
├── logs/                       # 训练日志（.gitignore忽略）
│   ├── train_log.txt
│   └── tensorboard_logs/
│
├── checkpoints/                # 模型权重（.gitignore忽略）
│   ├── hovernet_best.pth
│   ├── steerer_best.pth
│   └── pet_best.pth
│
├── results/                    # 实验结果（.gitignore忽略）
│   ├── predictions/
│   ├── visualizations/
│   └── metrics_summary.json
│
└── docs/                       # 文档
    ├── dataset_intro.md       # 数据集介绍
    ├── methods.md             # 方法说明
    └── progress_report.md     # 进度报告
</details>details>

## 方法 (Methods)

本项目考虑实现以下核心算法（下为举例）：

### 1. HoVer-Net (Instance Segmentation)
同时实现细胞核的分割与分类。通过水平与垂直距离图（Horizontal and Vertical maps）解决实例粘连问题。
*   **Reference**: Graham et al., Medical Image Analysis, 2019 [6]

### 2. STEERER (Density Map Regression)
通过选择性继承学习（Selective Inheritance Learning）解决计数和定位中的尺度变化问题。
*   **Reference**: Han et al., ICCV 2023 [4]

### 3. PET (Point-query Quadtree)
基于点查询四叉树的人群计数、定位方法，适用于密集细胞场景。
*   **Reference**: Liu et al., ICCV 2023 [5]

## 参考文献 (references)
[1] Huang, Z., et al. (2020). Bc A large-scale dataset and benchmark for cell detection and counting. MICCAI.

[2] Graham, S., et al. (2024). CoNIC Challenge: Pushing the frontiers of nuclear detection, segmentation, classification and counting. Medical Image Analysis.

[3] Kumar, N., et al. (2019). A multi-organ nucleus segmentation challenge. IEEE TMI.

[4] Graham, S., et al. (2019). HoVer-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images. Medical Image Analysis.

[5] Liu, C., et al. (2023). Point-Query Quadtree for Crowd Counting, Localization, and More. ICCV.

[6] Han, T., et al. (2023). STEERER: Resolving scale variations for counting and localization via selective inheritance learning. ICCV.

## 快速开始 (Quick Start)

### 环境配置 (Installation)

```bash
# 克隆仓库
git clone https://github.com/YourUsername/Pathology-Cell-Counting.git
cd Pathology-Cell-Counting

# 创建虚拟环境
conda create -n path_cell python=3.8 （或许需要统一版本，避免环境不兼容）
conda activate path_cell

# 安装依赖
pip install -r requirements.txt
