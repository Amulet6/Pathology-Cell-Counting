# HoVer-Net 基线方法

基于 HoVer-Net 在三病理细胞分割/计数数据集上的完整复现与评估。CoNSeP/MoNuSeg 使用 HoVerNet (original mode, 270→80)，CoNIC 使用 HoVerNetExt (ResNet50 + padded decoder, fast mode, 256→256)。

**2026-06-16 更新**：三阶段全完成 + overlap4x 失败分析 + 三模型 Fold1 全维度对比。

---

## 1. 快速状态

| 数据集 | 分支 | 核心分割指标 | 点匹配 F1@12 | 状态 |
|------|------|:---:|:---:|:---:|
| **CoNSeP** | official | Dice 0.843, AJI 0.529, F1_d 0.739 | ⏳ 待跑 | ✅ |
| **MoNuSeg** | official | Dice 0.803, AJI 0.637, PQ 0.624, F1_d 0.859 | 0.860 | ✅ |
| **CoNIC** | conic_branch | mPQ⁺ 0.459, R² 0.568 | 0.704 | ✅ |

> **⚠️ CoNIC overlap4x 实验失败** — HoVer-Net 不适合用 CellViT 预包装的 4× overlap 增强数据。根因：HV Map 空间扭曲 + 无几何增强 → 捷径学习。详见根因分析文档。

---

## 2. 目录结构

```
baselines/hovernet/
├── README.md                                   # 本文件
├── src/                                        # 工具脚本
│   ├── label_to_centroids(1).py                # 实例 mask → 点坐标（跨方法统一格式）
│   ├── centroid_eval.py                        # 统一点匹配评估（Hungarian 双边匹配）
│   ├── gen_predictions_json.py                 # 生成 predictions.json
│   ├── benchmark_hovernet_efficiency.py         # HoVerNet 效率测量（FLOPs/latency/FPS）
│   ├── benchmark_cellvta_efficiency.py          # CellViT 效率测量
│   ├── preprocess_monuseg.py                   # MoNuSeg XML 标注 → .mat 实例 mask
│   ├── extract_patches_monuseg.py              # MoNuSeg 全图 → patches
│   └── convert_overlap4x.py                    # overlap4x PNG → .npy 格式转换
├── official/                                   # master 分支（CoNSeP / MoNuSeg）
│   ├── models/hovernet/                        # HoVerNet (original mode: 270→80)
│   ├── dataloader/train_loader.py              # FileLoader（官方增强）
│   ├── dataloader/train_loader_unified.py      # FileLoader（统一增强: RandomCrop）
│   ├── run_train.py                            # 训练入口（官方增强）
│   ├── run_train_unified.py                    # 训练入口（统一增强）
│   ├── run_infer.py                            # 全图推理 → .mat
│   ├── compute_stats.py                        # 实例分割指标：Dice/AJI/PQ/F1
│   ├── config.py / config_unified.py / dataset.py
│   ├── extract_patches.py                      # 补丁提取
│   ├── infer/wsi.py / infer/tile.py            # 推理引擎
│   └── metrics/ / misc/ / run_utils/           # 辅助模块
├── conic_branch/                               # conic 分支（CoNIC）
│   ├── models/hovernet/                        # HoVerNetExt (ResNet50, fast mode, 256→256)
│   │   ├── net_desc.py / net_utils.py          # 网络定义
│   │   ├── opt.py / run_desc.py                # 训练配置 / 训练逻辑
│   │   ├── targets.py                          # HV Map + NP + TP 目标生成
│   │   ├── post_proc.py                        # 后处理（watershed）
│   │   └── utils.py                            # 工具函数
│   ├── dataloader/train_loader.py              # DataLoader（官方增强）
│   ├── dataloader/train_loader_unified.py      # DataLoader（统一增强: Scale 0.8-1.2）
│   ├── run.py                                  # CoNIC 训练入口（官方增强）
│   ├── run_unified.py                          # CoNIC 训练入口（统一增强）
│   ├── run_overlap.py                          # overlap4x R1 训练（官方增强）
│   ├── run_unified_overlap.py                  # overlap4x R2 训练（统一增强）
│   ├── run_train.py                            # 训练引擎
│   ├── gen_official_preds.py                   # 官方基线预测生成（原始 .npy 数据）
│   ├── gen_overlap_preds.py                    # overlap4x 预测生成（Fold1 PNG 推理）
│   ├── infer_test.py                           # 测试集推理脚本
│   ├── generate_split.py                       # 划分生成
│   ├── conic_eval/compute_stats.py             # CoNIC 官方评估 (mPQ⁺ + R²)
│   ├── run_utils/engine.py                     # 训练引擎
│   └── misc/ / metrics/                        # CoNIC 辅助模块
```

---

## 3. 环境与预训练权重

```bash
conda activate hovernet    # Python 3.8, PyTorch 2.4.1, CUDA 12.1, RTX 4090 D (24GB)
```

### 预训练权重（需自行下载）

| 分支 | 权重文件 | 来源 |
|------|------|------|
| official (CoNSeP/MoNuSeg) | `pretrained/ImageNet-ResNet50-Preact_pytorch.tar` | [Google Drive](https://drive.google.com/file/d/1KntZge40tAHgyXmHYVqZZ5d2p_4Qr2l5/view) |
| conic_branch (CoNIC) | `exp_output/local/[ImageNet]resnet50-0676ba61.pth` | torchvision (`IMAGENET1K_V1`) |

---

## 4. 数据集

| 数据集 | 划分 | 图像尺寸 | 标注格式 | 细胞类型 |
|------|:---:|:---:|------|:---:|
| **CoNSeP** | 官方固定 27/14 | 1000×1000 | .mat (inst_map + type_map) | 4 类 |
| **MoNuSeg** | 队友统一 split JSON (30/7/14) | 1000×1000 | XML 多边形 → inst_map | 无 |
| **CoNIC** | 自划 stratified split (3503/679/799) | 256×256 | .npy [inst_map, type_map] | 6 类 + 背景 |

> CoNIC 的 6 种细胞类型：Neutrophil(0), Epithelial(1), Lymphocyte(2), Plasma(3), Eosinophil(4), Connective(5)。

---

## 5. 训练

### 启动命令

```bash
# CoNSeP / MoNuSeg (official 分支)
cd baselines/hovernet/official
python run_train.py --gpu=0          # 官方增强
python run_train_unified.py --gpu=0  # 统一增强 (RandomCrop)

# CoNIC (conic_branch)
cd baselines/hovernet/conic_branch
python run.py --gpu=0                # 官方增强 (CenterCrop + Flip)
python run_unified.py --gpu=0        # 统一增强 (Scale 0.8-1.2)
```

### 训练机制

两阶段训练（Phase 0 → Phase 1）：

| 阶段 | 说明 | official batch | conic batch | epochs |
|:---|:---|:---|:---|:---|
| Phase 0 | 冻结 backbone，仅训练 decoder | 16/16 | 6/6 | 50 |
| Phase 1 | 全模型微调 | 4/8 | 6/6 | 50 |

### 损失函数

- **np_loss**：核概率图 = BCE + Dice
- **hv_loss**：水平/垂直距离图 = MSE + MSGE
- **tp_loss**：细胞类型分类 = BCE + Dice（仅 CoNIC / CoNSeP）

### 数据增强

| 分支 | 空间增强 | 像素增强 |
|------|------|------|
| official (CoNSeP/MoNuSeg) | Affine(缩放/旋转/平移/剪切) + CenterCrop + Fliplr + Flipud | 模糊三选一 + 颜色扰动 |
| official 统一增强 | RandomCrop(替代 CenterCrop) | 同上 |
| conic_branch (CoNIC) | CenterCrop + Fliplr + Flipud | 同上 |
| conic_branch 统一增强 | Scale(0.8-1.2) + CenterCrop + Fliplr + Flipud | 同上 |

---

## 6. 评估

### 6.1 分割评估（模型质量验证）

```bash
# CoNSeP / MoNuSeg: 全图推理 → 实例评估
cd baselines/hovernet/official
python run_infer.py --gpu=0 --model_path=<ckpt> --model_mode=original
python compute_stats.py --mode=instance --pred_dir=<out/mat/> --true_dir=<GT>

# CoNIC: 生成预测 → 官方评估
cd baselines/hovernet/conic_branch
python gen_official_preds.py
cd conic_eval
python compute_stats.py --mode=seg_class --pred=<seg_preds.npy> --true=<gt_seg.npy>
python compute_stats.py --mode=regression --pred=<reg_preds.csv> --true=<gt_reg.csv>
```

### 6.2 点匹配计数评估（跨方法统一）

```bash
# 1. 从 inst_map 提取质心 → predictions.json
python baselines/hovernet/src/label_to_centroids(1).py <dataset> ...

# 2. 匈牙利双边匹配评估 @ 6/12/24px
python baselines/hovernet/src/centroid_eval.py --gt gt.json --pred pred.json --thresholds 6 12 24
```

### 6.3 效率测量

```bash
python baselines/hovernet/src/benchmark_hovernet_efficiency.py \
    --ckpt <checkpoint.tar> --num-types 7 --output-dir results/efficiency/
```

### 6.4 评估维度总览

| 维度 | 指标 | 评估脚本 | 用途 |
|:---|:---|:---|:---|
| **分割** | Dice, AJI, PQ, mPQ⁺, R² | `compute_stats.py` | 模型质量验证 |
| **点匹配** | MAE, MSE, Precision, Recall, F1@6/12/24px | `centroid_eval.py` | 跨方法统一对比 |
| **效率** | Params, FLOPs, Latency, FPS | `benchmark_hovernet_efficiency.py` | 推理性能 |

---

## 7. 实验结果

### 7.1 CoNSeP

| 来源 | Dice | AJI | PQ | F1_d |
|------|:--:|:--:|:--:|:--:|
| **我们** | **0.843** | 0.529 | 0.492 | **0.739** |
| 官方 PyTorch README | 0.850 | 0.601 | 0.546 | 0.756 |
| 差距 | -0.86% | -12.0% | -9.9% | -2.2% |

> Dice 在 ±2% 内。AJI/PQ 偏低因 RTX 4090 强制 PyTorch 2.4（官方用 1.6）。

### 7.2 MoNuSeg

| 分组 | Dice | AJI | PQ | F1_d |
|------|:--:|:--:|:--:|:--:|
| Seen organs (11 张) | **0.819** | **0.661** | **0.641** | **0.867** |
| Unseen organs (3 张) | 0.744 | 0.549 | 0.560 | 0.798 |
| **All (14 张)** | **0.803** | **0.637** | **0.624** | **0.859** |

> AJI 0.637 超过所有已发表 HoVer-Net MoNuSeg 基线（0.59-0.62）。

### 7.3 CoNIC

| 任务 | 指标 | 值 | 参考 |
|------|------|:--:|------|
| Task 1 (分割+分类) | mPQ⁺ | **0.459** | 已知投稿方案 0.39-0.41 |
| Task 1 | PQ (binary) | 0.521 | — |
| Task 2 (计数) | multi R² | **0.568** | 官方 HoVerNet 基线 ~0.55 |
| 点匹配 | F1@12px | **0.704** | — |

### 7.4 ⚠️ CoNIC overlap4x — 失败记录

在 CellViT 预包装的 4× overlap 数据上训练后，Fold 1 测试集性能崩溃：

| 模型 | mPQ⁺ | R² | F1@12px |
|------|:---:|:---:|:---:|
| 官方预训练（原始数据） | **0.430** | **0.594** | **0.700** |
| overlap R1（官方增强） | 0.119 | -1.976 | 0.536 |
| overlap R2（统一增强） | 0.219 | -0.279 | 0.652 |

> 根因：密集滑窗造成 ~35% 细胞边界截断 → HV Map 回归目标空间扭曲 → HoVer-Net（无几何增强）走捷径死记截断痕迹 → 测试分布断裂时彻底瘫痪。

---

## 8. Checkpoint 管理

训练产物（`logs/`、`exp_output/`、`pretrained/`）不上传 GitHub，本地备份路径见项目内部文档。

---

## 引用

- Graham, S., et al. (2019). HoVer-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images. *Medical Image Analysis*, 58, 101563.
- Graham, S., et al. (2024). CoNIC Challenge: Pushing the frontiers of nuclear detection, segmentation, classification and counting. *Medical Image Analysis*, 92, 103047.
