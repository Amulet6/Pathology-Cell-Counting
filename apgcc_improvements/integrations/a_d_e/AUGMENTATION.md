# APGCC 数据增强策略（Data Augmentation）

> 代码来源：`apgcc/datasets/dataset.py`（核心逻辑）、`apgcc/datasets/build.py`（transform 与
> collate）、`apgcc/config.py` 与 `apgcc/configs/*.yml`（超参）。
> 本文档逐张图片追踪「原始图片 → 模型输入」的完整处理流程，并对比各数据集配置差异。

---

## 0. 关键超参（决定增强行为）

增强行为完全由 `DATALOADER` 配置驱动，默认值见 `config.py:76-81`：

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `AUGUMENTATION` | 启用的增强项（字符串列表） | `['Normalize', 'Crop', 'Flip']` |
| `CROP_SIZE` | 随机裁剪块边长（正方形） | `128` |
| `CROP_NUMBER` | 每张图裁剪的 patch 数量 | `4` |
| `UPPER_BOUNDER` | 图像长边上界，`-1` 表示不限制 | `-1` |
| `SOLVER.BATCH_SIZE` | 每个 batch 的图片数 | `8` |

- 只有当 `'Crop'` 在 `AUGUMENTATION` 中时才启用随机裁剪（`self.patch`）。
- 只有当 `'Flip'` 在 `AUGUMENTATION` 中时才启用随机水平翻转（`self.flip`）。
- `'Normalize'` 实际并不由这个列表控制——归一化恒定执行（见下文 transform），列表里写它只是标注。

---

## 1. 训练阶段：一张图片的完整处理链

以下对应 `ImageDataset.__getitem__`（`dataset.py:56-134`），按执行顺序：

### 步骤 1 — 读图与读点（`load_data`, dataset.py:136-151）
- `cv2.imread(img_path)` 读入 **BGR** 图 → `cv2.cvtColor(..., BGR2RGB)` → `PIL.Image`（**RGB**，原始分辨率，不缩放）。
- 从 `.txt` 标注逐行读取头部坐标 `(x, y)`，得到 `point` 数组，形状 `(N, 2)`。

### 步骤 2 — ToTensor + Normalize（`build.py:26-29`）
固定的 torchvision transform，**训练/测试都执行，无随机性**：
1. `ToTensor()`：`HWC uint8 [0,255]` → `CHW float [0,1]`，张量形状 `[3, H, W]`。
2. `Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])`：ImageNet 标准化。

此时 `img` 是 `[3, H, W]` 的归一化张量，`point` 仍是原图坐标系。

### 步骤 3 — 随机缩放 Random Scale（dataset.py:77-96）
- `min_size = min(H, W)`，`max_size = max(H, W)`。
- 缩放系数 `scale` 的取值范围**取决于 `UPPER_BOUNDER` 与图像大小**（这是各数据集差异的核心，见第 3 节）：

  | 条件 | scale 范围 | 效果 |
  |------|-----------|------|
  | `UPPER_BOUNDER != -1` 且 `max_size > UPPER_BOUNDER` | `[ub/max-0.1, ub/max]`，其中 `ub=UPPER_BOUNDER` | 把超大图压到接近上界（仅缩小） |
  | `UPPER_BOUNDER != -1` 且 `max_size ≤ UPPER_BOUNDER` | `[0.7, 1.3]` | 允许**放大**到 1.3× |
  | `UPPER_BOUNDER == -1`（默认） | `[0.7, 1.0]` | 仅缩小或不变，**不放大** |

- **只有当 `scale * min_size > CROP_SIZE` 时才真正缩放**（保证缩放后仍 ≥ 裁剪块，裁剪不越界）：
  - `img = upsample_bilinear(img, scale_factor=scale)`，双线性插值。
  - `point *= scale`，坐标同步缩放。
  - 否则跳过缩放，保持原尺寸。

### 步骤 4 — 随机裁剪 Random Crop（`random_crop`, dataset.py:154-177）
仅当 `self.patch=True`。对**同一张缩放后的图**裁剪 `CROP_NUMBER`（=4）个 patch：
- 每个 patch 独立随机选起点 `start_h ∈ [0, H-CROP_SIZE]`、`start_w ∈ [0, W-CROP_SIZE]`。
- 裁出 `CROP_SIZE × CROP_SIZE` 的方块；落在该块内的点被保留并平移到 patch 局部坐标系。
- 输出 `result_img` 形状 `[CROP_NUMBER, 3, CROP_SIZE, CROP_SIZE]`，`point` 变成长度 4 的列表。
- **一张图 → 4 个裁剪块**。

### 步骤 5 — 随机水平翻转 Random Flip（dataset.py:105-109）
仅当 `self.flip=True` 且 `random.random() > 0.5`（≈50% 概率，对该图的 4 个 patch 整体生效）：
- `img = img[:, :, :, ::-1]`：沿宽度方向水平镜像。
- `point[:, 0] = CROP_SIZE - point[:, 0]`：x 坐标镜像（无垂直翻转、无旋转）。

### 步骤 6 — 打包成 batch（`collate_fn_crowd`, build.py:57-70）
- DataLoader 的 `BATCH_SIZE=8` 指 **8 张原图**；经裁剪后每张产生 4 个 patch。
- `collate_fn_crowd` 把 `[8, 4, 3, S, S]` 展平成 **`8×4 = 32` 个独立训练样本**。
- `_nested_tensor_from_tensor_list` 把同 batch 的 patch 零填充（pad）到 H、W 均为 **128 的整数倍**后堆叠。

**训练阶段总结（默认 SHHA 配置）：**
```
1 张原图 (RGB, H×W)
  → ToTensor + ImageNet Normalize           [3,H,W]
  → 随机缩放 scale∈[0.7,1.0]（满足条件才缩放） [3,H',W']
  → 随机裁剪 4 个 128×128 patch              [4,3,128,128]
  → 50% 概率整体水平翻转                       [4,3,128,128]
  → 与同 batch 其他图展平 + pad 到 128 倍数 → 模型输入 [32,3,128,128]
```

---

## 2. 验证 / 测试阶段（dataset.py:110-119）

测试**不做随机增强**，只保留确定性的尺寸归一与标准化：

1. 同样 `ToTensor + Normalize`（步骤 2）。
2. **整图缩放**（无裁剪、无翻转），规则：
   - `UPPER_BOUNDER != -1` 且 `max_size > UPPER_BOUNDER` → `scale = UPPER_BOUNDER / max_size`；
   - 否则 `max_size > 2560` → `scale = 2560 / max_size`（硬上限，防止超大图爆显存）；
   - 否则 `scale = 1.0`（原尺寸）。
3. `upsample_bilinear` 缩放整图，`point *= scale`。
4. 单图 `pad` 到 128 倍数后送入模型（batch=1）。

> 测试只缩小不放大，且整张图一次性推理。

---

## 3. 三个数据集的增强策略对比

**结论先行：三个细胞数据集（BCData / CoNIC / MoNuSeg）以及原始 SHHA 使用的增强*种类*完全相同**
——都是 `ToTensor → Normalize → 随机缩放 → 4×随机裁剪 → 50%水平翻转`，**没有任何数据集额外引入
颜色抖动、旋转、弹性形变等**。差异**仅在数值超参**上，而这些超参会通过第 1 节步骤 3 的分支逻辑
**间接改变随机缩放的行为**。

| 配置项 | SHHA（基线/论文） | BCData_finetune | CoNIC_finetune | MoNuSeg_finetune |
|--------|------------------|-----------------|----------------|------------------|
| `AUGUMENTATION` | `Normalize,Crop,Flip` | 同左 | 同左 | 同左 |
| `CROP_SIZE` | **128** | **256** | 128 | **256** |
| `CROP_NUMBER` | 4 | 4 | 4 | 4 |
| `UPPER_BOUNDER` | **-1** | **1024** | -1 | -1 |
| `NUM_WORKERS` | 0 | 4 | 4 | 4 |
| `BATCH_SIZE` | 8 | 8 | 8 | 8 |
| 典型原图尺寸 | 不定（人群图，长边可达数千） | ~640×640 | 256×256 | ~1000×1000 |
| **随机缩放分支** | `max_size>2560` 才压缩；训练 scale∈**[0.7,1.0]**（仅缩小） | `max_size≤1024` 命中中间分支 → scale∈**[0.7,1.3]**（**可放大**） | scale∈**[0.7,1.0]**（仅缩小） | scale∈**[0.7,1.0]**（仅缩小） |
| 测试整图缩放 | 长边>2560 才压到 2560 | 长边>1024 即压到 1024 | 长边>2560 才压 | 长边>2560 才压 |

### 差异要点解读
1. **裁剪块大小不同**：CoNIC/SHHA 用 `128²`，BCData/MoNuSeg 用 `256²`。细胞图细节更大、密度结构需要更大感受野，故裁剪块翻倍；但每张图仍固定裁 4 块。
2. **BCData 是唯一会"放大"的数据集**：它设了 `UPPER_BOUNDER=1024`，而 BCData 原图约 640×640 < 1024，于是命中
   "`UPPER_BOUNDER!=-1` 且 `max_size≤UPPER_BOUNDER`"分支，`scale∈[0.7,1.3]` —— **允许上采样到 1.3×**。
   SHHA/CoNIC/MoNuSeg 走 `UPPER_BOUNDER==-1` 分支，scale 被钳在 `[0.7,1.0]`，**永不放大**。
   这是三个细胞数据集之间唯一的*行为性*差异（而非单纯数值差异）。
3. **测试期尺寸上限不同**：BCData 受 `UPPER_BOUNDER=1024` 约束（>1024 即压缩）；其余受全局硬上限 2560 约束。
   由于这些图本身都 ≤1024，实际测试期基本都是 `scale=1.0` 原尺寸推理。
4. **`NUM_WORKERS` 仅影响加载并行度，与增强结果无关**；`BATCH_SIZE`、`CROP_NUMBER` 三者一致。

> 备注：所有细胞数据集的 `DATASETS.DATASET` 仍写作 `'SHHA'`，是为了**复用 SHHA 的 DataLoader 逻辑**
> （见各 finetune yml 注释），因此增强代码路径与 SHHA 完全一致，只是喂了不同超参。

---

## 4. 一句话总结
APGCC 的增强非常轻量且统一：**ImageNet 标准化 + 随机缩放（多数仅缩小，BCData 可放大到 1.3×）+
4 个随机正方形裁剪 + 50% 水平翻转**，没有任何颜色/几何强增强。三个细胞数据集与原始 SHHA 共用同一套
增强代码，差异只体现在 `CROP_SIZE`（128 vs 256）和 `UPPER_BOUNDER`（-1 vs 1024，后者让 BCData 允许上采样）。
