# 方向 D 核心代码变更记录

> 方向：DCNv2 + Edge Ignore | 基线：APGCC (VGG16-bn + IFI decoder)
>
> 本文件供撰写研究报告使用，按改动层次组织：新增文件 → 核心改动 → 配置扩散 → 工具脚本。

---

## 1. 新增文件

### 1.1 `models/dcnv2.py` — ModulatedDeformConv2d 实现（~156 行）

**内容**：可变形卷积 v2 的独立模块，基于 `torchvision.ops.deform_conv2d`。

**关键设计决策**：

| 决策点 | 做法 | 理由 |
|--------|------|------|
| offset 初始化 | `nn.init.constant_(..., 0.0)`，严格全零 | 确保 DCNv2 在初始化时等价于标准 Conv2d，预训练权重无损迁移 |
| mask 初始化 | `sigmoid(x + 20.0) ≈ 0.999999998` | 偏移量 0 + mask≈1 → 输出与标准卷积一致（已验证 <1e-6 diff） |
| offset 排列 | 标准交织 `[dy₀, dx₀, dy₁, dx₁, ...]` | 这是 torchvision 要求的输入格式 |
| 权重迁移 | `weight` → `deform_weight`（`_load_from_state_dict` override） | 兼容基线 checkpoint 加载，无需转换脚本 |
| post-load hook | 自动清理 `offset_conv.*` 的 missing_keys | 允许 `strict=True` 从旧 checkpoint 加载 |
| 自检 | `__name__ == '__main__'` 含 shape + identity + backward + load 四个检查 | 快速验证实现正确性 |

```python
# 核心架构（伪码）
class ModulatedDeformConv2d(nn.Module):
    def __init__(self, in_c, out_c, k=3, stride=1, padding=1):
        self.offset_conv = Conv2d(in_c, 3*k², k, stride, padding)  # 全零初始化
        self.deform_weight = Parameter(...)  # 从预训练 weight 继承
        self.bias = Parameter(...)           # 从预训练 bias 继承

    def forward(self, x):
        offset_mask = self.offset_conv(x)           # [B, 3k², H, W]
        offset = interleave(dy, dx)                 # [B, 2k², H, W]
        mask = sigmoid(mask_raw + 20)               # [B, k², H, W]
        return deform_conv2d(x, offset, self.deform_weight, self.bias,
                             stride, padding, mask=mask)
```

---

## 2. 核心模型改动

### 2.1 `models/Encoder.py` — Backbone 中注入 DCNv2（+30 行）

**改动点**：

1. `Base_VGG.__init__()` 新增参数 `dcnv2_enabled`、`dcnv2_layers`
2. 新增方法 `_inject_dcnv2(dcnv2_layers)`：
   - 定位 `body4`（conv5，H/16 特征）中所有 `nn.Conv2d`
   - 将最后 `dcnv2_layers` 个替换为 `ModulatedDeformConv2d`
   - 使用 `nn.Sequential.__setitem__` 原地替换（`body4[idx] = new_mod`）
   - 自动迁移预训练权重
   - 打印替换日志 `[DCNv2] body4[{idx}] Conv2d → ModulatedDeformConv2d`

**为什么只替换 body4**：body4 是 H/16 尺度的最深特征（512 通道）。conv5 的语义抽象最高，形变卷积在这里学习几何扭曲最有意义；浅层（body1-3）以纹理/边缘为主，不适合形变建模。

### 2.2 `models/APGCC.py` — Edge Ignore + DCNv2 config 透传（~160 行改动）

#### 2.2.1 `Model_builder._build_encoder()` — DCNv2 配置透传（+3 行）

```python
self.cfg.MODEL.ENCODER_kwargs['dcnv2_enabled'] = self.cfg.MODEL.get('DCNV2_ENABLED', False)
self.cfg.MODEL.ENCODER_kwargs['dcnv2_layers'] = self.cfg.MODEL.get('DCNV2_LAYERS', 0)
```

向后兼容：未在 yml 中设置时自动退化为 `False` / `0`。

#### 2.2.2 `SetCriterion_Crowd.__init__()` — Edge Ignore 参数（+4 行）

新增参数 `edge_band`（band 宽度，px）和 `edge_weight`（band 内 anchor 的 loss 权重，0~1）。

#### 2.2.3 `SetCriterion_Crowd.loss_labels()` — 分类 loss 的边缘屏蔽（+2 行）

```python
if self.edge_band > 0 and self.current_edge_mask is not None:
    loss_per_anchor = loss_per_anchor * self.current_edge_mask
```

对所有 anchor（正样本 + 负样本）的 CE loss 按边缘距离加权。

#### 2.2.4 `SetCriterion_Crowd.loss_points()` — 定位 loss 的边缘屏蔽（+3 行）

同上，对 Hungarian 匹配上的正样本的 L2 回归 loss 加权。

#### 2.2.5 `SetCriterion_Crowd._compute_edge_mask()` — Anchor 级边缘权重计算（~30 行，核心新增）

```python
def _compute_edge_mask(self, outputs, crop_size):
    # 1. 恢复 anchor 网格坐标 = pred_points - offset
    anchor_xy = outputs['pred_points'] - outputs['offset']
    # 2. 计算每个 anchor 到 crop 四边的最短距离
    dist_left, dist_right = x, crop_w - x
    dist_top, dist_bottom = y, crop_h - y
    min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
    # 3. 线性过渡：边界处 weight=edge_weight，band 边缘 weight=1.0
    mask = where(min_dist < edge_band,
                 edge_weight + (1 - edge_weight) * (min_dist / edge_band),
                 1.0)
    return mask  # [B, N]
```

**设计要点**：
- 以 anchor 的网格坐标（非预测坐标）为基准——这是模型在设计上"应该关心"的位置，不受预测偏移干扰
- 线性过渡避免硬截断（梯度不连续）
- 正样本和负样本使用同一 mask（逻辑一致）

#### 2.2.6 `SetCriterion_Crowd.forward()` — 边缘 mask 的生成和缓存（+4 行）

在每个 batch 前计算 mask，缓存到 `self.current_edge_mask`，供 `loss_labels` / `loss_points` 消费。

### 2.3 `models/__init__.py` — criterion 构造透传 edge 参数（+2 行）

```python
criterion = SetCriterion_Crowd(...
    edge_band=cfg.MODEL.get('EDGE_BAND', 0),
    edge_weight=cfg.MODEL.get('EDGE_WEIGHT', 0.0))
```

向后兼容：yml 中未设置时默认为 0（禁用 Edge Ignore）。

---

## 3. 推理管线改动

### 3.1 `engine.py` — 训练期 eval + 推理时 edge filter

#### 3.1.1 `evaluate_crowd_counting()` — 新增 edge_band 参数（+10 行）

```python
def evaluate_crowd_counting(model, data_loader, device, threshold=0.5,
                            edge_band=0):
    ...
    keep = outputs_scores > threshold
    if edge_band > 0:
        keep = edge_aware_filter(outputs_points, outputs_scores,
                                 global_shape, patch_box, edge_band, threshold)
```

#### 3.1.2 `Trainer.handle_new_epoch()` — 训练期 eval 传递 edge_band（改 1 行）

```python
result = evaluate_crowd_counting(self.model, self.val_dl,
    next(self.model.parameters()).device,
    edge_band=self.cfg.TEST.get('EDGE_BAND', 0))
```

训练期 eval（val set）默认 `EDGE_BAND=0`，仅在 test 时使用 edge filter。

### 3.2 `util/misc.py` — edge_aware_filter 函数（~60 行，新增）

**功能**：推理时过滤靠近内部 patch 接缝的预测。仅对**内部接缝**（非大图真实边界）提升置信度阈值（+0.3）。

```python
def edge_aware_filter(pred_pts, pred_scores, global_shape, patch_box,
                      edge_band, threshold):
    # 逐边检查是否为内部接缝
    if xmin > 0:   suppress |= (dist_left < edge_band)
    if xmax < W:   suppress |= (dist_right < edge_band)
    if ymin > 0:   suppress |= (dist_top < edge_band)
    if ymax < H:   suppress |= (dist_bottom < edge_band)
    # 内部接缝处 threshold += 0.3，大图边界不抑制
    adjusted_threshold = where(suppress, threshold + 0.3, threshold)
    keep = pred_scores > adjusted_threshold
```

**关键设计**：大图真实边界不抑制（测的是整张图，不需要抑制全图边界细胞）。只有滑动窗口推理时，内部接缝附近的预测才需要更高置信度阈值。

---

## 4. 评测脚本

### 4.1 `eval_centroid.py` — 统一评测包装器（约 105 行）

**功能**：
- 加载模型 + 数据集，遍历 test set 生成 GT 和预测 JSON
- 自动调用 `centroid_eval.py` 计算 MAE/MSE/RMSE/P/R/F1@多阈值
- 支持 `--edge-band` 参数（默认 0）

**改动点**：
- 原版 `centroid_eval.py` 只做点匹配评估，不含模型推理管线
- 本脚本完整复用 APGCC 的 dataloader/model 构建，确保推理逻辑和训练一致
- 修复了 `Path(__file__).resolve().parent` 路径计算

### 4.2 `analyze_region_error.py` — 4×4 网格区域误差分析（~150 行）

独立分析脚本，划分 4×4 网格，分别统计外圈/内部的 Over%/Under%。

---

## 5. 实验配置文件

### 5.1 新增配置（9 个 yml）

每个配置基于对应数据集的 `*_finetune.yml` 复制，仅修改：

| 变体 | DCNV2_ENABLED | DCNV2_LAYERS | EDGE_BAND | EDGE_WEIGHT | EPOCHS |
|------|:-----------:|:----------:|:-------:|:---------:|:-----:|
| `*_finetune_dcnv2.yml` | True | 2 | — | — | <原值> |
| `*_finetune_edge.yml` | — | — | 16 | 0.1 | <原值> |
| `*_finetune_dcnv2_edge.yml` | True | 2 | 16 | 0.1 | <原值> |

三数据集 × 3 新变体 = 9 个配置文件。所有新配置的关键参数（SEED=1229、BATCH_SIZE=8、LR=5e-5/5e-6、FINETUNE=True、RESUME_PATH=SHHA_best.pth）与基线完全一致，仅改 TAG、OUTPUT_DIR、模型块参数。

### 5.2 关键超参数选择

| 参数 | 值 | 选择理由 |
|------|-----|---------|
| DCNV2_LAYERS | 2 | body4 最后 2 层 Conv2d。1 层形变自由度不足，>2 层参数量过大且可能过度扭曲 |
| EDGE_BAND | 16 px | 在 stride-8 anchor 网格上约 2 个 grid 格距，覆盖 crop 边界的截断过渡带 |
| EDGE_WEIGHT | 0.1 | 边界 anchor 的 loss 贡献降为 10%。0 会导致完全忽略边界信号；0.5 抑制太弱 |

---

## 6. 改动影响范围矩阵

| 文件 | 改动类型 | DCNv2 相关 | Edge Ignore 相关 | 向后兼容 |
|------|----------|:--------:|:--------------:|:------:|
| `models/dcnv2.py` | 🆕 新增 156 行 | ✓ | | ✅ 独立模块 |
| `models/Encoder.py` | ✏️ +30 行 | ✓ | | ✅ 用 dcnv2_enabled=False 退化为原 behavior |
| `models/APGCC.py` | ✏️ ~60 行 | ✓ | ✓ | ✅ edge_band=0 时完全跳过 |
| `models/__init__.py` | ✏️ +2 行 | | ✓ | ✅ 默认 edge_band=0 |
| `engine.py` | ✏️ ~12 行 | | ✓ | ✅ 默认 edge_band=0 |
| `util/misc.py` | ✏️ +60 行 | | ✓ | ✅ 独立函数 |
| `eval_centroid.py` | ✏️ 路径修复 | | ✓ | ✅ 不改就找不到 centroid_eval.py |
| `configs/*.yml` | 🆕 9 个文件 | ✓ | ✓ | ✅ 新增，不动基线 |

**总计**：1 个新文件、3 个核心文件改动（Encoder / APGCC / __init__）、2 个推理文件改动（engine / misc）、1 个评测脚本修复、9 个配置文件新增。

---

## 7. 参数量变化

| 配置 | 参数量 | Δ |
|------|:-----:|:--:|
| Baseline (VGG16-bn + IFI) | 17.75M | — |
| +DCNv2 (2 layers) | 18.00M | +0.25M (+1.4%) |
| +Edge Ignore | 17.75M | 0（纯训练技巧，无参数） |
| DCNv2+Edge | 18.00M | +0.25M |

每个 DCNv2 替换层增加的参数：`3×3×3` offset_conv = 27×in_ch×out_ch/k² 的 offset 预测。以 512→512 卷积为例，offset_conv 的参数量约为 `3×3×3×27 = 13,824`，相对于原卷积的 `512×512×3×3 = 2,359,296` 仅为 0.6%。

---

## 8. 验证与回退安全

所有改动遵循零侵入原则：
1. **checkpoint 兼容**：D 训练的模型可通过 `_load_from_state_dict` 加载基线权重；基线 checkpoint 也可正常加载（offset_conv 自动零初始化）
2. **配置兼容**：不设置新 yml key 时所有 feature 默认为 disabled
3. **推理兼容**：`--edge-band` 默认为 0（不执行边缘过滤）
4. **自检覆盖**：`dcnv2.py` 的 `__main__` 自检确保 zero-init identity + backward NaN-free + load 兼容性
