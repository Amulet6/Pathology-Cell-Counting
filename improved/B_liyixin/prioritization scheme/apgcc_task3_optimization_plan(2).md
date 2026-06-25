# APGCC 第 3 阶段五个改进方向与分工计划

## 1. 主线结论

本阶段选择 APGCC 作为主要改进模型，但不将其表述为“所有指标综合最优”。更稳妥的结论是：

> APGCC 在 CoNIC unified 下存在较高 MSE 和系统性少计问题，但该问题可以通过阈值扫描和子集分析被明确诊断；同时，APGCC 直接输出细胞中心点，天然适合区域计数、TP/FP/FN 可视化和后处理优化。因此，本阶段选择 APGCC 作为可解释、可改进的主模型。

选择 APGCC 的理由：

- APGCC 直接输出中心点，与统一评估脚本的中心点匹配协议一致。
- 相比 STEERER，APGCC 不需要额外从密度图转换为点坐标，减少后处理偏差。
- 相比 PET，APGCC 当前复现实验中更适合作为轻量、可解释的点定位改进基线。
- APGCC 的主要错误集中在高密度区域漏检和置信度偏低，问题明确，便于设计针对性优化。
- APGCC 的点输出方便完成题目要求的预测位置可视化、区域计数和改进前后指标对比。

老师给出的 Count Anything 论文对本阶段最有参考价值的是三点：其一，最终输出仍是实例点集，与 APGCC 和 12px 中心点评估一致；其二，它使用 region-level sparse counter 与 pixel-level dense counter 的双粒度计数思想，说明稀疏候选和密集小目标召回应当互补；其三，Complementary Count Fusion 的思想可以转化为 APGCC 点预测与密集辅助预测的融合策略。

## 2. 现有问题诊断

APGCC 在 CoNIC unified 上的高 MSE 主要来自高密度子集中的系统性少计，而不是全局随机误差。

已有阈值扫描结果：

| score 阈值 | Total Pred | 总误差 | MAE | MSE | RMSE |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 96063 | -14.64% | 19.63 | 974.08 | 31.21 |
| 0.40 | 92289 | -18.00% | 22.17 | 1160.23 | 34.06 |
| 0.45 | 90394 | -19.68% | 23.62 | 1263.83 | 35.55 |
| 0.50 | 88094 | -21.73% | 25.50 | 1395.13 | 37.35 |

按来源子集统计，0.30 阈值下仍然主要坏在高密度子集：

| 子集 | 总误差 | MAE | MSE |
|---|---:|---:|---:|
| crag | -6.25% | 9.89 | 212.41 |
| dpath | -16.12% | 22.80 | 962.28 |
| glas | -20.92% | 37.78 | 2256.51 |
| pannuke | -27.20% | 50.56 | 6465.12 |

错误链条可以概括为：

```text
高密度细胞 / 粘连 / 形态不规则
        ↓
点 proposal 与真实细胞中心匹配更困难
        ↓
模型对部分真实细胞输出较低置信度
        ↓
固定 0.50 阈值过滤低置信预测点
        ↓
Recall 下降，出现系统性漏检
        ↓
高密度图像少计几十个细胞
        ↓
平方误差放大，MSE 升高
```

所以本阶段优化目标不是泛泛地“提升 APGCC”，而是明确解决：

1. CoNIC 高密度区域少计。
2. 统一增强后置信度偏低。
3. 密集粘连导致相邻细胞被合并。
4. 不规则形态导致中心点定位偏移。
5. 不同数据集和染色来源导致跨域泛化不稳定。

## 3. 五个改进方向

统一评测、统一可视化和报告汇总不单独分配给某一个人，而是全员共同完成。每个人负责一个改进方向，并且都需要交付：可运行模块、消融指标、可视化样例和报告文字。

| 成员 | 改进方向 | 解决问题 | 主要数据集 | 核心交付 |
|---|---|---|---|---|
| A | K=8 reference points + 置信度校准 | proposal 覆盖不足、固定阈值导致少计 | CoNIC | K4/K8 对比、阈值曲线、少计诊断 |
| B | Count Anything 式密集辅助分支 | 高密度小细胞漏检 | CoNIC / BCData | dense head、密度辅助损失、召回提升分析 |
| C | 点级 Repulsion Loss + Adaptive NMS | 密集粘连、相邻细胞合并、重复点 | CoNIC / MoNuSeg | repulsion 消融、NMS 消融、粘连区可视化 |
| D | DCNv2 形变卷积 + 边缘截断处理 | 细胞形态不规则、边缘不完整细胞 | 三个数据集 | DCNv2 模块、edge ignore、形态鲁棒性对比 |
| E | GRL 域适应 + 病理染色鲁棒增强 | 染色差异、跨来源泛化不稳 | 三个数据集 | 域分类器、GRL 消融、跨数据集指标 |

## 4. 方向 A：K=8 Reference Points + 置信度校准

### 目标

缓解 CoNIC 高密度区域 proposal 覆盖不足和固定 0.50 阈值导致的系统性少计。

### 方法

- 将 APGCC 每个特征图位置的 reference points 从 K=4 提高到 K=8。
- 在验证集扫描 score threshold：0.25、0.30、0.35、0.40、0.45、0.50。
- 建立验证集阈值选择规则，固定 `val best threshold`，测试集只评一次。
- 按 crag、dpath、glas、pannuke 子集统计 Total Pred Error、MAE、MSE、Recall、F1。

### 实验表

| 方法 | K | 阈值 | MAE | MSE | Precision | Recall | F1 | Total Pred Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| APGCC baseline | 4 | 0.50 |  |  |  |  |  |  |
| APGCC + Calibration | 4 | val best |  |  |  |  |  |  |
| APGCC-K8 | 8 | 0.50 |  |  |  |  |  |  |
| APGCC-K8 + Calibration | 8 | val best |  |  |  |  |  |  |

### 预期结论

如果 K=8 提升 Recall 并降低 glas、pannuke 等高密度子集 MSE，说明原始 K=4 的局部 proposal 覆盖不足。若阈值校准能降低 MSE，说明原始 0.50 阈值过于保守。

## 5. 方向 B：Count Anything 式密集辅助分支

### 目标

借鉴 Count Anything 的 Pixel-level Dense Counter 思想，在 APGCC 稀疏点 proposal 之外增加密集召回能力。

### 方法

保留 APGCC 原点预测分支，新增一个轻量 dense auxiliary head：

- 输入高分辨率特征或 backbone 中间层特征。
- 输出 Gaussian density map 或 dense point confidence map。
- 从 GT 点生成 Gaussian density target。
- 增加辅助损失：

```text
Loss = L_APGCC + lambda_density * L_density
```

简单版实现为 density map 辅助监督；进阶版实现为 dense point candidates，并与 APGCC 点输出融合。

### 融合规则

- APGCC 高置信点优先保留。
- Dense head 只在高密度区域补充候选点。
- 距离过近的点用 NMS 或 fusion 去重。
- 最终仍输出统一点格式：`image_id, x, y, score`。

### 实验表

| 方法 | Dense Head | K | 阈值 | MAE | MSE | Precision | Recall | F1 |
|---|---|---:|---|---:|---:|---:|---:|---:|
| K8 + Calibration | 否 | 8 | val best |  |  |  |  |  |
| + Density Aux | 是 | 8 | val best |  |  |  |  |  |
| + Dense Fusion | 是 | 8 | val best |  |  |  |  |  |

### 预期结论

该方向主要看 Recall、FN 数量和高密度 top 20% patch MSE 是否改善。

## 6. 方向 C：点级 Repulsion Loss + Adaptive NMS

### 目标

解决密集粘连区域中相邻细胞点被合并、重复点和点间分离不足的问题。

### 方法一：点级 Repulsion Loss

Repulsion Loss 原本用于密集行人检测，本课题中改为点级约束：

- 只对 matched positive predictions 做点间排斥。
- 不对所有预测点无差别排斥，避免破坏真实密集细胞结构。
- 排斥半径参考 12px 评测半径，可扫描 6px、8px、12px。
- 权重从小开始扫描：0.01、0.05、0.1。

形式上可写为：

```text
Loss = L_APGCC + lambda_rep * L_repulsion
```

### 方法二：Adaptive NMS

- 密集区域使用较小 NMS 半径，保留相邻细胞。
- 稀疏区域使用较大 NMS 半径，减少重复预测。
- 区域密度可由预测点局部密度或 density head 响应估计。

### 实验表

| 方法 | Repulsion | NMS | MAE | MSE | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|---:|---:|
| K8 + Calibration | 否 | default |  |  |  |  |  |
| + Repulsion | 是 | default |  |  |  |  |  |
| + Adaptive NMS | 否 | adaptive |  |  |  |  |  |
| + Repulsion + Adaptive NMS | 是 | adaptive |  |  |  |  |  |

## 7. 方向 D：DCNv2 形变卷积 + 边缘截断处理

### 目标

增强模型对形态不规则细胞和边缘不完整细胞的鲁棒性。

### 方法一：DCNv2 形变卷积

在 APGCC backbone 高层特征提取部分加入 deformable convolution：

- 优先替换最后一到两层普通卷积。
- 可使用 `torchvision.ops.deform_conv2d` 或 MMCV 的 DeformConv2dPack。
- offset conv 初始化为 0，使初始行为接近普通卷积。
- 避免一开始大规模替换，先保证训练稳定。

报告表述：

> DCNv2 允许卷积采样位置根据目标形态自适应偏移，从而提升对不规则细胞轮廓和非标准形态的特征提取能力。

### 方法二：边缘截断处理

- 图像四周设置 12px 或 16px ignore band。
- 训练时降低边缘截断细胞 loss 权重。
- 推理时对边缘低置信点使用更谨慎阈值。

### 实验表

| 方法 | DCNv2 | Edge Ignore | MAE | MSE | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|---:|---:|
| K8 + Calibration | 否 | 否 |  |  |  |  |  |
| + DCNv2 | 是 | 否 |  |  |  |  |  |
| + Edge Ignore | 否 | 是 |  |  |  |  |  |
| + DCNv2 + Edge Ignore | 是 | 是 |  |  |  |  |  |

## 8. 方向 E：GRL 域适应 + 病理染色鲁棒增强

### 目标

缓解不同数据集、不同染色来源和不同组织域之间的视觉域偏移。

### 方法一：GRL 域适应

在 APGCC backbone feature 后加入梯度反转层和域分类器：

```text
Feature -> GRL -> Domain Classifier -> Domain Loss
```

训练目标：

```text
Loss = L_APGCC + lambda_domain * L_domain
```

GRL 会让特征提取器学习更难区分来源域的表示，从而减少染色和来源差异对计数的影响。

域标签可以使用：

- BCData / CoNIC / MoNuSeg；
- 或 CoNIC 内部 crag / dpath / glas / pannuke。

### 方法二：病理染色鲁棒增强

- stain jitter；
- brightness / contrast jitter；
- 轻量 blur/noise 消融；
- 关闭过强增强，观察是否恢复 APGCC 置信度。

### 实验表

| 方法 | GRL | Stain Aug | MAE | MSE | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|---:|---:|
| K8 + Calibration | 否 | 否 |  |  |  |  |  |
| + Stain Aug | 否 | 是 |  |  |  |  |  |
| + GRL | 是 | 否 |  |  |  |  |  |
| + GRL + Stain Aug | 是 | 是 |  |  |  |  |  |

## 9. 全员共同任务：统一评测、可视化与报告

这些工作不分配给某一个人，而是五个人共同完成。每个成员必须把自己的改进结果转换为统一格式，再统一进总表。

### 统一输出格式

```text
image_id, x, y, score
```

### 统一指标

- MAE；
- MSE；
- RMSE；
- Precision@12px；
- Recall@12px；
- F1@12px；
- Total Pred Error；
- 高密度子集 MAE/MSE；
- 区域级 MAE/MSE。

主表使用 12px 匹配半径，同时补充 6px 和 24px 作为鲁棒性分析。

### 统一可视化

每个方向至少提供 3 张改进前后可视化：

- 原图 + GT 点；
- 原图 + Pred 点；
- TP/FP/FN 三色图；
- 预测密度热力图；
- 4x4 区域误差热力图；
- 改进前后预测点叠加图。

### 最终总表

| 方法 | 负责方向 | MAE | MSE | Precision | Recall | F1 | 主要变化 |
|---|---|---:|---:|---:|---:|---:|---|
| APGCC baseline | 原始模型 |  |  |  |  |  |  |
| + K8 + Calibration | A |  |  |  |  |  | 减少少计 |
| + Dense Aux / Fusion | B |  |  |  |  |  | 补密集召回 |
| + Repulsion / Adaptive NMS | C |  |  |  |  |  | 分离粘连点 |
| + DCNv2 / Edge Ignore | D |  |  |  |  |  | 适应不规则形态 |
| + GRL / Stain Robust | E |  |  |  |  |  | 改善跨域鲁棒性 |
| Final Combined | 全员 |  |  |  |  |  | 组合有效模块 |

## 10. 推荐执行顺序

1. 所有人先统一 baseline 输出格式和评测脚本。
2. A 先完成 `K8 + Calibration`，作为其他方向的共同基础分支。
3. B/C/D/E 分别在相同基础分支上做单独改进。
4. 每个方向只和同一个基础分支比较，避免指标不可比。
5. 最终只组合有效且不冲突的模块，不把所有模块无脑堆叠。

最终候选模型可以命名为：

```text
APGCC-DCR
```

含义：

- D: Dense recall / Deformable feature；
- C: Confidence calibration；
- R: Repulsion regularization。

## 11. 报告可用总结

本文选择 APGCC 作为主改进模型，并非因为其所有指标均最优，而是因为其点定位输出与细胞计数任务、区域统计和可视化要求高度一致。实验发现，APGCC 在 CoNIC unified 上出现较高 MSE，主要原因是高密度子集中的系统性少计：统一增强后模型预测置信度偏低，固定 0.50 阈值过滤了大量潜在正确的低置信预测点。

针对该问题，本文提出五个方向的改进：首先将 reference point 数量由 K=4 提高至 K=8，并通过验证集阈值校准缓解低置信漏检；其次借鉴 Count Anything 的双粒度计数思想，增加密集辅助分支以补充高密度小细胞召回；然后引入点级 Repulsion Loss 和自适应 NMS，减少粘连区域相邻细胞被合并；进一步使用 DCNv2 和边缘截断处理增强对不规则形态和边缘不完整细胞的鲁棒性；最后通过 GRL 域适应和病理染色增强提升跨数据集泛化能力。

所有改进均统一输出中心点坐标，并使用相同的 12px 匹配协议评估 MAE、MSE、Precision、Recall 和 F1，同时通过 TP/FP/FN 散点图、密度热力图和 4x4 区域计数图展示改进前后的差异。
