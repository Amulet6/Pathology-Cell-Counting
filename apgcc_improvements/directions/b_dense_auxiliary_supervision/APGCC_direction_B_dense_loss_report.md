# APGCC 方向 B 改进汇报

## 1. 任务背景

本次工作对应方向 B：**Count Anything 式密集辅助分支**。该方向的核心目标是缓解 APGCC 在**高密度、小细胞场景**中的漏检问题，重点关注如下现象：

- APGCC 原始训练以点级匹配监督为主，更擅长学习“离散、明确”的点目标。
- 在细胞密集、目标尺寸小、相邻目标间距近的区域，模型容易出现召回不足，即**漏检较多**。
- 因此，方向 B 的基本思路是：在原有点检测主分支之外，引入一个对局部密集区域更敏感的辅助分支，为主干特征增加连续的密集监督信号。

本次最终采用的改进版本为 **dense loss-only**，即：**只在训练阶段增加 dense auxiliary branch 与 dense loss，不修改 APGCC 的最终推理流程**。

---

## 2. 基线与对比原则

为了保证改进实验与复现基线可直接对比，本次所有实验均遵循同组同学整理的复现说明：

- 基线模型：APGCC
- 预训练权重：`SHHA_best.pth`
- 对比路线：使用 `*_finetune.yml`，即 **native / 官方原生增强**

本次改进与 baseline 的公平对比原则如下：

- 保持主干网络结构不变：仍为 `VGG16-bn + IFI decoder`
- 保持训练范式不变：仍从 `SHHA_best.pth` 微调
- 保持数据划分与评测方式不变
- 保持测试阶段输出逻辑不变
- 仅增加 dense auxiliary 分支与对应训练损失

因此，本次实验的差异只来自于“是否加入 dense auxiliary supervision”。

---

## 3. 改进动机

APGCC 原始训练中，监督主要依赖于预测点与 GT 点之间的匹配。这种监督方式能够直接优化计数与定位，但也存在一个局限：它对局部区域的“密集程度”表达较弱。换句话说，模型知道某些位置应该有点，但不一定能充分感知“这一片区域整体上是高密度细胞区域”。

对于病理细胞计数任务，尤其是 BCData 和部分 MoNuSeg 图像，存在以下问题：

- 单个细胞很小；
- 邻近细胞之间距离短；
- 局部区域内往往存在多个紧邻目标；
- 点级监督对这些密集区域的结构信息利用不足。

因此，本次尝试在 APGCC 中补充一个**密集图学习任务**。该任务并不直接替代原来的点检测，而是作为一个辅助任务，引导主干特征在训练时更关注“细胞聚集区域”。

---

## 4. 方法设计

### 4.1 总体思路

本次最终采用的方法可以概括为：

> 在 APGCC 原模型基础上新增一个 dense auxiliary branch，训练时通过 dense loss 监督该分支学习细胞密集响应图；测试时仍仅使用 APGCC 原始点检测分支输出最终预测点。

也就是说：

- **训练阶段**：主分支损失 + dense 辅助损失
- **测试阶段**：仍然只使用主分支输出

这种设计的优点是：

- 实现简单，改动集中；
- 不破坏 APGCC 原有预测机制；
- 与 baseline 的公平性较强；
- 可以单独验证“密集监督是否有效”，而不引入额外推理端不确定性。

### 4.2 Dense Auxiliary Branch

在模型实现中，我们在编码器中间层特征后增加了一个额外的 `dense_head`，输出单通道 `dense_map`。

该分支从中层特征中学习每个位置的密集响应，输出可以理解为：

- 响应越高，表示该位置越可能处于细胞中心附近或细胞密集区域。

### 4.3 Dense Supervision 构造方式

GT 监督并不是新的分割标注，而是由原有的细胞质心点标注直接构造：

- 对每个 GT 质心，在特征图尺度上生成一个二维高斯响应；
- 多个细胞的高斯响应通过逐点最大值方式叠加；
- 得到一张 dense target map；
- 用该 dense target map 去监督 `dense_head` 的输出。

本次使用的关键参数为：

- `DENSE_AUX_EN: True`
- `DENSE_AUX_LEVEL: 3`
- `DENSE_SIGMA: 2.0`
- `loss_dense: 0.1`

### 4.4 Dense Loss

训练时对 `dense_head` 输出做 `sigmoid`，并与构造的 dense target map 计算 MSE loss，作为附加损失项加入总损失。

即：

- 原 APGCC 损失负责点分类与点位置回归；
- dense loss 负责学习局部密集结构。

这使得模型在训练时不仅关注“某个点是否存在”，也关注“局部区域是否整体呈现细胞聚集模式”。

---

## 5. 实现过程

### 5.1 第一阶段：Dense Loss-Only

第一阶段的目标是验证一个最基本的问题：

> 仅增加 dense 辅助监督，而不改推理逻辑，是否能提升 APGCC 的效果？

为此我们完成了以下改动：

- 在模型中增加 `dense_head`
- 在损失中增加 `loss_dense`
- 在配置中增加：
  - `DENSE_AUX_EN`
  - `DENSE_AUX_LEVEL`
  - `DENSE_SIGMA`
- 修改预训练权重加载逻辑，使新增加的 `dense_head` 在加载 SHHA 权重时允许缺失参数

该阶段是本次最终采用的版本。

### 5.2 第二阶段：Dense Candidate Fusion 尝试

在确认 dense supervision 有潜力后，我们进一步尝试了更激进的方案：在推理阶段将 dense 分支产生的候选点与主分支点预测进行融合，以进一步提升召回。

这一阶段我们实现了两版：

- **fusion v1**：直接从 dense map 中提取局部峰值并与主分支点做半径去重后融合
- **fusion v2**：引入更保守的门控策略，仅在低置信主分支候选且 dense 支持较强时恢复点，同时严格限制新增点数

结论是：

- `fusion v1` 在 BCData 上明显退化，FP 激增；
- `fusion v2` 相比 v1 明显改善，但最终仍未超过 `dense loss-only`；
- 因此最终不采用 fusion 路线，而保留更稳定的 `dense loss-only` 作为正式改进版本。

---

## 6. 实验设置

### 6.1 配置来源

所有训练配置均参考 `REPRODUCE.md`。

本次采用的是 native / finetune 路线：

- `BCData_finetune.yml`
- `CoNIC_finetune.yml`
- `MoNuSeg_finetune.yml`

统一原则：

- 从 `SHHA_best.pth` 微调
- 训练中使用 `val.list` 选 `best.pth`
- 最终使用 `eval_centroid.py` 在 `test.list` 上做正式评测

### 6.2 数据集说明

本次实验使用的数据均为已经转换为 APGCC 格式的数据集。

其中：

- CoNIC 使用 release-based APGCC 格式数据；
- MoNuSeg 使用 `monuseg_split.json` 对应的 `30/7/14` 划分。

### 6.3 评测指标

正式评测使用以下指标：

- Counting:
  - MAE
  - RMSE
- Localization:
  - F1@6
  - F1@12
  - F1@24

---

## 7. 实验结果

### 7.1 BCData

#### baseline

- MAE: 18.27
- RMSE: 23.73
- F1@6: 0.7176
- F1@12: 0.8145
- F1@24: 0.8387

#### dense loss-only

- MAE: 17.51
- RMSE: 23.41
- F1@6: 0.7372
- F1@12: 0.8247
- F1@24: 0.8464

#### 结果分析

BCData 上改进效果最明显：

- 计数误差下降；
- 三个定位阈值下的 F1 均提升；
- 说明 dense supervision 能够有效提升该数据集上的召回能力，同时没有破坏整体定位质量。

因此，BCData 是本次改进最有力的正向证据。

### 7.2 CoNIC

#### baseline

- MAE: 11.99
- RMSE: 19.74
- F1@6: 0.6460
- F1@12: 0.7944
- F1@24: 0.8794

#### dense loss-only

- MAE: 13.69
- RMSE: 22.89
- F1@6: 0.6618
- F1@12: 0.8005
- F1@24: 0.8744

#### 结果分析

CoNIC 上改进并不成立：

- 近距离和中距离定位指标略有提升；
- 但 MAE 和 RMSE 均明显变差；
- 总体预测数偏少，说明仍存在更严重的漏检。

因此，dense loss-only 在 CoNIC 上没有带来稳定收益。

### 7.3 MoNuSeg

#### baseline

- MAE: 111.71
- RMSE: 116.01
- F1@6: 0.7361
- F1@12: 0.8032
- F1@24: 0.8291

#### dense loss-only

- MAE: 110.36
- RMSE: 114.74
- F1@6: 0.7278
- F1@12: 0.8101
- F1@24: 0.8363

#### 结果分析

MoNuSeg 上改进是小幅有效的：

- MAE 和 RMSE 均略有下降；
- F1@12 和 F1@24 提升；
- F1@6 略有下降。

这说明 dense supervision 对 MoNuSeg 也有帮助，但增益幅度弱于 BCData。

---

## 8. Fusion 尝试结果

为了验证 dense 分支是否不仅能改善训练特征，还能直接参与测试阶段补点，我们在 BCData 上额外尝试了两版 fusion。

### 8.1 fusion v1

- MAE: 18.80
- RMSE: 24.08
- F1@6: 0.6704
- F1@12: 0.7632
- F1@24: 0.7907

结果明显退化，主要原因是：

- dense 峰值候选过于激进；
- 引入了大量假阳性；
- Precision 和整体定位质量明显下降。

### 8.2 fusion v2

- MAE: 18.67
- RMSE: 24.23
- F1@6: 0.7229
- F1@12: 0.8173
- F1@24: 0.8404

fusion v2 相比 v1 已有明显恢复，但仍未超过 `dense loss-only`，且在计数误差上仍不如 baseline。

### 8.3 fusion 路线结论

虽然 dense 分支本身在训练阶段具有价值，但当前实现的 dense candidate fusion 在推理阶段未能带来额外收益。其主要问题在于：

- dense 候选缺少足够可靠的置信度校准；
- 推理阶段补点策略仍容易引入额外假阳性；
- 对不同数据集的稳定性不足。

因此，fusion 路线不作为本次最终方案。

---

## 9. 最终结论

本次方向 B 的最终改进方案确定为：**dense loss-only**。

即：

- 在 APGCC 中加入 dense auxiliary branch；
- 训练时加入 dense loss；
- 推理时保持原 APGCC 主分支输出不变。

从三数据集结果来看：

- BCData：明显有效；
- MoNuSeg：小幅有效；
- CoNIC：无效。

因此可以得到以下结论：

1. dense supervision 的确能够在部分病理细胞数据上提升 APGCC，尤其是在更容易出现高密度漏检的数据集上。
2. 这种改进并不是对所有数据集都稳定成立，说明其收益与数据分布、细胞密度和标注形态有关。
3. 与直接在推理阶段做 dense candidate fusion 相比，仅将 dense 分支作为训练辅助监督，更稳定、更干净，也更适合作为正式对比方案。

---

## 10. 本次工作的贡献总结

本次改进工作的实际贡献可以总结为三点：

1. 在 APGCC 中实现了一个可工作的 dense auxiliary supervision 版本，并完成了从代码到训练评测的完整闭环。
2. 系统验证了两条路线：
   - 训练辅助监督路线（dense loss-only）
   - 测试阶段补点路线（dense fusion）
3. 最终证明：
   - dense loss-only 是当前更稳健、更可汇报的改进；
   - fusion 路线在当前实现下不具备正式采用价值。

---
