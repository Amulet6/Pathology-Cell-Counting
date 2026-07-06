# APGCC 方向 B + D 叠加改动说明

## 1. 改动目标

本次改动是在方向 D 代码的基础上，叠加方向 B 的最终方案：

- 方向 D：`DCNv2 + Edge Ignore`
- 方向 B：`dense loss-only`

最终目标是：

- 保留方向 D 的几何增强与边界抑制能力；
- 额外加入一个密集辅助分支 `dense_head`；
- 仅在训练阶段使用 `loss_dense`；
- 测试阶段仍保持原有 APGCC 输出，不做 dense 融合。

---

## 2. 新增内容

### 2.1 Dense Auxiliary Branch

在 `apgcc/models/APGCC.py` 中，为 `Model_builder` 增加了一个密集辅助分支：

- 从编码器中间层特征提取 `dense_map`
- 使用一个轻量卷积头 `dense_head`
- 输出单通道密集响应图

该分支只用于训练监督，不参与最终点预测。

### 2.2 Dense Loss

新增 `loss_dense`：

- 用现有点标注生成高斯密集监督图
- 对 `dense_head` 输出做 `sigmoid`
- 与目标密集图计算 MSE loss

总损失由原来的点级损失扩展为：

- `loss_ce`
- `loss_points`
- `loss_aux`（若开启）
- `loss_dense`

---

## 3. 代码修改点

### 3.1 `apgcc/models/APGCC.py`

主要改动：

- 新增 `dense_aux_en`
- 新增 `dense_aux_level`
- 新增 `dense_sigma`
- 新增 `dense_head`
- 新增 `loss_dense`
- 在 `forward()` 中写入 `dense_map` 和 `img_shape`
- 在 `forward()` 的 loss 计算中支持 `loss_dense`

### 3.2 `apgcc/models/__init__.py`

主要改动：

- 构建 `criterion` 时新增 `dense_aux_en` 和 `dense_sigma`
- 当 `DENSE_AUX_EN=False` 时自动移除 `loss_dense`
- 复制 `weight_dict`，避免直接修改全局配置对象

### 3.3 `apgcc/config.py`

主要改动：

- 增加默认配置项：
  - `MODEL.DENSE_AUX_EN`
  - `MODEL.DENSE_AUX_LEVEL`
  - `MODEL.DENSE_SIGMA`
- 给 `WEIGHT_DICT` 增加默认 `loss_dense`

### 3.4 `apgcc/engine.py`

主要改动：

- 加载 checkpoint 时改为 `strict=False`
- 兼容新增的 `dense_head`
- 保持方向 D 的 `DCNv2` / `Edge Ignore` 逻辑不变

### 3.5 `apgcc/main.py`

主要改动：

- 测试阶段加载权重时改为 `strict=False`

### 3.6 `apgcc/eval_centroid.py`

主要改动：

- 测试阶段加载权重时改为 `strict=False`

### 3.7 `apgcc/benchmark_efficiency.py`

主要改动：

- 同步支持 `strict=False` 的权重加载

---

## 4. 新增配置

新增了三份 dense-only 配置：

- `apgcc/configs/BCData_finetune_dcnv2_edge_dense.yml`
- `apgcc/configs/MoNuSeg_finetune_dcnv2_edge_dense.yml`
- `apgcc/configs/CoNIC_finetune_dcnv2_edge_dense.yml`

这些配置在方向 D 的基础上额外加入：

- `DENSE_AUX_EN: True`
- `DENSE_AUX_LEVEL: 3`
- `DENSE_SIGMA: 2.0`
- `WEIGHT_DICT` 中增加 `loss_dense: 0.1`

其余方向 D 参数保持不变：

- `DCNV2_ENABLED: True`
- `DCNV2_LAYERS: 2`
- `EDGE_BAND: 16`
- `EDGE_WEIGHT: 0.1`

---

## 5. 训练与评测行为

### 5.1 训练阶段

训练时：

- 主分支继续优化点分类与点位置回归；
- `dense_head` 额外接收 `loss_dense` 监督；
- 方向 D 的 `DCNv2` 和 `Edge Ignore` 继续生效。

### 5.2 测试阶段

测试时：

- 仍然只使用 APGCC 原始点预测；
- 不做 dense candidate fusion；
- 即保持 `dense loss-only`。

---

## 6. 兼容性

本次改动保持向后兼容：

- 旧 checkpoint 可加载；
- 不开启 `DENSE_AUX_EN` 时行为等同于方向 D；
- 不改方向 D 的 DCNv2 / Edge Ignore 实现；
- 只新增一条训练监督分支，不改变最终预测接口。

---

## 7. 使用方式

直接使用新增的 dense-only 配置即可。

例如 MoNuSeg：

```bash
cd apgcc
python main.py -c ./configs/MoNuSeg_finetune_dcnv2_edge_dense.yml \
  GPU_ID 0 \
  DATASETS.DATA_ROOT <MoNuSeg data root> \
  RESUME_PATH <SHHA_best.pth> \
  OUTPUT_DIR ./output/MoNuSeg_finetune_dcnv2_edge_dense/
```

评测时仍使用 `eval_centroid.py`，并保持 `test.list` 作为最终评测集。

---

## 8. 一句话总结

这次改动的本质是：

> 在方向 D 的 APGCC 上，额外加入一个只用于训练的 dense auxiliary branch，用 `loss_dense` 增强密集小细胞区域的监督，但不修改最终测试阶段的出点逻辑。
