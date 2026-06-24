# APGCC 三数据集 · 原生增强 vs 统一增强 对比报告

> 目的：在 BCData / MoNuSeg / CoNIC 三个病理细胞计数数据集上，对比 APGCC
> 在**自带原生增强（native-aug）**与**小组统一增强协议（unified-aug，并集版）**两套训练策略下的表现。
> 模型结构、输入尺寸、训练 epoch（200）完全一致，只改训练阶段的数据增强。
> 统一增强协议见 [`data_augmentation_protocol.md`](data_augmentation_protocol.md)。

## 评测口径

- **best.pth 选模**：在 `val.list` 上选最优；**所有汇报指标均在 `test.list`（测试集）上计算**。
- 计数指标（MAE / RMSE）与定位指标（Precision/Recall/F1 @ 6/12/24 px）由 `eval_centroid.py` + 仓库
  `centroid_eval.py` 统一计算，centroid 阈值 6/12/24 px。
- 效率指标由 `benchmark_efficiency.py` 计算（256² 输入，batch=1，RTX 3090，预热 10 / 测 100 次）。
- 测试集规模：BCData 402 张、MoNuSeg 14 张、CoNIC 991 张。

---

## 一、计数指标（测试集，越低越好）

| 数据集 | 增强 | MAE | RMSE | total_gt | total_pred | 误差% |
|--------|------|-----|------|----------|------------|-------|
| **BCData** | native | **18.27** | **23.73** | 65432 | 63031 | −3.67% |
| **BCData** | unified | 19.13 | 24.67 | 65432 | 61815 | −5.53% |
| **MoNuSeg** | native | 111.71 | 116.01 | 6697 | 8261 | +23.35% |
| **MoNuSeg** | unified | **94.36** | **98.90** | 6697 | 8018 | +19.73% |
| **CoNIC** | native | **11.99** | **19.74** | 112545 | 109084 | −3.08% |
| **CoNIC** | unified | 25.50 | 37.35 | 112545 | 88094 | −21.73% |

## 二、定位指标 F1（测试集，越高越好）

| 数据集 | 增强 | F1@6px | F1@12px | F1@24px |
|--------|------|--------|---------|---------|
| **BCData** | native | **0.7176** | **0.8145** | **0.8387** |
| **BCData** | unified | 0.7136 | 0.8125 | 0.8358 |
| **MoNuSeg** | native | 0.7361 | 0.8032 | 0.8291 |
| **MoNuSeg** | unified | **0.7489** | **0.8239** | **0.8462** |
| **CoNIC** | native | 0.6460 | 0.7944 | **0.8794** |
| **CoNIC** | unified | **0.6858** | 0.7888 | 0.8347 |

<details>
<summary>完整 Precision / Recall（点开展开）</summary>

| 数据集·增强 | 阈值 | Precision | Recall | F1 |
|---|---|---|---|---|
| BCData·native | 6/12/24 | 0.731/0.830/0.855 | 0.704/0.800/0.823 | 0.718/0.815/0.839 |
| BCData·unified | 6/12/24 | 0.734/0.836/0.860 | 0.694/0.790/0.813 | 0.714/0.813/0.836 |
| MoNuSeg·native | 6/12/24 | 0.666/0.727/0.751 | 0.822/0.897/0.926 | 0.736/0.803/0.829 |
| MoNuSeg·unified | 6/12/24 | 0.687/0.756/0.777 | 0.823/0.905/0.930 | 0.749/0.824/0.846 |
| CoNIC·native | 6/12/24 | 0.656/0.807/0.893 | 0.636/0.782/0.866 | 0.646/0.794/0.879 |
| CoNIC·unified | 6/12/24 | 0.781/0.898/0.951 | 0.611/0.703/0.744 | 0.686/0.789/0.835 |

</details>

## 三、效率指标（结构相同，三数据集一致）

| 项目 | 数值 |
|------|------|
| Params | 17.75 M |
| FLOPs | 40.72 G |
| Latency | 8.5–8.9 ms |
| Throughput | 112–118 FPS |
| 设备 / 环境 | RTX 3090，torch 2.4.1+cu121 |

> 模型结构在两套增强下完全相同，效率指标基本一致（latency 波动来自不同次测量）。

## 四、训练收敛（best epoch / val 指标）

| 数据集·增强 | best epoch | val MAE | val RMSE | 训练时长 |
|---|---|---|---|---|
| BCData·native | ep40 | 18.44 | 24.99 | 4:58:21 |
| BCData·unified | ep40 | 18.38 | 24.12 | 2:48:06 |
| MoNuSeg·native | ep190 | 69.43 | 80.69 | 0:18:05 |
| MoNuSeg·unified | ep180 | 64.00 | 79.06 | 0:11:44 |
| CoNIC·native | ep30 | 3.41 | 4.68 | ~4:10:00（ep63 手动停） |
| CoNIC·unified | ep65 | 3.86 | 5.14 | 10:49:28 |

> CoNIC·native 在 ep63 手动停止（best 已在 ep30，验证集已收敛），故未跑满 200 epoch，
> 时长为 ep0→ep63 墙钟（约 4h10m）；其余各项均跑满 200 epoch。

> 注 1：val 指标（上表）来自训练日志 `Best[ep K]`，是在验证集上；CoNIC/BCData 验证集为稀疏 overlap 子图，
> 故 val MAE 远低于测试集 MAE，属预期，不可与第一节测试集数字直接比较。
>
> 注 2：**命名约定**——APGCC 训练日志（`engine.py`）里打印的 "MSE" 实际是 **RMSE**（`np.sqrt(np.mean(squared_err))`，
> crowd-counting 惯例），故与 MAE 同数量级，上表列名已据此改为 val RMSE。真正的 MSE（未开方）见
> 第一节产物 `pred_centroid_eval.json` 的 `mse` 字段（如 BCData·native MSE=563.06，开方即 RMSE=23.73）。
> 第一节表中的 RMSE 列用的就是 `rmse` 字段，标注无误。

---

## 五、结论

1. **BCData**：两套增强几乎持平，原生略优（MAE 18.27 vs 19.13，F1 各阈值差异 <0.004）。
   该数据集本身均衡，增强策略影响很小。

2. **MoNuSeg**：**统一增强明显更好**。MAE 从 111.71 降到 94.36，RMSE 从 116.0 降到 98.9，
   F1 在 6/12/24 px 三个阈值全面提升，过计数从 +23.4% 收窄到 +19.7%。
   原因符合预期——MoNuSeg 训练集仅 30 张，统一协议的 affine + 翻转 + 像素增强显著缓解了小样本过拟合/过计数。

3. **CoNIC**：**统一增强反而显著拖累计数**。MAE 从 11.99 升到 25.50，由轻微欠计数（−3.1%）
   变成严重欠计数（−21.7%）；表现为 precision 升高、recall 大幅下降（漏检增多）。
   统一协议对 CoNIC 默认关闭 affine，但仍保留缩放/翻转/blur-noise/颜色扰动，
   推测过强的像素级增强 + 仅 200 epoch 使密集小核场景下模型趋于保守、漏检上升。
   定位上 F1@6px 反而更高（0.686 vs 0.646，定位更准），但 F1@24px 下降（0.835 vs 0.879），整体计数变差。

**小结**：统一增强对小样本数据集（MoNuSeg）收益明显，对均衡数据集（BCData）影响中性，
对密集大样本数据集（CoNIC）在当前 epoch 预算下有害。下一步建议：CoNIC 单独调弱像素增强 /
增加训练 epoch，再复核统一协议在 CoNIC 上的设置。

---

## 六、与其他方法对比

> 与组内其他方法（PET / CellVTA / STEERER / HoVer-Net）横向对比，定位指标统一取 **12px** 匹配阈值。
> **注意口径**：本节 `MSE` 列为**真·MSE（未开方，`mean((pred-gt)²)`）**，与 PET/CellVTA 一致，
> 取自 `pred_centroid_eval.json` 的 `mse` 字段；与第一节表的 `RMSE` 列（已开方）口径不同，勿混用。
> 空白单元格为对应方法尚未跑出的结果。

### 6.1 各模型官方增强版本

**计数指标（12px）**

| 数据集 | 方法 | MAE ↓ | MSE ↓ | Precision ↑ | Recall ↑ | F1 ↑ |
|:---|:---|---|---|---|---|---|
| **BCData** | APGCC | **18.27** | **563.06** | 0.830 | 0.800 | 0.8145 |
|  | STEERER |  |  |  |  |  |
|  | PET | 18.09 | 584.91 | 0.8259 | 0.8019 | 0.8137 |
| **CoNIC** | HoVer-Net |  |  |  |  |  |
|  | APGCC | 11.99 | 389.60 | 0.807 | 0.782 | 0.7944 |
|  | STEERER |  |  |  |  |  |
|  | PET | 37.20 | 3023.76 | 0.8583 | 0.5798 | 0.6921 |
|  | CellVTA | 2.7558 | 15.3073 | 0.8249 | 0.8035 | 0.8141 |
| **MoNuSeg** | HoVer-Net |  |  |  |  |  |
|  | APGCC | 111.71 | 13458.86 | 0.727 | 0.897 | 0.8032 |
|  | STEERER |  |  |  |  |  |
|  | PET | 82.64 | 8435.79 | 0.6603 | 0.7744 | 0.7128 |
|  | CellVTA | 13.1116 | 239.3616 | 0.5557 | 0.7676 | 0.6446 |

**效率指标**

| 方法 | 参数量 (M) | FLOPs (G) | 推理时间 (ms/image) | FPS | 输入尺寸 | 硬件 |
|:---|---|---|---|---|---|---|
| HoVer-Net |  |  |  |  |  |  |
| APGCC | 17.75 | 40.72 | 8.5–8.9 | 112–118 | 256×256 | RTX 3090 |
| STEERER |  |  |  |  |  |  |
| PET | 20.9094 | 57.9185–63.8453 | 11.2282–14.9204 | 67.0223–89.0616 | 256×256 | RTX 4090 |
| CellVTA | 387.7511 | 732.2967 | 61.7097 | 16.20 | 256×256 | RTX 3090 |

### 6.2 统一增强版本

**计数指标（12px）**

| 数据集 | 方法 | MAE ↓ | MSE ↓ | Precision ↑ | Recall ↑ | F1 ↑ |
|:---|:---|---|---|---|---|---|
| **BCData** | APGCC | 19.13 | 608.67 | 0.836 | 0.790 | 0.8125 |
|  | STEERER |  |  |  |  |  |
|  | PET | 19.33 | 627.51 | 0.8062 | 0.7652 | 0.7852 |
| **CoNIC** | HoVer-Net |  |  |  |  |  |
|  | APGCC | 25.50 | 1395.13 | 0.898 | 0.703 | 0.7888 |
|  | STEERER |  |  |  |  |  |
|  | PET | 24.10 | 1176.24 | 0.8691 | 0.6910 | 0.7699 |
|  | CellVTA | 2.7248 | 15.2974 | 0.8137 | 0.7945 | 0.8040 |
| **MoNuSeg** | HoVer-Net |  |  |  |  |  |
|  | APGCC | 94.36 | 9781.07 | 0.756 | 0.905 | 0.8239 |
|  | STEERER |  |  |  |  |  |
|  | PET | 116.71 | 14967.00 | 0.6747 | 0.8393 | 0.7481 |
|  | CellVTA | 13.1116 | 239.3616 | 0.5557 | 0.7676 | 0.6446 |

**效率指标**

| 方法 | 参数量 (M) | FLOPs (G) | 推理时间 (ms/image) | FPS | 输入尺寸 | 硬件 |
|:---|---|---|---|---|---|---|
| HoVer-Net |  |  |  |  |  |  |
| APGCC | 17.75 | 40.72 | 8.5–8.9 | 112–118 | 256×256 | RTX 3090 |
| STEERER |  |  |  |  |  |  |
| PET | 20.9094 | 57.9185–60.2892 | 11.0451–14.6515 | 68.2525–90.5379 | 256×256 | RTX 4090 |
| CellVTA | 387.7507 | 732.2548 | 61.8845 | 16.16 | 256×256 | RTX 3090 |

> 待补：STEERER（全部）、HoVer-Net（CoNIC/MoNuSeg）。CellVTA 在 MoNuSeg 两个增强版本数值相同
> （13.1116 / 239.3616），疑为占位/复制，需对方确认。

---

## 复现命令

```bash
conda activate apgcc
cd baselines/APGCC/apgcc
PY=/home/lixinli/anaconda3/envs/apgcc/bin/python

# 以 BCData·unified 为例，其余数据集替换 config / weight / data-root
$PY eval_centroid.py --config ./configs/BCData_unified.yml \
  --weight ./output/BCData_unified/best.pth \
  --data-root /data1/llx/BCData --gpu 3 \
  --out-dir ./output/BCData_unified/centroid_eval
$PY benchmark_efficiency.py --config ./configs/BCData_unified.yml \
  --weight ./output/BCData_unified/best.pth --gpu 3 \
  --out ./output/efficiency_apgcc_unified_BCData.json
```

数据根路径：BCData `/data1/llx/BCData`、MoNuSeg `/data1/llx/MoNuSegdata`、CoNIC `/data1/llx/CoNICdata`。

## 产物清单

- 计数/定位明细：`output/{dataset}_{finetune|unified}/centroid_eval/pred_centroid_eval.json`
- 效率：`output/efficiency_apgcc.json`（native CoNIC）、`output/efficiency_apgcc_unified*.json`
- 权重：`output/{dataset}_{finetune|unified}/best.pth`（`_finetune` = 原生增强，`_unified` = 统一增强）
