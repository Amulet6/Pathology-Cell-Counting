# APGCC 复现记录 · ShanghaiTech Part A

> 目的：验证 APGCC 模型在本地环境跑通，且指标与论文一致（用官方预训练权重做**指标验证复现**，非从头训练）。

## 结论

✅ 复现成功。用官方权重 `SHHA_best.pth` 在 ShanghaiTech Part A 测试集（182 张）上推理，
计数指标 **MAE 48.73 ≈ 论文 48.7**，模型运行无误。

## 指标对比

| 指标 | 复现值 | 论文 (APGCC, ECCV 2024) |
|------|--------|------------------------|
| **MAE** | **48.73** | 48.7 |
| **MSE** | **76.74** | ~80.0 |
| Localization Prec/Recall/F1 (σ=4) | 0.4387 / 0.4279 / 0.4332 | — |
| Localization Prec/Recall/F1 (σ=8) | 0.7732 / 0.7541 / 0.7635 | — |

- 推理耗时：182 张图约 18–20s（单卡 GPU）
- 多次运行结果完全一致（MAE=48.725275），确定性推理，无随机性问题

## 实验配置

- 环境：conda env `apgcc`（torch 2.4.1+cu121）
- 权重：`baselines/APGCC/apgcc/output/SHHA_best.pth`（官方提供）
- 数据集：`/data1/llx/part_A_final/`（原始 ShanghaiTech Part A，300 train / 182 test）
- 转换后数据：`/data1/llx/part_A_apgcc/`（APGCC 所需 `.list` + 逐图 `.txt`，绝对路径）
- 配置文件：`apgcc/configs/SHHA_test.yml`（`DATA_ROOT` 指向上面转换后路径）

## 复现命令

```bash
conda activate apgcc
cd baselines/APGCC/apgcc
# 1) 一次性：原始 .mat 标注 -> APGCC list/txt 格式
python datasets/prepare_shha_local.py /data1/llx/part_A_final /data1/llx/part_A_apgcc
# 2) 测试（指标验证）
python main.py -t -c ./configs/SHHA_test.yml \
  TEST.WEIGHT './output/SHHA_best.pth' OUTPUT_DIR ./output/ TEST.THRESHOLD 0.5
```

## 为打通流程所做的改动

1. `datasets/prepare_shha_local.py`（新增）：适配 `part_A_final/{train,test}_data` 目录结构的数据转换脚本
   （自带的 `prepare_label.py` 假设目录名为 `part_A/{train,test}`，与本地下载结构不符）。
2. `configs/SHHA_test.yml`：`DATASETS.DATA_ROOT` 改为 `/data1/llx/part_A_apgcc`。
3. `models/backbones/vgg.py`：VGG16 backbone 原硬编码不存在的本地 ImageNet 权重路径
   `/mnt/191/c/torch/...`，改为「本地有则用、否则从官方 URL 下载并缓存」。
   对测试无影响（被 checkpoint 覆盖），但从头训练时必需。

## 备注

- 本次为**指标验证复现**（用官方权重测试），用于确认模型与数据 pipeline 正确。
- **从头训练复现**（`EPOCHS: 3500`，单卡约十几小时）尚未进行，如汇报需要再行启动。
