# 病理细胞计数 — CellVTA 复现 & APGCC 方向 B 改进

本目录整理了两部分课程设计工作，便于单独推送到 GitHub：

| 子目录 | 内容 |
|---|---|
| [`cellvta/`](cellvta/) | CellVTA 在 BCData / MoNuSeg / CoNIC 上的复现实验 |
| [`apgcc_direction_b/`](apgcc_direction_b/) | APGCC 方向 B：Count Anything 式 dense loss-only 改进 |

## 不包含的内容

以下内容体积较大或涉及数据许可，**未随仓库上传**，复现时需自行准备：

- 原始数据集（BCData / MoNuSeg / CoNIC）
- 预训练权重（CellVTA checkpoint、APGCC `SHHA_best.pth`、训练 best.pth）
- conda / Python 环境

## 快速导航

### CellVTA 复现

```bash
cd cellvta/code/repos/CellVTA
# 参考 cellvta/manifest/README.md 与 cellvta/scripts/ 下的脚本
```

主要结果见 [`cellvta/reports/cellvta_results_summary.md`](cellvta/reports/cellvta_results_summary.md)。

### APGCC 方向 B

```bash
cd apgcc_direction_b/apgcc
conda create -n apgcc python=3.8 -y && conda activate apgcc
pip install -r ../requirements.txt

# 下载 SHHA 预训练权重到 apgcc/output/SHHA_best.pth
python main.py -c ./configs/BCData_finetune.yml
```

详细说明见 [`apgcc_direction_b/docs/REPRODUCE.md`](apgcc_direction_b/docs/REPRODUCE.md) 与 [`apgcc_direction_b/docs/APGCC_direction_B_dense_loss_report.md`](apgcc_direction_b/docs/APGCC_direction_B_dense_loss_report.md)。

## 方向 B 核心结论

- 在 APGCC 编码器中间层新增 **dense auxiliary branch**，训练时用高斯密度图 MSE 损失（`loss_dense=0.1`）辅助监督
- 测试阶段**完全移除** dense 分支，保持 APGCC 原始推理接口
- BCData 上收益最明确（MAE 18.27→17.51）；MoNuSeg 小幅提升；CoNIC 计数误差恶化
- dense 候选融合方案不稳定，最终采用 **dense loss-only**

## 目录结构

```
liyixin/
├── README.md
├── .gitignore
├── cellvta/
│   ├── code/          # CellVTA 源码与配置
│   ├── scripts/       # 数据准备、评测、效率 benchmark 脚本
│   ├── share/         # 统一划分、增强协议、评估工具
│   ├── logs/          # 关键训练/推理日志
│   ├── reports/       # 指标汇总与效率结果
│   └── manifest/      # 交付说明
└── apgcc_direction_b/
    ├── apgcc/         # 含 dense 分支的 APGCC 代码
    ├── docs/          # 复现说明、改动文档、实验报告
    ├── results/       # 实验日志与评测 JSON（不含权重）
    ├── requirements.txt
    ├── data_augmentation_protocol.md
    └── LICENSE
```
