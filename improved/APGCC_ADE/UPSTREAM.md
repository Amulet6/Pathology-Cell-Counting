# 官方代码溯源 (Upstream)

- 仓库: https://github.com/AaronCIH/APGCC
- 起始 commit: `e3e997b` (Update README.md)
- 拉取日期: 2026-06-07

## 本基线的工作方式

本基线**直接在官方代码上修改**（代码已平铺在 `baselines/APGCC/` 下并纳入 git 追踪），
不保留独立的 `official/` 干净副本。如需对照原版，可访问上述官方仓库或 checkout 起始 commit。

针对病理细胞数据的主要改动（持续更新）：
- [ ] 数据加载/转换：BCData / CoNIC / MoNuSeg → APGCC 点格式 (x y) + list 文件
- [ ] configs：新增三数据集配置
- [ ] 训练/评测：对接全队统一评测（MAE/MSE/P/R/F1 + 参数量/FLOPs/推理时间）
