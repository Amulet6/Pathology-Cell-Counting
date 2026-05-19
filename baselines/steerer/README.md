# STEERER 复现与细胞计数适配

本目录用于存放本人负责的 STEERER 模型复现、病理细胞数据集适配、评估与可视化代码。

## 方法定位

STEERER 属于密度图回归类计数与定位方法。本项目中计划将其适配到 BCData、CoNIC、MoNuSeg 等病理细胞数据集，用于完成细胞计数、点定位评估和密度图可视化。

## 目录说明

- official/: 官方 STEERER 代码或官方仓库说明
- src/data/: 数据集转换、预处理脚本
- src/configs/: 细胞数据集相关配置文件
- src/tools/: 评估、可视化和统计工具
- src/experiments/: 实验启动脚本和记录
- docs/: 复现计划、问题记录和实验总结

