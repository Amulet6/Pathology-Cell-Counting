# 病理图像细胞计数阶段报告（截至 2026-05-23）

## 一、阶段性总体进展
目前已完成 `CellViT++` 方法的基础复现准备，并在首个数据集 `MoNuSeg` 上完成了完整推理评测流程。现阶段已实现：

1. `CellViT++` 仓库部署、环境搭建与关键依赖修复。
2. 官方模型权重下载、校验与加载验证。
3. `MoNuSeg` 数据集格式转换与尺寸兼容处理。
4. `CellViT-256` 与 `CellViT-SAM-H` 两个 backbone 在 `MoNuSeg` 上的完整推理评测。
5. 下一个数据集可行性排查，包括 `BCData` 与 `CoNIC`。

## 二、已完成内容

### 1. 仓库与环境
已完成以下基础准备：

- 项目目录：`pathology-cell-counting`
- 复现仓库：`repos/CellViT-plus-plus`
- 主要环境：`envs/cellvitpp-py310`

当前只对 `cellvitpp-py310` 环境进行了修改，未动其他环境。

### 2. 环境安装与修复
已完成的关键依赖安装或修复包括：

- `torch==2.2.2`
- `torchvision==0.17.2`
- `torchaudio==2.2.2`
- `cupy==13.3.0`
- `cucim==24.04.00`
- `ray==2.9.3`
- `pathopatch==1.0.2`
- `gdown`
- `pandarallel==1.6.5`
- `h5py`

同时修复了以下关键环境问题：

- `pyvips/libvips` 相关动态库符号与软链接问题。
- `glib` 版本兼容问题。
- `scipy / scikit-learn / albumentations / qudida` 的 ABI 冲突问题。

目前 `CellViT++` 的基础导入、数据读取、模型加载与 GPU 前向均已正常。

## 三、模型权重进展

### 1. 已成功获取并验证的权重
已成功下载并验证以下官方权重：

- `downloads/gdrive_test/HIPT-256/CellViT-256-x40-AMP.pth`
- `downloads/gdrive_retry/SAM/CellViT-SAM-H-x40-AMP.redownload.pth`

两者均已通过 `torch.load` 与 `load_state_dict` 验证，可以正常构建模型。

### 2. 参数规模
- `CellViT-256`：`46,750,349` 参数
- `CellViT-SAM-H`：`699,741,149` 参数

### 3. 异常文件说明
以下旧文件下载损坏，但未删除：

- `downloads/gdrive_test/SAM/CellViT-SAM-H-x40-AMP.pth`

为避免覆盖原文件，后续重新下载到了新文件名：

- `downloads/gdrive_retry/SAM/CellViT-SAM-H-x40-AMP.redownload.pth`

## 四、MoNuSeg 数据集复现进展

### 1. 原始数据情况
已存在 `MoNuSeg` 原始数据，包括：

- `Tissue Images/*.tif`
- `Annotations/*.xml`

### 2. 已完成的数据处理
已编写并运行数据转换脚本，将 `MoNuSeg` 原始数据转换为 `CellViT++` 可读格式：

- 转换脚本：`scripts/prepare_monuseg_cellvitpp.py`
- 输出目录：`datasets/processed/monuseg_full`

转换后格式为：

- `images/*.png`
- `labels/*.npy`

### 3. 尺寸兼容处理
由于 `MoNuSeg` 原图为 `1000x1000`，而 `CellViT++` 的该推理入口需要按 `256` patch 切块，因此已额外构建补边版本数据集：

- `datasets/processed/monuseg_full_1024`

处理方式：

- 图像从 `1000x1000` padding 到 `1024x1024`
- 图像补白值为 `255`
- mask 补零

原始转换结果与补边结果都保留，未覆盖。

## 五、MoNuSeg 推理评测结果

### 1. CellViT-256 结果
输出目录：

- `results/monuseg_cellvit256_1024`

指标结果：

- `Binary-Cell-Dice-Mean`: `0.8457346558570862`
- `Binary-Cell-Jacard-Mean`: `0.7339013814926147`
- `bPQ`: `0.6109450873748368`
- `bDQ`: `0.791508080068951`
- `bSQ`: `0.7707229987910562`
- `f1_detection`: `0.819227559447764`
- `precision_detection`: `0.7999801836394894`
- `recall_detection`: `0.8445831230805627`

### 2. CellViT-SAM-H 结果
输出目录：

- `results/monuseg_cellvitsam_1024`

指标结果：

- `Binary-Cell-Dice-Mean`: `0.847751796245575`
- `Binary-Cell-Jacard-Mean`: `0.7370210886001587`
- `bPQ`: `0.6295003134398981`
- `bDQ`: `0.8103149158542259`
- `bSQ`: `0.7758975520299387`
- `f1_detection`: `0.8307815393977582`
- `precision_detection`: `0.8190574087976177`
- `recall_detection`: `0.847704875317994`

### 3. 当前结论
在 `MoNuSeg` 数据集上：

- `CellViT-SAM-H` 效果优于 `CellViT-256`
- 两个模型均已成功完成完整推理评测
- 当前 `CellViT-SAM-H` 是该方法内部更优的 backbone 版本

## 六、BCData 与 CoNIC 当前排查情况

### 1. BCData
已检查 `BCData.zip` 的内部结构，确认：

- 标注文件为 `.h5`
- 每个标注文件仅包含一个字段：`coordinates`
- 即 `BCData` 是点标注数据集，不是实例分割 mask 数据集

结论：

- `BCData` 更适合点监督或定位匹配类方法
- 不适合作为 `CellViT++` 的原生实例分割输入直接复用
- 若强行接入 `CellViT++`，需要额外设计从点标注到伪 mask 或检测格式的转换逻辑

### 2. CoNIC
当前 `datasets/CoNIC` 目录为空。

同时已排查下载条件：

- `CoNIC` 官方入口需要通过 `Grand Challenge` 获取下载权限
- 可检索到 `Kaggle` 镜像入口，但当前环境中：
  - 未安装 `kaggle` 客户端
  - 无 `~/.kaggle/kaggle.json` 凭据

结论：

- `CoNIC` 当前尚未落地
- 后续若继续复现，需要先解决下载权限或提供 `Kaggle API` 凭据

## 七、目前已完成与未完成的边界

### 已完成
- `CellViT++` 环境搭建
- 官方权重下载与验证
- `MoNuSeg` 数据适配
- `MoNuSeg` 上 `CellViT-256` 与 `CellViT-SAM-H` 的完整推理评测
- `BCData` 与 `CoNIC` 的初步数据结构排查

### 未完成
- `BCData` 的适配与复现
- `CoNIC` 的下载与适配
- 三个数据集的统一对比表
- 改进方法设计与改进前后对比实验
- 效率指标的统一整理，包括推理时间、FLOPs、参数量
- 可视化结果与汇报图片整理

## 八、下一步建议
优先建议如下：

1. 先解决 `CoNIC` 数据下载问题。
2. 若短期无法拿到 `CoNIC`，可再评估是否对 `BCData` 做点标注到检测格式的转换。
3. 在 `CellViT++` 已跑通的基础上，继续整理 `MoNuSeg` 结果表格与可视化，为阶段汇报提供现成材料。
4. 后续若以综合效果最优模型为基础做改进，优先以 `CellViT-SAM-H` 为基础进行。

## 九、补充说明
- 所有工作均限制在项目目录 `pathology-cell-counting` 下完成。
- 未删除任何已有文件。
- 对损坏的旧权重文件仅保留，不覆盖、不删除，改为新文件名重新下载。
