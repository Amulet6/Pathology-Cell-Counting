# APGCC 三数据集复现说明（BCData / MoNuSeg / CoNIC）

本文档说明如何在一台新服务器上，从零复现我们基于 **APGCC**（点检测计数模型）在三个病理细胞数据集
（**BCData / MoNuSeg / CoNIC**）上的「细胞计数 + 定位」实验，包含 **native 增强** 与 **团队统一增强(unified)** 两条对比路线。

模型本体沿用 APGCC 原始结构（VGG16-bn 编码器 + IFI 解码器），适配工作集中在：
**数据转点标注格式 → 各数据集独立配置 → 增强协议可切换 → 评测改为 P/R/F1 + 计数误差**。

---

## 0. 前置要求

- Linux + NVIDIA GPU（显存 ≥ 8 GB，单卡即可）
- 已安装 CUDA 驱动、`conda`、`git`
- 磁盘：原始数据 + 转换后数据约需 10–20 GB

所有命令默认在仓库的 **`baselines/apgcc/apgcc/`** 目录下执行（配置文件里用的是 `./output`、`./configs` 等相对路径，且脚本 `import config`，必须在该目录运行）。

---

## 1. 拉取代码与创建环境

```bash
git clone <你的仓库地址> Pathology-Cell-Counting
cd Pathology-Cell-Counting/baselines/apgcc

conda create -n apgcc python=3.8 -y
conda activate apgcc

# PyTorch 请按服务器 CUDA 版本安装（示例为 CUDA 11.x）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt        # tensorboardX easydict pandas numpy scipy matplotlib Pillow opencv-python
pip install h5py                        # BCData 读取 HDF5 需要
```

---

## 2. 下载 APGCC 的 SHHA 预训练权重（必需）

三个数据集都从 APGCC 官方 ShanghaiTech-A 权重 **微调**，需先放好该权重：

```bash
cd apgcc
mkdir -p output
wget --no-check-certificate \
  'https://docs.google.com/uc?export=download&id=1pEvn5RrvmDqVJUDZ4c9-rCJcl2I7bRhu' \
  -O ./output/SHHA_best.pth
```

> 若 wget 受限，可手动从 APGCC 官方仓库的 Google Drive 链接下载，重命名为 `SHHA_best.pth` 放到 `apgcc/output/` 下。
> 配置文件中 `RESUME_PATH: ./output/SHHA_best.pth` 即指向此文件。

---

## 3. 下载原始数据集

分别从官方渠道下载，放到任意目录（下文用 `<原始数据目录>` 占位）：

| 数据集 | 官方来源 | 需要的内容 |
|---|---|---|
| **BCData** | 乳腺癌细胞检测数据集 | `images/{train,validation,test}/`、`annotations/{...}/{positive,negative}/*.h5` |
| **MoNuSeg 2018** | monuseg.grand-challenge.org | `MoNuSeg 2018 Training Data/`（Tissue Images + Annotations）、`MoNuSegTestData/` |
| **CoNIC** | CoNIC Challenge | `images.npy`、`labels.npy`、`patch_info.csv` |

---

## 4. 转换为 APGCC 点标注格式

APGCC 需要 **`*.list` 索引 + 每张图一个「x y」坐标 txt** 的格式。下面三个脚本会把原始标注
（HDF5 坐标 / XML 多边形质心 / npy 实例质心）统一转成该格式，并生成 `train/val/test` 切分。
**请把 `<转换输出目录>` 设为一个新目录，它就是后面配置里的 `DATA_ROOT`。**

```bash
cd apgcc          # 始终在此目录运行

# --- BCData ---（正/负坐标合并为点；3 路切分）
python datasets/prepare_bcdata.py  <BCData原始目录>
# 注意：BCData 脚本把转换结果直接写在它的原始目录下（生成 train/ val/ test/ 与 *.list）

# --- MoNuSeg ---（多边形顶点均值=质心；按 monuseg_split.json 做 30/7/14 切分）
python datasets/prepare_monuseg.py  <MoNuSeg转换输出目录> \
    --train-img "<.../MoNuSeg 2018 Training Data/Tissue Images>" \
    --train-ann "<.../MoNuSeg 2018 Training Data/Annotations>" \
    --test-dir  "<.../MoNuSegTestData>"

# --- CoNIC ---（实例图像素均值=质心；按 patch_info.csv 源图切分）
python datasets/prepare_conic.py  <CoNIC转换输出目录>  --val-ratio 0.2 --seed 42
# 需保证 <CoNIC转换输出目录>/data/ 下有 images.npy / labels.npy / patch_info.csv
```

转换完成后，每个 `DATA_ROOT` 下应有：

```
DATA_ROOT/
  train/  val/  test/            # 图片（多为软链接）
  train_gt/  val_gt/  test_gt/   # 每图一个 .txt，每行 "x y" 一个细胞质心
  train.list  val.list  test.list  # 每行: "<split>/<id>.png <split>_gt/<id>.txt"
```

> `train.list` 用于训练；`val.list` 用于训练中选 best 权重；`test.list` 是最终留出测试集。

---

## 5. 修改配置文件

每个数据集有两份配置，区别只在**增强协议**：

| 路线 | 配置文件 | 增强 |
|---|---|---|
| native（原 SHHA 增强） | `configs/BCData_finetune.yml` 等 `*_finetune.yml` | random scale + crop + flip |
| unified（团队统一增强） | `configs/BCData_unified.yml` 等 `*_unified.yml` | scale→affine→crop→flip→blur/noise→color jitter |

**复现前必须改两处**（用你自己的路径/显卡）：

```yaml
GPU_ID: 0                              # 改成你的空闲卡号
DATASETS:
  DATA_ROOT: /your/path/to/BCData      # 改成第 4 步的转换输出目录
```

其余关键参数已按数据集调好，无需改动（供参考）：

| 参数 | BCData | MoNuSeg | CoNIC |
|---|---|---|---|
| `CROP_SIZE` | 256 | 256 | 128 |
| `UPPER_BOUNDER` | 1024 | -1 | -1 |
| `BATCH_SIZE` | 8 | 8 | 16 |
| `UNIFIED_AFFINE` | true | true | false |

---

## 6. 训练

在 `apgcc/` 目录下，按数据集 × 增强路线选择对应配置：

```bash
# 统一增强（unified）
python main.py -c ./configs/BCData_unified.yml
python main.py -c ./configs/MoNuSeg_unified.yml
python main.py -c ./configs/CoNIC_unified.yml

# 原生增强（native，对比组）
python main.py -c ./configs/BCData_finetune.yml
python main.py -c ./configs/MoNuSeg_finetune.yml
python main.py -c ./configs/CoNIC_finetune.yml
```

也可在命令行临时覆盖配置项（无需改 yml），例如换卡和数据路径：

```bash
python main.py -c ./configs/CoNIC_unified.yml  GPU_ID 1  DATASETS.DATA_ROOT /your/CoNICdata
```

训练产物（日志、配置副本、`best.pth`）保存在配置里的 `OUTPUT_DIR`（如 `./output/BCData_unified/`）。

---

## 7. 评测（计数误差 + 定位 P/R/F1）

使用 `eval_centroid.py` 在 **test.list** 上评测，输出 MAE/MSE/RMSE 与各距离阈值下的 Precision/Recall/F1：

```bash
python eval_centroid.py \
  --config   ./configs/BCData_unified.yml \
  --weight   ./output/BCData_unified/best.pth \
  --data-root /your/path/to/BCData \
  --eval-list test.list \
  --score-threshold 0.5 \
  --thresholds 6 12 24 \
  --out-dir  ./output/BCData_unified/centroid_eval \
  --gpu 0
```

- `--thresholds`：定位匹配的像素距离阈值（默认 6/12/24 px）。
- 结果写入 `--out-dir`：`gt.json` / `pred.json`，并打印汇总指标。
- 其它数据集把 `--config / --weight / --data-root / --out-dir` 换成对应路径即可。

---

## 8. 复现对照

- 各数据集 native vs unified 的指标、效率、收敛对比见 **`../REPORT_aug_comparison.md`**。
- 统一增强协议的完整定义见 **`../data_augmentation_protocol.md`**（实现：`datasets/aug_unified.py`）。
- 随机种子固定为 `SEED: 1229`；为提速 cuDNN 用 benchmark 模式（非 bit-exact 确定性），同卡同环境下指标应在很小波动内可复现。

---

## 9. 常见问题

- **必须在 `apgcc/` 目录运行**：脚本 `import config`、配置用相对路径，换目录会 import 失败或找不到权重。
- **`SHHA_best.pth` 找不到**：检查是否放在 `apgcc/output/SHHA_best.pth`，与配置 `RESUME_PATH` 一致。
- **显存不足**：调小 `SOLVER.BATCH_SIZE` 或 `DATALOADER.CROP_NUMBER`。
- **空 patch / 无细胞图**：已支持（计数目标为 0），属正常，不是 bug。
- **多卡**：训练用单卡，通过 `GPU_ID` 指定卡号即可，不需要分布式。
