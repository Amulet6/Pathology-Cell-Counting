# STEERER 病理细胞计数与定位复现

本目录由 **suqiseven** 维护，基于 ICCV 2023 论文 [STEERER: Resolving Scale Variations for Counting and Localization via Selective Inheritance Learning](https://arxiv.org/abs/2308.10468) 及其[官方实现](https://github.com/taohan10200/STEERER)适配。模型属于**密度图回归**方法；本复现增加了 BCData、CoNIC、MoNuSeg 的数据转换、训练配置、点定位评估和效率测试。

> 数据集、预训练权重、训练 checkpoint 和运行日志不纳入 Git。

## 1. 方法与代码改动

STEERER 使用 HRNet-W48 保留多分辨率特征，通过选择性继承机制让不同尺度分支学习对应尺度的密度响应。预测密度图积分用于计数，密度峰值经过后处理可转换为细胞中心点。

相对官方代码，本目录主要增加：

- `lib/datasets/{bcdata,conic,monuseg}.py`：三个病理数据集加载器；
- `tools/convert_*_to_steerer.py`：三种原始标注到 STEERER 点标注格式的转换器；
- `configs/{BCData,CoNIC,MoNuSeg}_*.py`：训练、测试和迁移微调配置；
- `tools/eval_bcdata_points.py`：一对一点匹配的 Precision、Recall、F1 和 MLE 评估；
- `tools/profile_bcdata.py`：参数量、FLOPs、延迟和 FPS 测量；
- `tools/augmentation/`：小组统一增强协议、格式转换和评估工具；
- `run_conic_monuseg_pipeline.sh`：CoNIC 训练、MoNuSeg 迁移微调和测试流水线。

## 2. 环境

复现实验使用 Python 3.9、PyTorch 1.12.0、torchvision 0.13.0 和 CUDA 11.3：

```bash
conda create -n steerer python=3.9 -y
conda activate steerer
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch
cd Pathology-Cell-Counting/baselines/steerer
pip install -r requirements.txt
pip install -r requirements_aug.txt   # 仅统一增强/效率评估需要
```

若使用新 CUDA/PyTorch，请先执行第 6 节的 smoke test；旧版 MMCV、Timm 与新版 PyTorch 组合可能不兼容。

## 3. 目录结构

所有命令均从 `baselines/steerer` 执行。配置使用 `../ProcessedData` 和 `../PretrainedModels`，推荐结构如下：

```text
Pathology-Cell-Counting/
└── baselines/
    ├── steerer/
    ├── ProcessedData/                   # 不提交到 Git
    │   ├── BCData_numeric/
    │   ├── CoNIC_numeric/
    │   └── MoNuSeg_numeric/
    └── PretrainedModels/                # 不提交到 Git
        └── hrnetv2_w48_imagenet_pretrained.pth
```

每个转换后数据集采用：

```text
<dataset>/
├── images/                 # 000000.png 等图像
├── jsons/                  # points、boxes、human_num 等训练标注
├── train.txt
├── val.txt
├── test.txt
├── train_gt_loc.txt
├── val_gt_loc.txt
└── test_gt_loc.txt
```

`*_gt_loc.txt` 每行格式为：

```text
image_id num_points x y box_w box_h scale_category ...
```

## 4. 数据预处理

### 4.1 BCData

HDF5 标注包含 positive/negative 两组坐标。默认将两组均作为待计数细胞，并保留原图尺寸：

```bash
python tools/convert_bcdata_to_steerer.py \
  --input /path/to/BCData.zip \
  --output ../ProcessedData/BCData_numeric \
  --count-mode all
```

可用 `--count-mode positive` 或 `negative` 只选一类。转换器支持 zip 或解压目录。

### 4.2 CoNIC

每个非零 instance ID 转换为一个质心。若输入无官方拆分，脚本以种子 `3035` 按 70%/10%/20% 划分：

```bash
python tools/convert_conic_to_steerer.py \
  --input /path/to/CoNIC \
  --output ../ProcessedData/CoNIC_numeric \
  --seed 3035
```

支持根目录 `images.npy`/`labels.npy`、按 split 放置的数组及显式 `--images`/`--labels`。与其他方法对比时必须传入相同 split，不能各自随机划分。

### 4.3 MoNuSeg

XML 多边形通过鞋带公式计算几何质心。脚本支持解压目录、单个 zip 或包含训练/测试 zip 的目录：

```bash
python tools/convert_monuseg_to_steerer.py \
  --input /path/to/MoNuSeg \
  --output ../ProcessedData/MoNuSeg_numeric \
  --seed 3035
```

正式转换前可加 `--limit-per-split 2` 做格式检查。转换后需检查图片与 JSON 数量一致、点坐标未越界、split 无重复样本。

### 4.4 统一增强（可选）

`tools/augmentation/` 实现跨 baseline 共用的增强协议，详见 [统一增强说明](tools/augmentation/README_unified_aug.md)。native 实验使用 STEERER 在线随机裁剪、多尺度和翻转；unified 实验使用预生成的 `640×640` 数据。

## 5. 训练、测试与评估

训练前检查配置中的 `dataset.root`、`network.pretrained_backbone`、`train.resume_path`、GPU、worker 和 batch size。首次训练必须将 `resume_path` 设为 `None`。

| 数据集 | 训练配置 | 测试配置 |
|---|---|---|
| BCData | `configs/BCData_train.py` | `configs/BCData_test.py` |
| CoNIC | `configs/CoNIC_train.py` | `configs/CoNIC_test.py` |
| MoNuSeg | `configs/MoNuSeg_train.py` 或 `configs/MoNuSeg_finetune_conic.py` | `configs/MoNuSeg_test.py` |

### 5.1 训练

```bash
# 第二个参数是 GPU ID
bash train.sh configs/BCData_train.py 0
bash train.sh configs/CoNIC_train.py 0
bash train.sh configs/MoNuSeg_train.py 0
```

训练输出写入 `exp/`，模型按验证集 MAE/RMSE 保存 checkpoint。MoNuSeg 小样本实验推荐从 CoNIC 最优权重微调；先在 `configs/MoNuSeg_finetune_conic.py` 设置恢复权重路径。

### 5.2 测试与预测点导出

```bash
bash test.sh configs/BCData_test.py /path/to/best_checkpoint.pth 0

# 定位入口在测试目录生成 pred_points.txt
python -m torch.distributed.launch --nproc_per_node=1 \
  tools/test_loc.py \
  --cfg configs/BCData_test.py \
  --checkpoint /path/to/best_checkpoint.pth \
  --launcher pytorch
```

当前复现环境保留 `torch.distributed.launch`。新版 PyTorch 可改用 `torchrun`，但须同步处理 `LOCAL_RANK`。

### 5.3 点定位统一评估

```bash
python tools/eval_bcdata_points.py \
  --gt ../ProcessedData/BCData_numeric/test_gt_loc.txt \
  --pred /path/to/pred_points.txt \
  --radii 6 12 24 \
  --output /path/to/location_metrics.json
```

每张图内一对一匹配：预测点与 GT 距离不超过指定半径才记为 TP，同一点不能重复匹配。主报告使用 12 px，并保留 6/24 px 分析尺度敏感性。

### 5.4 效率评估

```bash
python tools/profile_bcdata.py \
  --cfg configs/BCData_test.py \
  --checkpoint /path/to/best_checkpoint.pth \
  --height 640 --width 640 --warmup 20 --iters 100
```

GPU 延迟测量包含 warm-up，并在计时前后同步 CUDA。跨 baseline 比较须统一输入尺寸、batch size、warm-up、重复次数和硬件。

## 6. 快速自检

```bash
python -m compileall -q lib tools
python tools/convert_bcdata_to_steerer.py --help
python tools/convert_conic_to_steerer.py --help
python tools/convert_monuseg_to_steerer.py --help
python tools/eval_bcdata_points.py --help
```

随后用 `--limit-per-split 2` 生成微型数据并跑一个 DataLoader batch。错误点坐标或 split 可能产生看似正常、实际不可比较的结果。

## 7. 复现结果

### 7.1 官方实现核验

使用官方 SHHB checkpoint 得到 MAE 5.8344、RMSE 8.5252，与论文 5.8/8.5 基本一致。论文未报告三个病理数据集结果，所以下表属于病理任务迁移实验，不应与 SHHB 直接比较。

### 7.2 三数据集计数结果

采用统一中心点评估口径，对 `pred_points.txt` 中每张图的点数计算误差。这里 `MSE` 是平均平方误差，`RMSE = sqrt(MSE)`；STEERER 历史日志中名为 `MSE` 的字段通常实际表示 RMSE，引用时须区分。

| 数据集 | 设置 | MAE ↓ | MSE ↓ | RMSE ↓ |
|---|---:|---:|---:|---:|
| BCData | native | 22.41 | 878.15 | 29.63 |
| BCData | unified | **17.59** | **535.56** | **23.14** |
| CoNIC | native | **13.39** | 487.59 | 22.08 |
| CoNIC | unified | **13.39** | **487.58** | **22.08** |
| MoNuSeg | native | **81.79** | **9574.93** | **97.85** |
| MoNuSeg | unified | 100.43 | 15542.71 | 124.67 |

### 7.3 点定位结果（12 px）

| 数据集 | 设置 | Precision ↑ | Recall ↑ | F1 ↑ |
|---|---:|---:|---:|---:|
| BCData | native | 0.7889 | **0.8707** | **0.8278** |
| BCData | unified | **0.8297** | 0.8232 | 0.8265 |
| CoNIC | native | **0.8459** | **0.7879** | **0.8158** |
| CoNIC | unified | **0.8459** | **0.7879** | **0.8158** |
| MoNuSeg | native | 0.6630 | **0.7599** | 0.7081 |
| MoNuSeg | unified | **0.7090** | 0.7196 | **0.7142** |

CoNIC 两行使用同一 ep27 权重和同一 native 测试集，并非两次独立训练。BCData 统一增强降低计数误差但略降 Recall；MoNuSeg 统一增强提高 F1，却增大单图计数误差，说明计数与定位指标不能互相替代。

### 7.4 效率（RTX 4090，batch size 1）

| 输入 | 参数量 | FLOPs | 推理延迟 | FPS |
|---|---:|---:|---:|---:|
| 256×256 | 64.64 M | 46.79 G | 61.49–62.41 ms | 16.02–16.26 |
| 640×640 | 64.64 M | 292.46 G | 72.94–75.70 ms | 13.21–13.71 |

延迟仅含模型前向，不含全切片滑窗、磁盘读取和点提取后处理。

## 8. 指标定义与注意事项

设第 `i` 张图 GT 数量为 `g_i`，预测数量为 `p_i`，图像数为 `N`：

- `MAE = (1/N) * Σ |p_i - g_i|`
- `MSE = (1/N) * Σ (p_i - g_i)^2`
- `RMSE = sqrt(MSE)`
- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1 = 2 * Precision * Recall / (Precision + Recall)`

计数可由密度图积分或离散点数量得到。为与点定位、实例分割方法可比，本项目最终主表使用后者，两种口径不可混用。

常见问题：

- 找不到图片：确认 JSON 的 `img_id` 指向 `images/` 内真实文件，列表 ID 与 JSON 文件名一致；
- 找不到权重：修改 `network.pretrained_backbone` 或按第 3 节建立目录；
- 意外恢复旧实验：首次训练把 `train.resume_path` 设为 `None`；
- F1 异常：resize 后点坐标和匹配半径必须使用同一坐标系；
- 显存不足：优先降低 `batch_size_per_gpu`，改变输入尺寸后重新报告 FLOPs 和延迟。

## 9. 许可证与引用

本目录继承官方代码许可证，详见 [LICENSE](LICENSE)。使用时请引用：

```bibtex
@inproceedings{han2023steerer,
  title     = {STEERER: Resolving Scale Variations for Counting and Localization via Selective Inheritance Learning},
  author    = {Han, Tao and Bai, Lei and Liu, Lingbo and Ouyang, Wanli},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2023}
}
```
