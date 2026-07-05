# APGCC_E 最小集成指南

本文档只说明如何集成 APGCC_E 中最有用的两部分：

```text
1. stain/pathology augmentation 训练预处理
2. DomainTh-MAE 后处理阈值校准
```

不建议组员直接全量覆盖自己的 APGCC 改进模型，尤其不要盲目覆盖 `models/APGCC.py` 和 `engine.py`，因为这些文件里还包含 GRL、hard-density reweight 等消融代码，不是最终推荐主方法。

## 1. 需要下载什么

从 GitHub 仓库下载：

```text
apgcc_improvements/directions/e_stain_domain_calibration/
```

如果只想快速下载，也可以下载：

```text
apgcc_improvements/directions/e_stain_domain_calibration/APGCC_E_code_bundle.tar.gz
```

解压后重点看：

```text
apgcc_improvements/directions/e_stain_domain_calibration/apgcc_patch/
```

## 2. 最有用的部分一：stain 训练预处理

### 需要参考/复制的文件

```text
apgcc_improvements/directions/e_stain_domain_calibration/apgcc_patch/datasets/dataset.py
apgcc_improvements/directions/e_stain_domain_calibration/apgcc_patch/config.py
apgcc_improvements/directions/e_stain_domain_calibration/apgcc_patch/configs/CoNIC_finetune_stain.yml
```

### 对应到自己 APGCC 的位置

```text
apgcc/datasets/dataset.py
apgcc/config.py
apgcc/configs/CoNIC_finetune_stain.yml
```

### 如果自己的 APGCC 没有改过这些文件

可以直接复制覆盖：

```bash
cp apgcc_improvements/directions/e_stain_domain_calibration/apgcc_patch/datasets/dataset.py  /path/to/apgcc/datasets/dataset.py
cp apgcc_improvements/directions/e_stain_domain_calibration/apgcc_patch/config.py            /path/to/apgcc/config.py
cp apgcc_improvements/directions/e_stain_domain_calibration/apgcc_patch/configs/CoNIC_finetune_stain.yml \
   /path/to/apgcc/configs/CoNIC_finetune_stain.yml
```

### 如果自己的 APGCC 已经改过这些文件

不要直接覆盖，手动合并下面几块：

#### 2.1 合并 `config.py` 中的配置项

把这些配置项加到自己的 `apgcc/config.py`：

```python
_C.DATALOADER.PATHOLOGY_AUG = False
_C.DATALOADER.STAIN_JITTER = False
_C.DATALOADER.BLUR_NOISE = False
_C.DATALOADER.DOMAIN_NAMES = ['crag', 'dpath', 'glas', 'pannuke', 'consep']
```

#### 2.2 合并 `dataset.py` 中的 pathology augmentation 函数

从 `apgcc_improvements/directions/e_stain_domain_calibration/apgcc_patch/datasets/dataset.py` 复制：

```python
def pathology_augment(img, stain_jitter=False, blur_noise=False):
    ...
```

这个函数包含：

- brightness jitter；
- contrast jitter；
- color jitter；
- RGB channel scale/bias；
- 可选轻量 blur/noise。

#### 2.3 在 `ImageDataset.__init__` 中读取配置

把下面逻辑合进自己的 dataset 初始化里：

```python
self.pathology_aug = bool(getattr(aug_dict, 'PATHOLOGY_AUG', False)) if aug_dict is not None else False
self.stain_jitter = bool(getattr(aug_dict, 'STAIN_JITTER', False)) if aug_dict is not None else False
self.blur_noise = bool(getattr(aug_dict, 'BLUR_NOISE', False)) if aug_dict is not None else False
self.domain_names = list(getattr(aug_dict, 'DOMAIN_NAMES', ['crag', 'dpath', 'glas', 'pannuke', 'consep'])) if aug_dict is not None else ['crag', 'dpath', 'glas', 'pannuke', 'consep']
```

#### 2.4 在 `__getitem__` 中调用 stain augmentation

在读取图片和点标注之后，训练增强之前，加：

```python
domain_label = get_domain_label(img_path, self.domain_names)
if self.train and self.pathology_aug:
    img = pathology_augment(
        img,
        stain_jitter=self.stain_jitter,
        blur_noise=self.blur_noise,
    )
```

#### 2.5 保留 domain label

给每个 target 加：

```python
target[i]['domain'] = torch.Tensor([domain_label]).long()
```

并复制这个辅助函数：

```python
def get_domain_label(img_path, domain_names):
    stem = os.path.basename(img_path).split('.')[0].lower()
    for idx, name in enumerate(domain_names):
        if stem.startswith(name.lower() + '_') or stem.startswith(name.lower() + '-') or stem == name.lower():
            return idx
    return 0
```

`domain` 字段主要给 GRL 或后续 domain 分析用。只用 stain augmentation 时，它不会影响原 APGCC loss。

### 训练配置

自己的配置文件里启用：

```yaml
DATALOADER:
  PATHOLOGY_AUG: True
  STAIN_JITTER: True
  BLUR_NOISE: True
  DOMAIN_NAMES: ['crag', 'dpath', 'glas', 'pannuke', 'consep']
```

训练命令示例：

```bash
cd /path/to/apgcc
python main.py -c ./configs/CoNIC_finetune_stain.yml
```

## 3. 最有用的部分二：DomainTh-MAE 后处理

### 需要复制的文件

```text
apgcc_improvements/directions/e_stain_domain_calibration/apgcc_patch/eval_domain_threshold.py
```

复制到自己的 APGCC：

```text
apgcc/eval_domain_threshold.py
```

命令：

```bash
cp apgcc_improvements/directions/e_stain_domain_calibration/apgcc_patch/eval_domain_threshold.py \
   /path/to/apgcc/eval_domain_threshold.py
```

### 作用

这个脚本不会改模型结构，也不会重新训练。

它做的是：

```text
1. 在 val.list 上，对每个来源域分别扫描 score threshold
2. 每个来源域选择验证集 MAE 最低的 threshold
3. 固定这些 threshold
4. 在 test.list 上评估一次
```

也就是说：

```text
val 用来选阈值
test 只用来最终报告
```

不要根据 test 结果反过来调阈值。

### 使用命令

训练完自己的模型后运行：

```bash
cd /path/to/apgcc

python eval_domain_threshold.py \
  --config ./configs/你的配置.yml \
  --weight ./output/你的模型/best.pth \
  --data-root /path/to/CoNIC_APGCC_seed19 \
  --val-list val.list \
  --test-list test.list \
  --out-dir ./output/你的模型/domain_threshold_eval_mae \
  --select-metric mae \
  --score-candidates 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 \
  --thresholds 6 12 24 \
  --gpu 0
```

输出文件：

```text
domain_threshold_eval_mae/
  thresholds.json
  summary.json
  gt.json
  pred.json
```

其中：

- `thresholds.json`：每个来源域选出的 threshold；
- `summary.json`：test.list 上的 MAE/MSE/RMSE/P/R/F1；
- `gt.json` / `pred.json`：统一点格式结果，可用于后续可视化。

## 4. 推荐最小集成流程

组员如果已有自己的 APGCC 改进模型，建议这样集成：

```text
1. 保留自己的模型结构和训练代码
2. 手动合并 stain augmentation 到 dataset.py/config.py
3. 用 stain augmentation 重新训练自己的模型
4. 复制 eval_domain_threshold.py
5. 训练结束后运行 DomainTh-MAE 后处理
6. 报告固定 0.5 阈值结果和 DomainTh-MAE 结果
```

## 5. 不建议最小集成时覆盖的文件

如果只想使用最佳方法，不建议直接覆盖：

```text
apgcc/models/APGCC.py
apgcc/models/__init__.py
apgcc/engine.py
```

原因：

```text
这些文件包含 GRL 域分类器、Weak Warmup GRL、hard-density reweight 等消融代码。
这些消融实验有分析价值，但不是最终推荐主方法。
```

最终推荐主方法是：

```text
Stain Augmentation + DomainTh-MAE
```

不是 GRL，也不是 hard-density reweight。

## 6. 我们当前实验中最有用的结果

CoNIC overlap test.list：

| 方法 | MAE | MSE | RMSE | P@12 | R@12 | F1@12 |
|---|---:|---:|---:|---:|---:|---:|
| APGCC baseline | 12.9273 | 468.8628 | 21.6532 | 0.8161 | 0.7709 | 0.7929 |
| Stain Aug | 12.4581 | 449.6448 | 21.2048 | 0.8376 | 0.7718 | 0.8034 |
| Stain Aug + DomainTh-MAE | **11.4844** | **409.1877** | **20.2284** | 0.8263 | **0.7783** | 0.8016 |

结论：

```text
stain augmentation 对训练有稳定收益；
DomainTh-MAE 对计数误差改善最大；
这两部分最值得被其他 APGCC 改进模型复用。
```
