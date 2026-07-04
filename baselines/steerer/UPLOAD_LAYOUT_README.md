# 按 STEERER 项目结构上传说明（不覆盖现有 converter）

这个包是“解压到现有 STEERER 项目根目录”的版本，并且不会覆盖已有的：

```text
tools/convert_bcdata_to_steerer.py
tools/convert_conic_to_steerer.py
tools/convert_monuseg_to_steerer.py
```

本包把新增脚本都放在：

```text
tools/augmentation/
```

解压后新增/补充：

```text
STEERER/
  requirements_aug.txt
  UPLOAD_LAYOUT_README.md

  tools/
    augmentation/
      augment_bcdata.py
      augment_monuseg.py
      original_augment_common.py
      convert_bcdata_unified_aug_to_steerer.py
      label_to_centroids_unified.py
      run_bcdata_unified_aug.sh
      README_unified_aug.md
      data_augmentation_protocol_with_rationale.pdf

  outputs/
    README_BCData_unified_aug.md
```

## 安装依赖

在 `STEERER/` 根目录运行：

```bash
pip install -r requirements_aug.txt
```

## 运行 BCData 统一增强

在 `STEERER/` 根目录运行：

```bash
bash tools/augmentation/run_bcdata_unified_aug.sh /path/to/original/BCData
```

默认输出：

```text
outputs/BCData_unified_aug/
  raw/       # 增强后的 BCData 原始格式
  steerer/   # 转换后的 STEERER processed 格式
```

如果要适配当前 STEERER 的 `640 x 640` 输入：

```bash
bash tools/augmentation/run_bcdata_unified_aug.sh /path/to/original/BCData outputs/BCData_unified_aug_640 640 1
```

训练时数据路径指向：

```text
outputs/BCData_unified_aug/steerer
```

或：

```text
outputs/BCData_unified_aug_640/steerer
```

## 文件命名说明

为了避免覆盖现有脚本：

- 原来的 `tools/convert_bcdata_to_steerer.py` 不动。
- 新增转换脚本叫 `tools/augmentation/convert_bcdata_unified_aug_to_steerer.py`。
- 你提供的点标注脚本叫 `tools/augmentation/label_to_centroids_unified.py`。

BCData 原 STEERER 增强结果可以沿用；这个包只额外生成统一增强版 BCData。
