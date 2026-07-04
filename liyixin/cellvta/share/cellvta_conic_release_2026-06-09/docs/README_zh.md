# CoNIC 统一划分说明

## 先看这里

组内统一时，直接使用已经划分好的：

- `docs/conic_split_seed19.json`

不要再各自随机划分 `train / val / test`。

## 具体怎么用

1. 解压两个压缩包：
   - `data/conic_cellvit_patient_x40_linear_withOverlap.zip`
   - `data/conic_cellvit_patient.zip`
2. 读取 `docs/conic_split_seed19.json`
3. `train` 和 `val` 按 `json` 中给出的名单，从 `conic_cellvit_patient_x40_linear_withOverlap` 里读取
4. `test` 按 `json` 中给出的名单，从 `conic_cellvit_patient` 的 `fold1` 里读取

## 数据增强说明

这套组内统一数据已经完成了统一的数据增强处理。

具体来说：

- `train` 和 `val` 使用的 `conic_cellvit_patient_x40_linear_withOverlap.zip`
- 是在原始 patch 基础上做过离线增强后的版本
- 每个原始 patch 被固定生成了 `4` 个 overlap 子样本

因此：

- 现在提供给组内使用的训练数据，已经是增强后的数据
- 组内在使用这套共享数据时，不需要再额外重复做这一套离线 overlap 增强

这里的统一含义是：

- 大家都使用同一份已经增强好的训练数据
- `train / val` 的划分也是在这份增强后的数据上固定完成的

需要注意：

- 这里说“不用再额外数据增强”，指的是不用再重复做这一步离线数据生成
- 如果某个模型训练时还需要自己的在线随机增强（模型特有），例如翻转、颜色扰动等，那属于训练阶段策略，和这里这份统一离线数据不冲突

## 样本数量

| 集合 | 样本数 |
| --- | --- | ---: |
| train | `12768` |
| val | `3192` |
| test | `991` |

## `conic_split_seed19.json` 里有什么

这份 `json` 只保留直接使用需要的信息：

- `seed`
- `val_ratio`
- `counts`
- `splits.train`
- `splits.val`
- `splits.test`

其中每个样本只包含：

- `stem`
- `image_relpath`
- `label_relpath`

实际使用时，直接按 `splits.train / splits.val / splits.test` 中的路径读取数据即可。

## 当前文件说明

### `data/`

- `conic_cellvit_patient_x40_linear_withOverlap.zip`
  - 训练和验证使用的数据包
  - `train` 和 `val` 都从这个压缩包中读取
- `conic_cellvit_patient.zip`
  - 测试使用的数据包
  - `test` 从这个压缩包的 `fold1` 读取

### `docs/`

- `README_zh.md`
  - 当前这份说明文件
  - 用来说明数据怎么使用
- `conic_split_seed19.json`
  - 已经固定好的 `train / val / test` 划分文件
  - 组内直接按它读取数据，不再自己重新划分
- `conic_fold_summary.json`
  - 原始 `fold0 / fold1` 的统计信息
  - 用于辅助核对划分规模和来源分布，不是训练和测试时必须读取的文件

### `scripts/`

- `generate_conic_split_json.py`
  - 用于重新生成 `conic_split_seed19.json`
  - 如果以后需要重新导出同一套固定划分，可以运行这个脚本
