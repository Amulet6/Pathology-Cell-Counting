# 三数据集统一数据增强协议（并集版）

这个目录提供离线数据增强脚本，增强发生在**数据集原始格式**层面，不是 STEERER 的 `jsons/*.json` 或 `*_gt_loc.txt` 层面。

- `augment_bcdata.py`：读写 BCData 原始 `images/{split}` 和 `annotations/{split}/{positive,negative}/*.h5`。
- `augment_monuseg.py`：读写 MoNuSeg 原始图像和 XML polygon 标注。
- CoNIC 暂未单独写脚本，本文件先固定统一协议口径。

增强后的数据仍然可以再分别喂给 STEERER、APGCC、PET、HoVer-Net 等方法各自的转换脚本。

## 统一原则

这里的“并集”定义为：只要任一对比方法使用了某种增强，并且该增强能同步作用到其他模型/标注格式，就纳入统一协议；只有明确不适用的增强才排除。

统一协议的增强顺序固定为：

```text
随机缩放 -> 随机 affine -> 随机裁剪 / patch -> 水平/垂直翻转 -> blur/noise -> 颜色扰动
```

统一协议只对训练集做随机增强。验证集和测试集不做随机增强，只做各模型自己的确定性预处理。

## 各方法原始增强

| 方法 | 随机缩放 | 裁剪 / patch | 水平翻转 | 垂直翻转 | affine / 旋转 / shear | blur / noise | 颜色扰动 | 备注 |
|---|---|---|---|---|---|---|---|---|
| STEERER | 有，当前代码离散采样，约覆盖 `0.5-2.0` | 随机裁剪后缩放到训练尺寸，当前常用 `640 x 640` | 有，`p=0.5` | 无 | 无 | 无 | 无 | 三个数据集当前配置基本一致 |
| PET | 有，`0.8-1.2` | 随机裁剪 / resize 到 `256 x 256` | 有，`p=0.5` | 无 | 无 | 无 | 无 | 点坐标内部多用 `[y, x]`，但增强语义一致 |
| APGCC | 有；BCData 可到 `0.7-1.3`，CoNIC/MoNuSeg 常见为 `0.7-1.0` | 多个随机 crop；BCData/MoNuSeg 常用 `256`，CoNIC 常用 `128` | 有，`p=0.5` | 无 | 无 | 无 | 无 | `CROP_NUMBER=4` 属于多 patch 采样 |
| HoVer-Net MoNuSeg | 有，affine scale `0.8-1.2` | CenterCrop | 有，`p=0.5` | 有，`p=0.5` | 有，translation `±1%`、shear `±5°`、rotation `±179°` | 有，Gaussian blur / median blur / Gaussian noise 三选一 | 有，hue / saturation / brightness / contrast | master 分支策略 |
| HoVer-Net CoNIC | 无 affine scale | CenterCrop | 有，`p=0.5` | 有，`p=0.5` | 无 | 有，Gaussian blur / median blur / Gaussian noise 三选一 | 有，hue / saturation / brightness / contrast | CoNIC 分支去掉 affine |

## 全局统一参数

除非在数据集小节里明确写为“按模型适配”，否则以下参数固定使用。

| 类别 | 统一规定 |
|---|---|
| 随机缩放 | 概率 `p=1.0`；`scale ~ U(0.8, 1.2)`。不同方法 native 的 `0.5-2.0` 或 `0.7-1.3` 作为 native/消融，不作为统一主协议默认值。 |
| 随机 affine | BCData、MoNuSeg 默认 `p=1.0`；affine scale `U(0.8, 1.2)`；translation `U(-0.01, 0.01) * image_size`；shear `U(-5°, 5°)`；rotation `U(-179°, 179°)`。 |
| 随机裁剪 | 概率 `p=1.0`；裁剪到目标 patch size。目标 patch size 按模型输入适配，必须记录。 |
| 水平翻转 | 概率 `p=0.5`。 |
| 垂直翻转 | 概率 `p=0.5`。 |
| blur/noise | 概率 `p=1.0`；三选一，Gaussian blur / median blur / Gaussian noise 各 `1/3`。blur kernel 从 `{1,3,5}` 采样；Gaussian noise 的 `sigma ~ U(0, 12.75)`。 |
| 颜色扰动 | 概率 `p=1.0`；hue `U(-8, 8)`；saturation `U(-0.2, 0.2)`；brightness 加性扰动 `U(-26, 26)`；contrast factor `U(0.75, 1.25)`。 |
| 归一化 | 不在离线增强脚本里做。ImageNet normalize 或各模型自己的 normalize 属于模型预处理，按模型实现保留。 |

## BCData 当前统一规定

| 类别 | 统一规定 |
|---|---|
| 输入输出 | 输入为原始 BCData：`images/{train,validation,test}` 和 `annotations/{split}/{positive,negative}/*.h5`。输出保持同样结构。每个 `.h5` 读写 `coordinates`，坐标为 `[x, y]`。 |
| 生效 split | 只增强 `train`，概率 `p=1.0`。`validation/test` 不做随机增强，原样复制。 |
| 每图增强次数 | 按模型采样策略适配，必须记录。脚本默认 `--num-augments 1`。APGCC 这类多 crop 采样器可按自身 `CROP_NUMBER` 处理。 |
| patch size | 按模型输入适配，必须记录。脚本默认 `256 x 256`；当前 STEERER 若不改 config，可设 `--patch-size 640`。 |
| 随机缩放 | `p=1.0`；`scale ~ U(0.8, 1.2)`。 |
| 随机 affine | `p=1.0`；affine scale `U(0.8, 1.2)`；translation `±1%`；shear `±5°`；rotation `±179°`。 |
| 随机裁剪 | `p=1.0`；裁到目标 patch size；图像小于 patch 时先 padding。 |
| 水平翻转 | `p=0.5`。 |
| 垂直翻转 | `p=0.5`。 |
| blur/noise | `p=1.0`；Gaussian blur / median blur / Gaussian noise 三选一，各 `1/3`。 |
| 颜色扰动 | `p=1.0`；hue、saturation、brightness、contrast 参数同全局统一参数。 |
| 标签同步 | `positive` 和 `negative` 分开保存，但同一张图两类坐标使用完全相同的几何变换。裁剪后落在 patch 外的点会被移除。 |
| 空 patch | 统一脚本默认保留空 patch：`--min-points 0`。如果某个模型训练策略必须过滤空 patch，可设 `--min-points 1`，但必须在实验记录中说明。 |

示例：

```bash
python tools/augmentation/augment_bcdata.py \
  --input /path/to/BCData \
  --output /path/to/BCData_aug \
  --num-augments 1 \
  --patch-size 256
```

## MoNuSeg 当前统一规定

| 类别 | 统一规定 |
|---|---|
| 输入输出 | 输入为原始 MoNuSeg 图像和同名 XML。脚本递归搜索图像和 `.xml`，并根据 `Training/Validation/Testing` 判断 split。输出为 `Training/Tissue Images`、`Training/Annotations` 等原始风格目录。 |
| 生效 split | 只增强 `train`，概率 `p=1.0`。`val/test` 不做随机增强，原样复制。 |
| 每图增强次数 | 按模型采样策略适配，必须记录。脚本默认 `--num-augments 1`。 |
| patch size | 按模型输入适配，必须记录。脚本默认 `256 x 256`；若喂给当前 STEERER，可设为 `640 x 640`。 |
| 随机缩放 | `p=1.0`；`scale ~ U(0.8, 1.2)`。 |
| 随机 affine | `p=1.0`；affine scale `U(0.8, 1.2)`；translation `±1%`；shear `±5°`；rotation `±179°`。 |
| 随机裁剪 | `p=1.0`；裁到目标 patch size；图像小于 patch 时先 padding。 |
| 水平翻转 | `p=0.5`。 |
| 垂直翻转 | `p=0.5`。 |
| blur/noise | `p=1.0`；Gaussian blur / median blur / Gaussian noise 三选一，各 `1/3`。 |
| 颜色扰动 | `p=1.0`；hue、saturation、brightness、contrast 参数同全局统一参数。 |
| 标签同步 | XML 中每个 `Region` 的 `Vertex` polygon 顶点做同样几何变换。裁剪边界上的 polygon 会被裁到 patch 内；完全落在 patch 外的 `Region` 会被删除。 |
| 空区域 | 统一脚本默认允许空区域：`--min-regions 0`。如果某个模型不能处理空 XML，可设 `--min-regions 1`，但必须在实验记录中说明。 |

示例：

```bash
python tools/augmentation/augment_monuseg.py \
  --input /path/to/MoNuSeg \
  --output /path/to/MoNuSeg_aug \
  --num-augments 1 \
  --patch-size 256
```

## CoNIC 当前统一规定

当前目录暂未单独写 CoNIC 原始格式增强脚本；这里先固定协议口径，后续实现时按 image + instance/type mask 同步增强。

| 类别 | 统一规定 |
|---|---|
| 固定 overlap 子样本 | 每个原始 patch 固定生成 4 个 overlap 子样本，概率 `p=1.0`。这一步是确定性 patch extraction / tiling，本身不是随机增强。具体 overlap 尺寸、stride 按现有 CoNIC 处理脚本或模型复现要求，必须记录。 |
| 生效 split | 只增强 `train`，概率 `p=1.0`。`val/test` 不做随机增强。 |
| 每图增强次数 | 按模型采样策略适配，必须记录。固定 4 overlap 是 tiling 数，不等同于 `--num-augments`。 |
| patch size | 按模型输入适配，必须记录。若 4 个 overlap 子样本已经等于模型输入尺寸，则随机裁剪退化为不裁剪或确定性裁剪。 |
| 随机缩放 | `p=1.0`；`scale ~ U(0.8, 1.2)`。 |
| 随机 affine | 当前统一规定为 `p=0`，即 CoNIC 默认关闭 affine。理由是 HoVer-Net CoNIC 分支明确去掉 affine，可暂视为 CoNIC 数据集级不适用项。若小组最终决定严格保留并集，则改为与 BCData/MoNuSeg 相同参数：`p=1.0`、scale `0.8-1.2`、translation `±1%`、shear `±5°`、rotation `±179°`。 |
| 随机裁剪 | `p=1.0`，但目标尺寸按模型输入适配；如果输入子 patch 已经是目标尺寸，则不额外随机裁剪。 |
| 水平翻转 | `p=0.5`。 |
| 垂直翻转 | `p=0.5`。 |
| blur/noise | `p=1.0`；Gaussian blur / median blur / Gaussian noise 三选一，各 `1/3`。 |
| 颜色扰动 | `p=1.0`；hue、saturation、brightness、contrast 参数同全局统一参数。 |
| 标签同步 | 若使用 instance/type mask，几何增强必须同步作用到 image、instance map、type map；instance id map 使用 nearest 插值，不能用 bilinear。若先转成点坐标，则点坐标按同样几何矩阵变换。 |
| 像素增强作用对象 | blur/noise 和颜色扰动只作用于 image，不作用于 mask 或点标签。 |

## 常用关闭参数

如果后续讨论认为某个增强不适用，可以通过参数关闭：

```bash
--no-affine
--no-pixel-aug
--vflip-prob 0
--rotate-deg 0
--blur-noise-prob 0
--color-aug-prob 0
```

如果采用离线增强后的数据训练，各模型内部 dataloader 里同类随机增强要么关掉，要么明确记录为“离线增强 + 模型 native 增强叠加”，避免同一个实验协议被实际增强两次。
