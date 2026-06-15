# 三数据集统一数据增强协议（并集版）

并集定义：任一对比方法使用过，且其他模型/标注格式可以同步实现的增强，都纳入统一协议；只有明确不适用才排除。统一协议只对训练集做随机增强，验证集和测试集不做随机增强。

## 1. 各方法原始增强总表

| 方法 | 原始使用的增强 | 未使用 / 限制 | 备注 |
|---|---|---|---|
| STEERER | 随机缩放：约 0.5-2.0；随机裁剪后缩放到训练尺寸；水平翻转 p=0.5。 | 无垂直翻转；无 affine；无 blur/noise；无颜色扰动。 | 当前三个数据集配置基本一致，常用训练输入为 640x640。 |
| PET | 随机缩放 0.8-1.2；随机裁剪 / resize 到 256x256；水平翻转 p=0.5。 | 无垂直翻转；无 affine；无 blur/noise；无颜色扰动。 | 点坐标内部多用 [y, x]，但增强语义与 [x, y] 一致。 |
| APGCC | 随机缩放；多个随机 crop；水平翻转 p=0.5。BCData 可到 0.7-1.3，CoNIC/MoNuSeg 常见 0.7-1.0。 | 无垂直翻转；无 affine；无 blur/noise；无颜色扰动。 | CROP_NUMBER=4 属于多 patch 采样；BCData/MoNuSeg 常用 256，CoNIC 常用 128。 |
| HoVer-Net MoNuSeg | CenterCrop；水平翻转 p=0.5；垂直翻转 p=0.5；affine scale 0.8-1.2；translation +-1%；shear +-5 deg；rotation +-179 deg；blur/noise；颜色扰动。 | 无明显排除项。 | master 分支策略，是并集中 affine 和像素增强的主要来源。 |
| HoVer-Net CoNIC | CenterCrop；水平翻转 p=0.5；垂直翻转 p=0.5；blur/noise；颜色扰动。 | CoNIC 分支去掉 affine。 | 如果小组认为这是 CoNIC 不适用 affine 的证据，则 CoNIC 统一协议需关闭 affine。 |
| 当前并集协议 | 随机缩放；随机裁剪；水平/垂直翻转；随机 affine；blur/noise；颜色扰动。 | 只排除明确不适用项。 | 默认参数见下一页。 |

## 2. 全局统一参数

| 类别 | 当前默认设置 | 作用对象 | 可关闭参数 |
|---|---|---|---|
| 随机缩放 | p=1.0；scale ~ U(0.8, 1.2)。native 的 0.5-2.0 或 0.7-1.3 作为消融，不作为统一默认。 | 图像、点坐标、polygon 顶点。 | --scale-min；--scale-max |
| 随机 affine | BCData/MoNuSeg：p=1.0，affine scale U(0.8,1.2)；translation U(-0.01,0.01)*image_size；shear U(-5 deg,5 deg)；rotation U(-179 deg,179 deg)。CoNIC 默认 p=0。 | 图像、点坐标、polygon 顶点；mask 用 nearest。 | --no-affine；--rotate-deg 0 |
| 随机裁剪 | p=1.0；裁剪到目标 patch size。patch size 按模型输入适配，必须记录。 | 图像和对应标注。 | --patch-size |
| 水平翻转 | p=0.5。 | 图像、点坐标、polygon 顶点、mask。 | --hflip-prob 0 |
| 垂直翻转 | p=0.5。 | 图像、点坐标、polygon 顶点、mask。 | --vflip-prob 0 |
| Blur / Noise | p=1.0；Gaussian blur / median blur / Gaussian noise 三选一，各 1/3。blur kernel 从 {1,3,5} 采样；noise sigma ~ U(0,12.75)。 | 只作用图像。 | --blur-noise-prob 0；--no-pixel-aug |
| 颜色扰动 | p=1.0；hue U(-8,8)；saturation U(-0.2,0.2)；brightness U(-26,26)；contrast factor U(0.75,1.25)。 | 只作用图像。 | --color-aug-prob 0；--no-pixel-aug |
| 归一化 | 不在离线增强脚本里做。ImageNet normalize 或模型自己的 normalize 按模型实现保留。 | 模型预处理。 | 按模型配置 |

增强顺序固定为：随机缩放 -> 随机 affine -> 随机裁剪 / patch -> 水平/垂直翻转 -> blur/noise -> 颜色扰动。除非数据集小节写明“按模型适配”，否则参数固定使用。

## 3. BCData：统一规定

BCData 的原始标注是 positive/negative 两套 h5 点坐标。增强时同一张图的两类点必须使用同一次几何变换，增强后仍保持原始目录结构。

| 类别 | 当前做法 |
|---|---|
| 输入输出 | 输入为原始 BCData：images/{train,validation,test} 和 annotations/{split}/{positive,negative}/*.h5。输出保持同样结构；每个 h5 读写 coordinates，坐标为 [x, y]。 |
| 生效 split | 只增强 train，p=1.0；validation/test 不做随机增强，原样复制。 |
| 每图增强次数 | 按模型采样策略适配，必须记录。脚本默认 --num-augments 1；APGCC 这类多 crop 采样器可按 CROP_NUMBER 处理。 |
| Patch 尺寸 | 按模型输入适配，必须记录。脚本默认 256x256；当前 STEERER 若不改 config，可设 --patch-size 640。 |
| 随机缩放 | p=1.0；scale ~ U(0.8,1.2)。 |
| 随机 affine | p=1.0；affine scale U(0.8,1.2)；translation +-1%；shear +-5 deg；rotation +-179 deg。 |
| 随机裁剪 | p=1.0；裁到目标 patch size；图像小于 patch 时先 padding。 |
| 水平翻转 | p=0.5。 |
| 垂直翻转 | p=0.5。 |
| blur/noise | p=1.0；Gaussian blur / median blur / Gaussian noise 三选一，各 1/3。 |
| 颜色扰动 | p=1.0；hue、saturation、brightness、contrast 参数同全局统一参数。 |
| 标签同步 | positive 和 negative 分开保存，但同一张图两类坐标使用完全相同的几何变换；裁剪后落在 patch 外的点会被移除。 |
| 空 patch | 脚本默认保留空 patch：--min-points 0。若某模型必须过滤空 patch，可设 --min-points 1，但必须记录。 |

命令示例：

```bash
python tools/augmentation/augment_bcdata.py --input /path/to/BCData --output /path/to/BCData_aug --num-augments 1 --patch-size 256
```

## 4. MoNuSeg：统一规定

MoNuSeg 的原始标注是 XML polygon。增强时不先转点，而是直接变换 polygon 顶点，后续分割模型和点模型都可以再各自转换。

| 类别 | 当前做法 |
|---|---|
| 输入输出 | 输入为原始 MoNuSeg 图像和同名 XML。脚本递归搜索图像和 .xml，并根据 Training/Validation/Testing 判断 split。输出为原始风格目录。 |
| 生效 split | 只增强 train，p=1.0；val/test 不做随机增强，原样复制。 |
| 每图增强次数 | 按模型采样策略适配，必须记录。脚本默认 --num-augments 1。 |
| Patch 尺寸 | 按模型输入适配，必须记录。脚本默认 256x256；当前 STEERER 可设为 640x640。 |
| 随机缩放 | p=1.0；scale ~ U(0.8,1.2)。 |
| 随机 affine | p=1.0；affine scale U(0.8,1.2)；translation +-1%；shear +-5 deg；rotation +-179 deg。 |
| 随机裁剪 | p=1.0；裁到目标 patch size；图像小于 patch 时先 padding。 |
| 水平翻转 | p=0.5。 |
| 垂直翻转 | p=0.5。 |
| blur/noise | p=1.0；Gaussian blur / median blur / Gaussian noise 三选一，各 1/3。 |
| 颜色扰动 | p=1.0；hue、saturation、brightness、contrast 参数同全局统一参数。 |
| 标签同步 | XML 中每个 Region 的 Vertex polygon 顶点做同样几何变换。裁剪边界上的 polygon 会被裁到 patch 内；完全落在 patch 外的 Region 会被删除。 |
| 空区域 | 脚本默认允许空区域：--min-regions 0。若某模型不能处理空 XML，可设 --min-regions 1，但必须记录。 |

命令示例：

```bash
python tools/augmentation/augment_monuseg.py --input /path/to/MoNuSeg --output /path/to/MoNuSeg_aug --num-augments 1 --patch-size 256
```

## 5. CoNIC：统一规定

当前目录暂未单独写 CoNIC 原始格式增强脚本；这里先固定协议口径，后续实现时按 image + instance/type mask 同步增强。

| 类别 | 当前做法 |
|---|---|
| 固定 overlap 子样本 | 每个原始 patch 固定生成 4 个 overlap 子样本，p=1.0。这是确定性 patch extraction / tiling，本身不是随机增强。overlap 尺寸和 stride 按现有处理脚本或模型复现要求，必须记录。 |
| 生效 split | 只增强 train，p=1.0；val/test 不做随机增强。 |
| 每图增强次数 | 按模型采样策略适配，必须记录。4 overlap 是 tiling 数，不等同于随机增强次数。 |
| Patch 尺寸 | 按模型输入适配，必须记录。若 4 个 overlap 子样本已经等于模型输入尺寸，则随机裁剪退化为不裁剪或确定性裁剪。 |
| 随机缩放 | p=1.0；scale ~ U(0.8,1.2)。 |
| 随机 affine | 当前统一规定为 p=0，即 CoNIC 默认关闭 affine。理由是 HoVer-Net CoNIC 分支明确去掉 affine，可暂视为 CoNIC 数据集级不适用项。若小组最终决定严格保留并集，则改为 p=1.0，参数同 BCData/MoNuSeg。 |
| 随机裁剪 | p=1.0，但目标尺寸按模型输入适配；若输入子 patch 已经是目标尺寸，则不额外随机裁剪。 |
| 水平翻转 | p=0.5。 |
| 垂直翻转 | p=0.5。 |
| blur/noise | p=1.0；Gaussian blur / median blur / Gaussian noise 三选一，各 1/3。 |
| 颜色扰动 | p=1.0；hue、saturation、brightness、contrast 参数同全局统一参数。 |
| 标签同步 | 若使用 instance/type mask，几何增强必须同步作用到 image、instance map、type map；instance id map 应使用 nearest 插值，不能用 bilinear。若先转成点坐标，则点坐标按同样几何矩阵变换。 |
| 像素增强作用对象 | blur/noise 和颜色扰动只作用于 image，不作用于 mask 或点标签。 |

常用关闭参数：--no-affine；--no-pixel-aug；--vflip-prob 0；--rotate-deg 0；--blur-noise-prob 0；--color-aug-prob 0。
