# 植物抠图(plant-matting)环境配置 —— 会话交接备忘

> 本文件是临时交接备忘(放在 APGCC 目录下,按你的要求)。
> **项目真正的进度台账在 `/home/lixinli/plant-matting/plan.md`,环境/权重记录在 `/home/lixinli/plant-matting/MODELS.md`**,下次以那两个为准,本文件只是断点续传用。
> 当前正在做 **Phase 0.3:各模型 conda 环境跑通 demo**。已完成 ViTMatte + Matte-Anything 环境;**正卡在 ZIM**。

---

## 0. 全局事实(下次直接用,别重新探查)

- 项目目录:`/home/lixinli/plant-matting/`(git 仓库,main 分支)
- 数据集:`/data1/llx/PM/`(150 对,用 `image_resize/`+`alpha_resize/` 1024×1024 版)
- **权重统一放**:`/data1/llx/plant-matting-ckpts/`(home 盘只剩 ~300G 且 91% 满,大文件一律放 /data1)
- 系统 CUDA:`/usr/local/cuda-11.6`(nvcc 11.6)
- 系统 gcc 4.8.5 太旧 → **编译 CUDA 扩展必须** `scl enable devtoolset-7`(gcc 7.3.1)
- pip 镜像:清华 `https://pypi.tuna.tsinghua.edu.cn/simple`(已写进各 env 的 pip.conf)
- GPU:10×RTX 3090(sm_86)→ 编译时 `TORCH_CUDA_ARCH_LIST="8.6"`;跑推理用空闲卡 `CUDA_VISIBLE_DEVICES=1`(或 nvidia-smi 看哪张空)
- **pytorch.org / Google Drive / HuggingFace 这些源很慢或不稳**;torch 用清华 PyPI 的默认 wheel(`torch==2.0.0` 即 cu117)最快。
- ⚠️ 注意:机器上另有同名大写 `ViTMatte` conda env 是**空的/坏的**,别用;真正能用的是小写 `vitmatte`。

---

## 1. ✅ ViTMatte 环境(已完成,demo 跑通)

- conda env:`vitmatte`(Python 3.8),torch 2.0.0+cu117 / torchvision 0.15.1
- **detectron2 0.6 本地可编辑安装**:源码在 `/home/lixinli/plant-matting/third_party/detectron2`,用 devtoolset-7 + CUDA11.6 编译,`_C` CUDA 扩展已成功。
- 权重:`/data1/llx/plant-matting-ckpts/vitmatte/ViTMatte_S_Com.pth`(99M,已下)
- 验证命令(可重跑):
  ```bash
  cd /home/lixinli/plant-matting/ViTMatte
  CUDA_VISIBLE_DEVICES=1 conda run -n vitmatte python run_one_image.py \
    --model vitmatte-s \
    --checkpoint-dir /data1/llx/plant-matting-ckpts/vitmatte/ViTMatte_S_Com.pth \
    --output-dir demo/result_test.png
  ```

## 2. ✅ Matte-Anything 环境(已完成,复用 vitmatte env;尚未跑 headless demo)

- 复用 `vitmatte` env;已额外装:`segment-anything`(pip)、**GroundingDINO 本地可编辑安装**
  (源码 `/home/lixinli/plant-matting/Matte-Anything/GroundingDINO`,devtoolset-7 编译,`_C` 已成功)。
- 已下权重(都在 /data1):
  - SAM vit_h:`/data1/llx/plant-matting-ckpts/sam/sam_vit_h_4b8939.pth`(2.4G)
  - GroundingDINO-T:`/data1/llx/plant-matting-ckpts/groundingdino/groundingdino_swint_ogc.pth`(662M)
  - ViTMatte-B(Distinctions):`/data1/llx/plant-matting-ckpts/vitmatte/ViTMatte_B_DIS.pth`(369M)
- 注意:装 GroundingDINO 时 opencv 被升到 4.13(原 ViTMatte 锁 4.5.3.56)。ViTMatte demo 在升级前已跑通;下次最好**复跑一次 ViTMatte demo 确认 opencv4.13 不影响**。
- 待办:Matte-Anything 只自带 gradio web-ui(`matte_anything.py`),Phase 2 要写 headless 的 `infer_matte_anything.py`;Phase 0 阶段可先跑一次 import + 加载模型确认环境 OK。
- ⚠️ matte_anything.py 里权重路径写死成 `./pretrained/...`,要把 /data1 的权重**软链接**进 `Matte-Anything/pretrained/`,或在 infer 脚本里改成 /data1 绝对路径。

---

## 3. 🔧 ZIM 环境(进行中 —— 下次从这里继续!)

ZIM 是 **onnx 推理**(`build_model.py` 读 `checkpoint/encoder.onnx` + `decoder.onnx`,引擎是 onnxruntime-gpu)。

### 已完成
- conda env:`zim`(Python 3.10)已建,pip 镜像已设
- 已装:torch 2.0.0+cu117 / torchvision 0.15.1、`zim_anything`(可编辑安装 `cd ZIM; pip install -e . --no-deps`)

### ❗ 还没做(下次的具体步骤)

1. **装 onnxruntime-gpu 1.17.0**(清华镜像最高只有 1.16.3,需官方 PyPI):
   ```bash
   conda run -n zim pip install onnxruntime-gpu==1.17.0 --index-url https://pypi.org/simple
   conda run -n zim pip install onnx opencv-python matplotlib pycocotools easydict   # 这些走清华
   ```
   - 备选:若官方源太慢,可退而用清华的 `onnxruntime-gpu==1.16.3`(只差小版本,CUDA11.x 兼容性基本一致)。
2. **GPU/cuDNN 适配(关键坑)**:onnxruntime-gpu 1.17 官方要 CUDA 11.8 + cuDNN 8.9.2;当前 torch cu117 带的是 cuDNN 8.5,**onnx 的 CUDAExecutionProvider 可能加载失败**。
   - 解决:`conda run -n zim pip install nvidia-cudnn-cu11==8.9.2.26`,跑推理时把它和 torch 的 cuda 库加入 `LD_LIBRARY_PATH`。
   - 先测 provider 是否可用:
     ```bash
     conda run -n zim python -c "import onnxruntime as ort; print(ort.get_available_providers())"
     ```
     要能看到 `CUDAExecutionProvider`,且实际 InferenceSession 不报 cudnn 错。
3. **权重(需用户自己下)**:先用 vit_b。
   - 下载:HuggingFace `https://huggingface.co/naver-iv/zim-anything-vitb/tree/main/zim_vit_b_2043`
   - 需要文件:该文件夹里的 `encoder.onnx` 和 `decoder.onnx`
   - **放到**:`/data1/llx/plant-matting-ckpts/zim/zim_vit_b_2043/`
     最终:
     ```
     /data1/llx/plant-matting-ckpts/zim/zim_vit_b_2043/encoder.onnx
     /data1/llx/plant-matting-ckpts/zim/zim_vit_b_2043/decoder.onnx
     ```
4. **跑 demo 验证**:用法见 ZIM/README(`zim_model_registry[backbone](checkpoint=<那个文件夹路径>)` → `ZimPredictor` → `predict(点/框 prompt)`)。
   demo 脚本:`ZIM/demo/gradio_demo.py`;批量参考 `ZIM/script/amg.py`。Phase 0 只需对 1 张图给个 prompt 出 alpha 验证环境。

---

## 4. ⬜ Matting-Anything 环境(还没开始)

- 计划 env:`matanything`(Python 3.10)**已建好、已设镜像,但还没装任何包**。
- 仓库 `/home/lixinli/plant-matting/Matting-Anything`,自带 bundled `segment-anything/` + `GroundingDINO/`(需 devtoolset-7 编译)+ M2M head。
- 看 `Matting-Anything/INSTALL.md` 和 `GETTING_STARTED.md` 定步骤。
- 权重(下次再列):SAM ckpt(可复用 /data1 的 sam_vit_h)、GroundingDINO ckpt(可复用)、M2M / MAM 自己的权重(需查 repo README 链接,让用户下到 `/data1/llx/plant-matting-ckpts/matanything/`)。

---

## 5. 工作方式约定(重要)

- **一步一步来,别一次性跑完全程**;每个模型单独推进、逐步确认。
- **大权重由用户自己下载**:我只负责告诉(1)下哪个/链接,(2)放到 `/data1/llx/plant-matting-ckpts/<model>/` 哪里。pip 包我自己装。
- 每完成一步,更新 `plan.md` 的 checkbox + 进度日志,以及 `MODELS.md`。

## 6. 下次开场建议

> "继续配置 plant-matting 的环境,接着 ZIM 来。先读 /home/lixinli/plant-matting/plan.md 和 MODELS.md,以及本交接文件。ZIM 的 env `zim` 已装好 torch+zim_anything,差 onnxruntime-gpu 和 onnx 权重。"
