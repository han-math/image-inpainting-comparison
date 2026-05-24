# 图像填充方法对比：regionfill / Exemplar / Pyramid / SD Inpainting

本仓库为中山大学 2025–2026 学年某课程第五次小作业的代码与实验结果。在四张测试图像（`bricks`、`crayon_paint`、`fingerprint256`、`white_house`）上比较了四种图像填充方法：

1. **MATLAB `regionfill`** —— 基于 Laplace 方程的边界插值；
2. **MATLAB `inpaintExemplar`** —— Criminisi 等人提出的范例块修复（Region Filling by Exemplar-Based Image Inpainting）；
3. **多尺度图像金字塔修复** —— 粗尺度用 `regionfill` 估计结构、细尺度用 `inpaintExemplar` 细化，层数 $L=2,3,4$；
4. **Stable Diffusion Inpainting** —— 调用预训练的 `runwayml/stable-diffusion-inpainting`（Latent Diffusion Model）做生成式修复，含黑色残留二次修复与白房子图像的结构先验后处理。

完整的方法说明、数学推导和实验分析见课程作业报告（PDF 因体积较大未随仓库发布）。本仓库只包含代码、输入数据与各方法的最终结果。

## 目录结构

```
.
├── input/                       # 四张原图与对应受损图
├── results_regionfill/          # 方法 1 输出
├── results_exemplar/            # 方法 2 输出
├── results_pyramid/             # 方法 3 输出（含 L=2,3,4）
├── results_sd_inpaint_hook/     # 方法 4 最终输出（GPU + 后处理 hook）
├── run_regionfill.m             # 方法 1 实现
├── run_exemplar.m               # 方法 2 实现
├── run_pyramid_inpaint.m        # 方法 3 实现
└── run_sd_inpaint_gpu_hook.py   # 方法 4 最终实现（GPU + 后处理）
```

## 运行环境

| 方法 | 工具 | 测试版本 | 备注 |
|---|---|---|---|
| 1, 2, 3 | MATLAB | R2025b | 需 Image Processing Toolbox（提供 `regionfill`, `inpaintExemplar`） |
| 4 | Python 3.10+ | — | 依赖 `torch`, `diffusers`, `transformers`, `accelerate`, `safetensors`, `Pillow`, `numpy`；推荐在带 CUDA 的 GPU 上运行（仓库代码即在 NVIDIA GPU 上调通） |

Python 依赖一键安装：

```bash
pip install torch torchvision diffusers transformers accelerate safetensors pillow numpy
```

## 复现步骤

### 重要：先改硬编码路径

为方便交作业评审，三份 MATLAB 脚本里写的是作者本机路径（`/Users/xiaohan/Downloads/第五次小作业/...`）。复现前请把以下文件**顶部的 `input_dir` / `output_dir` 等绝对路径**改为你本机仓库的实际路径：

- `run_regionfill.m`
- `run_exemplar.m`
- `run_pyramid_inpaint.m`

GPU 版本 `run_sd_inpaint_gpu_hook.py` 不含硬编码路径，通过环境变量配置（见下文）。

### 方法 1：`regionfill`

```matlab
>> run('run_regionfill.m')
```

### 方法 2：基于范例的修复

```matlab
>> run('run_exemplar.m')
```

### 方法 3：多尺度金字塔修复

```matlab
>> run('run_pyramid_inpaint.m')
```

### 方法 4：Stable Diffusion Inpainting（GPU）

`run_sd_inpaint_gpu_hook.py` 通过环境变量配置输入输出路径，默认指向 `/root/inpaint/`（仓库作者云 GPU 上的目录）。本地复现时设置：

```bash
export INPAINT_INPUT_DIR=/path/to/this/repo/input
export INPAINT_OUTPUT_DIR=/path/to/this/repo/results_sd_inpaint_hook
export INPAINT_DEVICE=cuda            # 或 cpu / mps（不推荐，速度极慢）
python run_sd_inpaint_gpu_hook.py
```

首次运行会自动从 Hugging Face 下载预训练权重 `runwayml/stable-diffusion-inpainting`（约 5 GB）。

## 主要结果

各方法在四张测试图像上的 PSNR / SSIM 见课程作业报告。SD Inpainting 在 `bricks` 和 `white_house` 上取得最优；`regionfill` 在 `crayon_paint` 和 `fingerprint256` 上仍占优，对应原因（生成式模型对规则纹理的像素级一致性较难保证）已在报告中讨论。

各方法的具体输出图像见 `results_*` 目录。

## 致谢与许可

- Stable Diffusion Inpainting 模型来自 [runwayml/stable-diffusion-inpainting](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting)（Rombach et al., 2022, CVPR）。
- 本仓库代码以 MIT 许可发布；测试图像版权归原作者所有，仅用于课程作业的方法对比。
