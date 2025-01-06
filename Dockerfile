# 使用PyTorch官方CUDA镜像作为基础镜像
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 设置环境变量避免交互式配置
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglu1-mesa \
    libgl1 \
    libglvnd0 \
    libgl1-mesa-dri \
    libglx0 \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# 分批安装Python依赖以便于调试
# 首先安装基础依赖
RUN pip install --no-cache-dir \
    "transformers>=4.35.0" \
    "accelerate>=0.24.0" \
    "huggingface_hub>=0.19.0"

# 安装diffusers和controlnet相关依赖
RUN pip install --no-cache-dir \
    "diffusers>=0.21.0" \
    "controlnet_aux>=0.0.7"

# 安装runpod和其他工具
RUN pip install --no-cache-dir \
    "runpod>=1.7.0" \
    "omegaconf>=2.3.0" \
    "pytorch_lightning>=2.0.0"

# 最后安装可能有更多依赖的包
RUN pip install --no-cache-dir \
    "gradio>=4.7.0" \
    "mediapipe>=0.10.0"

# 创建模型目录
RUN mkdir -p /app/models

# 预下载模型
RUN python -c 'import torch; \
    from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter; \
    from controlnet_aux.midas import MidasDetector; \
    print("Downloading T2I Adapter..."); \
    adapter = T2IAdapter.from_pretrained("TencentARC/t2i-adapter-depth-midas-sdxl-1.0", \
        torch_dtype=torch.float16, \
        cache_dir="/app/models"); \
    print("Downloading SDXL base model..."); \
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", \
        adapter=adapter, \
        torch_dtype=torch.float16, \
        cache_dir="/app/models"); \
    print("Downloading Midas model..."); \
    midas = MidasDetector.from_pretrained("valhalla/t2iadapter-aux-models", \
        filename="dpt_large_384.pt", \
        cache_dir="/app/models"); \
    print("All models downloaded successfully!")'

# 复制应用代码
COPY handler.py /app/handler.py

# 设置Python路径
ENV PYTHONPATH=/app

# 设置权限
RUN chmod +x /app/handler.py

# 运行应用
CMD [ "python", "-u", "handler.py" ]