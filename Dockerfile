# 使用支持CUDA的PyTorch基础镜像
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 安装git和基础工具
RUN apt-get update && apt-get install -y git

# 克隆仓库
RUN git clone https://github.com/kestraltorah/T2I-Adapter.git .

# 安装依赖
RUN pip install -r requirements.txt && \
    pip install runpod \
    diffusers>=0.21.4 \
    transformers \
    accelerate \
    safetensors \
    controlnet_aux==0.0.7 \
    xformers

# 创建handler.py
RUN echo 'import runpod
import torch
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from controlnet_aux import LineartDetector

def init_model():
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-lineart-sdxl-1.0",
        torch_dtype=torch.float16,
        varient="fp16"
    ).to("cuda")
    
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    euler_a = EulerAncestralDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        model_id,
        vae=vae,
        adapter=adapter,
        scheduler=euler_a,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to("cuda")
    
    pipe.enable_xformers_memory_efficient_attention()
    return pipe, LineartDetector.from_pretrained("lllyasviel/Annotators").to("cuda")

# 初始化模型
pipe, line_detector = init_model()

def handler(event):
    try:
        # 获取输入
        prompt = event["input"].get("prompt", "")
        image = event["input"].get("image")
        
        if not prompt or not image:
            return {"error": "Missing prompt or image"}
            
        # 处理图像
        image = line_detector(image, detect_resolution=384, image_resolution=1024)
        
        # 生成图像
        output = pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=30,
            adapter_conditioning_scale=0.8,
            guidance_scale=7.5
        )
        
        # 返回结果
        return {"generated_image": output.images[0]}
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})' > handler.py

# 设置环境变量
ENV PYTHONPATH=/app

# 启动命令
CMD [ "python", "-u", "handler.py" ]
