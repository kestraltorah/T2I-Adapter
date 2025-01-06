import runpod
import torch
from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter
from controlnet_aux.midas import MidasDetector
import base64
from io import BytesIO
from PIL import Image

def init_models():
    # 初始化模型
    adapter = T2IAdapter.from_pretrained(
        "TencentARC/t2i-adapter-depth-midas-sdxl-1.0",
        torch_dtype=torch.float16
    ).to("cuda")
    
    pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        adapter=adapter,
        torch_dtype=torch.float16
    ).to("cuda")
    
    midas = MidasDetector.from_pretrained(
        "valhalla/t2iadapter-aux-models",
        filename="dpt_large_384.pt"
    ).to("cuda")
    
    return pipe, midas

# 全局变量
PIPE = None
MIDAS = None

def handler(event):
    global PIPE, MIDAS
    
    # 首次运行时初始化模型
    if PIPE is None or MIDAS is None:
        PIPE, MIDAS = init_models()
    
    try:
        # 获取输入
        job_input = event.get("input", {})
        
        # 验证输入
        if not job_input:
            return {"error": "No input provided"}
        
        # 获取参数
        image_b64 = job_input.get("image")
        prompt = job_input.get("prompt", "")
        
        if not image_b64 or not prompt:
            return {"error": "Missing required input: image or prompt"}
        
        # 解码图像
        image = Image.open(BytesIO(base64.b64decode(image_b64)))
        
        # 生成深度图
        depth_image = MIDAS(image)
        
        # 生成图像
        output = PIPE(
            prompt=prompt,
            image=depth_image,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
        # 编码输出图像
        buffered = BytesIO()
        output.save(buffered, format="PNG")
        output_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {"output": output_b64}
        
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})