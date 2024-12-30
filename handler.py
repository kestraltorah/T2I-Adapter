import runpod
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

runpod.serverless.start({"handler": handler})
