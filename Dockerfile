FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglu1-mesa \
    libgl1 \
    libglvnd0 \
    libgl1-mesa-dri \
    libglx0

RUN git clone https://github.com/kestraltorah/T2I-Adapter.git .

RUN pip install --no-cache-dir \
    diffusers==0.21.4 \
    transformers==4.35.2 \
    accelerate==0.24.1 \
    controlnet_aux==0.0.7 \
    huggingface_hub==0.19.4 \
    runpod \
    omegaconf \
    datasets \
    pytorch_lightning \
    gradio \
    mediapipe

COPY handler.py /app/handler.py

ENV PYTHONPATH=/app

CMD [ "python", "-u", "handler.py" ]