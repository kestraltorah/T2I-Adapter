FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/kestraltorah/T2I-Adapter.git .

RUN pip install -r requirements.txt && \
    pip install runpod \
    diffusers>=0.21.4 \
    transformers \
    accelerate \
    safetensors \
    controlnet_aux==0.0.7 \
    xformers

COPY handler.py /app/handler.py

ENV PYTHONPATH=/app

CMD [ "python", "-u", "handler.py" ]