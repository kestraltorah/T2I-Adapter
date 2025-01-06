FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/kestraltorah/T2I-Adapter.git .

RUN pip install --no-cache-dir \
    diffusers==0.21.4 \
    transformers==4.35.2 \
    accelerate==0.24.1 \
    controlnet_aux==0.0.7 \
    huggingface_hub==0.19.4

COPY handler.py /app/handler.py

ENV PYTHONPATH=/app

CMD [ "python", "-u", "handler.py" ]