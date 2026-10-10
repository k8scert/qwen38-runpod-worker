FROM ghcr.io/ggml-org/llama.cpp:server-cuda

USER root

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    MODEL_REPO=k8scert/Qwen3.8-27B-ABLITERATED-Q8_0 \
    MODEL_FILE=Qwen3.8-27B-ABLITERATED-Q8_0.gguf \
    RUNPOD_MODEL_CACHE_DIR=/runpod-volume/huggingface-cache/hub \
    CTX_SIZE=32768 \
    PARALLEL=1 \
    GPU_LAYERS=999 \
    LLAMA_PORT=8080 \
    STARTUP_TIMEOUT=900 \
    REQUEST_TIMEOUT=900 \\
    WORKER_CONCURRENCY=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 \
       python3-venv \
       ca-certificates \
       curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py
COPY start.sh /app/start.sh

RUN chmod +x /app/start.sh

HEALTHCHECK NONE

ENTRYPOINT ["/app/start.sh"]
