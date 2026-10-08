# qwen38-runpod-worker

P13B inference worker for the AI platform.

This repository is intentionally separate from the frozen P1-P19 deployment package.

## Responsibilities

- Build a CUDA-enabled llama.cpp worker image.
- Run a Runpod Serverless queue handler.
- Proxy non-streaming OpenAI-compatible chat requests to local llama-server.
- Keep model storage/acquisition outside the Docker image.
- Resolve the Runpod cached Hugging Face model automatically at startup.

## Current baseline

- Worker image: `k8scert/qwen38-runpod-worker:p13b-v2`
- Model repo: `k8scert/Qwen3.8-27B-ABLITERATED-Q8_0`
- Model file: `Qwen3.8-27B-ABLITERATED-Q8_0.gguf`
- Initial context: 32K
- Initial concurrency: 1
- GPU target: 48GB class
- Model file is **not baked into the image**.

## Runpod cached model path

Runpod mounts cached Hugging Face models under:

`/runpod-volume/huggingface-cache/hub/models--ORG--REPO/snapshots/<revision>/`

The startup script resolves the Q8_0 GGUF from that cache automatically.

P13B remains independent from the frozen P1-P19 deployment package.
