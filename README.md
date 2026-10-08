# qwen38-runpod-worker

P13B inference worker for the AI platform.

This repository is intentionally separate from the frozen P1-P19 deployment package.

## Responsibilities

- Build a CUDA-enabled llama.cpp worker image.
- Run a Runpod Serverless queue handler.
- Proxy non-streaming OpenAI-compatible chat requests to local llama-server.
- Keep model storage/acquisition outside the Docker image.

## Current baseline

- Model target: Qwen3.8-27B-Abliterated Q8_0 GGUF
- Initial context: 32K
- Initial concurrency: 1
- GPU target: 48GB class
- Model file is **not baked into the image**.

P13B-3 will define and verify the Runpod model cache/storage strategy before paid GPU activation.
