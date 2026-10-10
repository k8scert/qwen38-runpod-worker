# qwen38-runpod-worker

P13B inference worker for the AI platform.

This repository is intentionally separate from the frozen P1-P19 deployment package.

## Responsibilities

- Build a CUDA-enabled llama.cpp worker image.
- Run a Runpod Serverless queue handler.
- Proxy OpenAI-compatible chat requests to local llama-server, including generator-based streaming.
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


## Build discipline

The workflow is manual (`workflow_dispatch`). To build the current main branch, start a **new Run workflow** on `main`; do not rely on re-running an older workflow run, because that can rebuild the old commit.

Every build publishes both:
- `k8scert/qwen38-runpod-worker:p13b-v2` for the Runpod endpoint.
- `k8scert/qwen38-runpod-worker:p13b-<git-sha>` for immutable traceability.

## Current llama.cpp defaults

- Flash Attention: on
- Context: 32768
- Parallel slots: 1
- GPU layers: 999
- Batch: 2048
- Micro-batch: 512
- CPU threads / batch threads: 8
- Reasoning: off
- Reasoning budget: 0

These values are the current A40 validation baseline and must be benchmarked before being treated as universal defaults.
