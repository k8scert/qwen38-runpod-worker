# Declared production image from the user's V1.2.2 package, not a moving llama tag.
FROM k8scert/qwen38-runpod-worker:p13b-6faffac2d4b737edf6a7bef086a507b0a94f5dd6
COPY baseline-diagnostics/handler.py /app/handler.py
ENV P13B_DIAGNOSTICS=1
# Keep original entrypoint/start.sh, CUDA/llama binaries and dependency versions.
