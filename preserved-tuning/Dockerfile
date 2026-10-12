FROM k8scert/qwen38-runpod-worker:p13b-6faffac2d4b737edf6a7bef086a507b0a94f5dd6
COPY baseline-diagnostics/handler.py /app/handler.py
COPY tuning/start.sh /app/start.sh
RUN chmod +x /app/start.sh
ENV P13B_DIAGNOSTICS=1
# Same declared CUDA/llama/Python engine, with explicit configurable KV/batch controls.
