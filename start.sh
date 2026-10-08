#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-27B-ABLITERATED-Q8_0.gguf}"
CTX_SIZE="${CTX_SIZE:-32768}"
PARALLEL="${PARALLEL:-1}"
GPU_LAYERS="${GPU_LAYERS:-999}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"

echo "========================================"
echo " P13B RUNPOD WORKER START"
echo "========================================"
echo "MODEL_PATH=${MODEL_PATH}"
echo "CTX_SIZE=${CTX_SIZE}"
echo "PARALLEL=${PARALLEL}"
echo "GPU_LAYERS=${GPU_LAYERS}"
echo "LLAMA_PORT=${LLAMA_PORT}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
fi

if [ ! -s "${MODEL_PATH}" ]; then
    echo "[FATAL] Model file not found: ${MODEL_PATH}"
    echo "[FATAL] P13B-3 must provide the model through the selected Runpod cache/storage strategy."
    exit 10
fi

/app/llama-server     --model "${MODEL_PATH}"     --host 127.0.0.1     --port "${LLAMA_PORT}"     --ctx-size "${CTX_SIZE}"     --parallel "${PARALLEL}"     --n-gpu-layers "${GPU_LAYERS}"     --flash-attn on &

LLAMA_PID=$!

cleanup() {
    kill "${LLAMA_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "[INFO] Waiting for llama-server health..."

deadline=$((SECONDS + STARTUP_TIMEOUT))

until curl -fsS "http://127.0.0.1:${LLAMA_PORT}/health" >/dev/null 2>&1; do
    if ! kill -0 "${LLAMA_PID}" >/dev/null 2>&1; then
        echo "[FATAL] llama-server exited during startup"
        wait "${LLAMA_PID}" || true
        exit 11
    fi

    if [ "${SECONDS}" -ge "${deadline}" ]; then
        echo "[FATAL] llama-server health timeout"
        exit 12
    fi

    sleep 2
done

echo "[OK] llama-server healthy"
export LLAMA_URL="http://127.0.0.1:${LLAMA_PORT}"

exec python /app/handler.py
