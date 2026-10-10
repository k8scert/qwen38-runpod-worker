#!/usr/bin/env bash
set -Eeuo pipefail

MODEL_REPO="${MODEL_REPO:-k8scert/Qwen3.8-27B-ABLITERATED-Q8_0}"
MODEL_FILE="${MODEL_FILE:-Qwen3.8-27B-ABLITERATED-Q8_0.gguf}"
MODEL_PATH="${MODEL_PATH:-}"
CACHE_ROOT="${RUNPOD_MODEL_CACHE_DIR:-/runpod-volume/huggingface-cache/hub}"

CTX_SIZE="${CTX_SIZE:-32768}"
PARALLEL="${PARALLEL:-1}"
GPU_LAYERS="${GPU_LAYERS:-999}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"

echo "========================================"
echo " P13B RUNPOD WORKER START"
echo "========================================"
echo "MODEL_REPO=${MODEL_REPO}"
echo "MODEL_FILE=${MODEL_FILE}"
echo "CTX_SIZE=${CTX_SIZE}"
echo "PARALLEL=${PARALLEL}"
echo "GPU_LAYERS=${GPU_LAYERS}"
echo "LLAMA_PORT=${LLAMA_PORT}"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
fi

resolve_cached_model() {
    local repo_cache
    local found

    repo_cache="${CACHE_ROOT}/models--${MODEL_REPO//\//--}"

    if [ ! -d "${repo_cache}/snapshots" ]; then
        return 1
    fi

    found="$(find -L "${repo_cache}/snapshots" -type f -name "${MODEL_FILE}" -print -quit 2>/dev/null || true)"

    if [ -z "${found}" ]; then
        return 1
    fi

    MODEL_PATH="${found}"
    export MODEL_PATH
    return 0
}

if [ -n "${MODEL_PATH}" ] && [ -s "${MODEL_PATH}" ]; then
    echo "[OK] Using explicit MODEL_PATH=${MODEL_PATH}"
elif resolve_cached_model; then
    echo "[OK] Runpod cached model found:"
    echo "     ${MODEL_PATH}"
else
    echo "[FATAL] Model file not found."
    echo "[FATAL] Expected Runpod cached model:"
    echo "        repo=${MODEL_REPO}"
    echo "        file=${MODEL_FILE}"
    echo "        cache_root=${CACHE_ROOT}"
    echo "[FATAL] Attach the Hugging Face repo in the endpoint Model/Cached Model setting."
    exit 10
fi

MODEL_SIZE="$(stat -c '%s' "${MODEL_PATH}" 2>/dev/null || echo 0)"
echo "MODEL_SIZE_BYTES=${MODEL_SIZE}"

if [ "${MODEL_SIZE}" -lt 1000000000 ]; then
    echo "[FATAL] Model file is unexpectedly small"
    exit 13
fi

/app/llama-server     --model "${MODEL_PATH}"     --host 127.0.0.1     --port "${LLAMA_PORT}"     --ctx-size "${CTX_SIZE}"     --parallel "${PARALLEL}"     --n-gpu-layers "${GPU_LAYERS}"     --flash-attn on \
    --reasoning off &

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
