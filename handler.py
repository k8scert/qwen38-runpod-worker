import os
import time
import requests
import runpod

LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8080")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "900"))

ALLOWED_FIELDS = {
    "messages",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "n_predict",
    "seed",
    "stop",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    "response_format",
    "chat_template_kwargs",
}

def handler(job):
    payload = job.get("input") or {}

    if not isinstance(payload, dict):
        return {"error": "input must be a JSON object"}

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return {"error": "input.messages must be a non-empty array"}

    body = {k: v for k, v in payload.items() if k in ALLOWED_FIELDS}
    body["messages"] = messages
    body["stream"] = False

    started = time.time()

    try:
        response = requests.post(
            f"{LLAMA_URL}/v1/chat/completions",
            json=body,
            timeout=REQUEST_TIMEOUT,
        )

        try:
            data = response.json()
        except ValueError:
            data = {"raw": response.text}

        if not response.ok:
            return {
                "error": "llama-server request failed",
                "status_code": response.status_code,
                "details": data,
            }

        return {
            "ok": True,
            "elapsed_seconds": round(time.time() - started, 3),
            "response": data,
        }

    except requests.RequestException as exc:
        return {
            "error": "llama-server unavailable",
            "details": str(exc),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
