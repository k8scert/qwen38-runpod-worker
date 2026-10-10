import json
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
        yield {"error": "input must be a JSON object"}
        return

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        yield {"error": "input.messages must be a non-empty array"}
        return

    wants_stream = bool(payload.get("stream", False))

    body = {k: v for k, v in payload.items() if k in ALLOWED_FIELDS}
    body["messages"] = messages
    body["stream"] = wants_stream

    started = time.time()

    try:
        response = requests.post(
            f"{LLAMA_URL}/v1/chat/completions",
            json=body,
            timeout=REQUEST_TIMEOUT,
            stream=wants_stream,
        )

        if not response.ok:
            try:
                details = response.json()
            except ValueError:
                details = {"raw": response.text}

            yield {
                "error": "llama-server request failed",
                "status_code": response.status_code,
                "details": details,
            }
            return

        if not wants_stream:
            try:
                data = response.json()
            except ValueError:
                yield {
                    "error": "invalid llama-server JSON",
                    "raw": response.text,
                }
                return

            yield {
                "ok": True,
                "elapsed_seconds": round(time.time() - started, 3),
                "response": data,
            }
            return

        # llama-server emits OpenAI-compatible SSE lines.
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            line = raw_line.strip()

            if not line.startswith("data:"):
                continue

            data_text = line[5:].strip()

            if data_text == "[DONE]":
                yield {"done": True}
                return

            try:
                chunk = json.loads(data_text)
            except json.JSONDecodeError:
                continue

            yield {
                "chunk": chunk,
            }

        yield {"done": True}

    except requests.RequestException as exc:
        yield {
            "error": "llama-server unavailable",
            "details": str(exc),
        }


if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": handler,
            "return_aggregate_stream": True,
        }
    )
