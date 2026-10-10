import json
import os
import time

import aiohttp
import runpod

LLAMA_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:8080")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "900"))
PARALLEL = max(1, int(os.getenv("PARALLEL", "1")))
WORKER_CONCURRENCY = max(
    1,
    min(
        int(os.getenv("WORKER_CONCURRENCY", str(PARALLEL))),
        PARALLEL,
    ),
)

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


async def handler(job):
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
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{LLAMA_URL}/v1/chat/completions",
                json=body,
            ) as response:
                if response.status < 200 or response.status >= 300:
                    raw = await response.text()
                    try:
                        details = json.loads(raw)
                    except json.JSONDecodeError:
                        details = {"raw": raw}

                    yield {
                        "error": "llama-server request failed",
                        "status_code": response.status,
                        "details": details,
                    }
                    return

                if not wants_stream:
                    raw = await response.text()
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        yield {
                            "error": "invalid llama-server JSON",
                            "raw": raw,
                        }
                        return

                    yield {
                        "ok": True,
                        "elapsed_seconds": round(time.time() - started, 3),
                        "response": data,
                    }
                    return

                # llama-server emits OpenAI-compatible SSE lines.
                # aiohttp StreamReader iteration is line-oriented here, so multiple
                # concurrent jobs can await network I/O without blocking the worker loop.
                async for raw_line in response.content:
                    if not raw_line:
                        continue

                    line = raw_line.decode("utf-8", errors="replace").strip()

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

                    yield {"chunk": chunk}

                yield {"done": True}

    except (aiohttp.ClientError, TimeoutError) as exc:
        yield {
            "error": "llama-server unavailable",
            "details": str(exc),
        }


def concurrency_modifier(_current_concurrency):
    return WORKER_CONCURRENCY


if __name__ == "__main__":
    print(f"RUNPOD_WORKER_CONCURRENCY={WORKER_CONCURRENCY}", flush=True)
    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": concurrency_modifier,
            "return_aggregate_stream": True,
        }
    )
