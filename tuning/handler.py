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
    "stream_options",
    "cache_prompt",
    "id_slot",
    "timings_per_token",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
}


async def handler(job):
    payload = job.get("input") or {}

    if not isinstance(payload, dict):
        yield {"error": "input must be a JSON object"}
        return

    if payload.get("p13b_action") == "diagnostics":
        async for result in diagnostics(payload):
            yield result
        return

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        yield {"error": "input.messages must be a non-empty array"}
        return

    wants_stream = bool(payload.get("stream", False))

    body = {k: v for k, v in payload.items() if k in ALLOWED_FIELDS}
    body["messages"] = messages
    body["stream"] = wants_stream
    if wants_stream:
        body["stream_options"] = {"include_usage": True}

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


async def diagnostics(payload):
    import asyncio
    import hmac
    expected = os.getenv("P13B_OPS_TOKEN", "")
    supplied = payload.get("ops_token", "")
    if os.getenv("P13B_DIAGNOSTICS", "0") != "1" or (expected and (not isinstance(supplied, str) or not hmac.compare_digest(expected, supplied))):
        yield {"error": "diagnostics unauthorized"}
        return
    result = {"configured_env": {key: os.environ.get(key) for key in (
        "CTX_SIZE", "PARALLEL", "WORKER_CONCURRENCY", "GPU_LAYERS", "BATCH_SIZE",
        "UBATCH_SIZE", "THREADS", "KV_UNIFIED", "KV_UNIFIED_PER_SLOT", "CACHE_IDLE_SLOTS",
        "CACHE_TYPE_K", "CACHE_TYPE_V", "FLASH_ATTN", "CONT_BATCHING", "CACHE_RAM_MIB"
    )}, "note": "env values are settings, not proof of actual GPU offload"}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
        for path in ("health", "slots", "props"):
            try:
                async with session.get(f"{LLAMA_URL}/{path}") as response:
                    data = await response.json()
                    if path == "slots" and isinstance(data, list):
                        safe = ("id", "n_ctx", "is_processing", "n_past", "n_tokens", "n_prompt_tokens_processed", "n_prompt_tokens")
                        result[path] = [{k: item[k] for k in safe if k in item} for item in data]
                    elif path == "props":
                        result[path] = {k: data[k] for k in ("total_slots", "default_generation_settings", "build_info") if k in data}
                    else:
                        result[path] = {"status_code": response.status}
            except Exception as exc:
                result[path] = {"error": type(exc).__name__}
        if "tokenize_content" in payload:
            content = payload["tokenize_content"]
            if not isinstance(content, str) or len(content.encode()) > 4000000:
                yield {"error": "invalid tokenize content"}
                return
            async with session.post(f"{LLAMA_URL}/tokenize", json={"content": content, "add_special": False}) as response:
                data = await response.json()
                result["token_count"] = len(data.get("tokens", []))
    try:
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,power.limit",
            "--format=csv,noheader,nounits", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        output, _ = await asyncio.wait_for(process.communicate(), 10)
        result["gpu"] = output.decode().strip() if process.returncode == 0 else "unavailable"
    except Exception as exc:
        result["gpu"] = type(exc).__name__
    yield {"ok": True, "diagnostics": result}


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
