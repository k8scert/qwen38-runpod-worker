# A40 tuning candidate: evidence and deployment boundary

This image is separate from production `p13b-v2`. It adds diagnostic and parameter controls; it has not been benchmarked on the endpoint until explicitly deployed.

## What source inspection confirms

- The original start script already passes Flash Attention on, GPU layers 999, batch 2048, microbatch 512, and reasoning off.
- Modern llama-server enables continuous batching by default. Not mentioning the flag is not evidence that batching was disabled.
- The original script did not expose cache-type K/V knobs. The original worker did not forward stream_options/include_usage, cache_prompt, id_slot, or tool schemas.
- The original Docker base uses a moving server-cuda tag. Its source today cannot prove the exact installed endpoint binary.
- Endpoint environment, actual offloaded layer count, GPU power/clocks/utilization, VRAM occupancy, and slots remain unverified while management authentication is unavailable.
- A successful tool-schema forwarding unit test does not prove the model can call tools or that a tool execution environment exists.

## Candidate changes

- Explicit CACHE_TYPE_K/V (f16, q8_0, q4_0), FLASH_ATTN (on/off/auto), CONT_BATCHING (0/1), optional CACHE_RAM_MIB; unchanged baseline defaults.
- Startup prints the binary version and final llama-server arguments. Effective settings are exported to the handler.
- Private localhost metrics/slots remain behind the authenticated Runpod job interface.
- An authenticated Runpod request with input.p13b_action=diagnostics returns filtered slots/props, allowlisted effective environment, and a GPU snapshot. Optional P13B_OPS_TOKEN adds an extra gate; set P13B_DIAGNOSTICS=0 to disable diagnostics. Never expose this handler as an unauthenticated public HTTP service.
- Native stream usage is requested; cache/slot controls and tool schemas are forwarded.
- Candidate images use immutable a40-tune-<commit> tags; build artifact records the manifest digest. Production tags are not overwritten.
- Four mock handler tests cover forwarding, native usage, diagnostic authorization, and prompt/secret filtering. GPU/runtime behavior still requires endpoint acceptance.

## Controlled comparison after management access works

1. Record exact deployed image digest, llama build, GGUF file SHA/size, actual args, offload/load logs and hot GPU snapshots before any tuning.
2. Freeze model weights/quantization and prompts. Repeat each scenario five times; retain all failures. Report native tokens, cache_n/prompt_n/predicted_n, prefill, decode, first text, and end-to-end time separately.
3. Baseline: use the actual current settings, not README defaults. Compare empty/8K/32K/60K/121K cache-warm contexts.
4. Change one variable: f16 KV -> q8_0 KV, same context/slots/batches. Check long-context retrieval and code/task quality, not only speed. q4 KV is experimental and requires the same quality gate.
5. Compare 2 vs current slots and matching WORKER_CONCURRENCY, retaining enough shared KV for the primary long request plus short traffic. A 128K context budget includes input and output.
6. Test microbatch 256/512/1024 with constant batch size, then batch size; do not assume larger is faster. Test single and 1 Auto + 1 Agent and queued Pro.
7. Force CUDA kernel switches or speculative decoding only after binary support and model compatibility are proven; do not advertise them as free gains.
8. Cancel owned jobs on timeout. Continue only after terminal state is confirmed. Retest recovery and full NewAPI/UI chain after selecting a profile.
9. Use an independently isolated developer/tool environment. The inference worker is not the developer sandbox; native function calling, tool execution, account isolation and dependencies need end-to-end tests.

Official parameter reference: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

No profile is called optimal until the runtime comparison is complete.
