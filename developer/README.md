# Isolated developer tools (internal test)

This is a separate CPU container, not part of the GPU inference worker. It provides a Bearer-authenticated OpenAPI `POST /execute` operation for Python 3.12 and Node 22 programs, with real stdout/stderr/exit status and a 1–30 second deadline. Only one execution runs at a time; extra callers wait at most 2 seconds. Finished/expired calls kill their child process group. Output files are bounded, returned output is capped, and child programs do not inherit the service token.

Installed tools: Python, Node/npm, git, curl, ripgrep, pytest, httpx, requests, FastAPI/Pydantic. The actual image stores `installed-requirements.txt`; CI imports those libraries, runs Python/JavaScript/timeout/recovery/auth/output-limit tests inside the constrained container, then publishes an immutable image.

This is a single-admin internal test environment. It is not validated for hostile multi-tenant workloads. A normal assistant answer saying it ran code is not a tool execution record.

The tool server must be connected to Open WebUI as an OpenAPI server with Bearer authentication and the model must support real native function calls. Source/protocol tests alone do not prove that end-to-end loop.

Remaining deployment prerequisites: root Docker/Compose access; sufficient CPU RAM for UI plus developer container; deployment of Adapter/worker tool-schema forwarding; configure the UI tool server and run a real model-selected Python calculation and timeout/recovery through the complete chain. Current new-win has insufficient spare RAM for the combined 512 MiB UI + 256 MiB tools budgets.

Official integration: https://docs.openwebui.com/features/extensibility/plugin/tools/openapi-servers/
