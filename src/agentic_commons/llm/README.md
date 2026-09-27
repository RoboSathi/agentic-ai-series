# llm — the only place a model provider is called

Adapters for chat completion, embeddings and local encoders, behind one
`LLMClient` interface. Business logic never touches a provider SDK. Every
hosted request goes through a rate-limited HTTP client, so the shared
provider budget is enforced here rather than remembered at each call site.

## Vocabulary

- **provider** — `nim` (NVIDIA NIM, OpenAI-compatible) or `local` (llama.cpp)
- **pinned client** — a client bound to one model that is deliberately not the agent's — the eval judge, an endpoint probe
- **probe** — the smallest real request `make llm-check` sends to one hosted endpoint

## Public interface

- `LLMClient` (ABC): `complete`, `health_check`, `model`, `tier`; `OpenAICompatibleClient`, `NvidiaNimClient`, `LocalLlamaCppClient`; `CascadingLLMClient`
- `create_llm_client()`, `create_pinned_llm_client(provider, model)` (`client.py`)
- Embeddings: `EmbeddingClient`, `NimEmbeddingClient`, `HashingEmbeddingClient` (offline) (`embeddings.py`)
- Encoders: hosted NLI/reranker (`encoders.py`), local NLI (`local_nli.py`), shared local-model loader (`local_models.py`)
- `create_http_client` / `create_sdk_http_client` (`http.py`) — `httpx` and `httpx2` clients wrapped by the rate limiter
- `RecordingLLMClient` (`recording.py`) — decorator that records requests, usage and cost (used by evals)
- `build_probes`, `local_models` (`probes.py`) and `python -m agentic_commons.llm.cli` (`make llm-check`)
- Types: `Message`, `CompletionRequest`, `CompletionResult`, `ModelToolCall`, `TokenUsage` (`types.py`)

## Configuration

Section `llm:` — `provider`, `max_tokens`, `log_payloads`, `payload_max_chars`, `providers.<name>`
(base URL, model, timeout, cost rates), `cascade`, `embedding`, `encoders`, `intent_classifier`,
`check` (the probe prompt). API keys are bare env vars (`nvidia_api_key`), never config.

## Seams

- Every agent collaborator that needs a model takes an injected `LLMClient`; tests use a scripted one.

## Collaborators

- The rate limit comes from `resilience.rate_limiter`; `make llm-check` probes agent chat, eval judge, embeddings and both hosted security guards, and lists local models.

## Episodes

E05 (the model interface), E08, E44 (cascading), E48a.

## Tests

Run with `./venv/bin/pytest tests/unit/test_llm_probes.py tests/unit/test_rate_limiter.py tests/unit/test_audit_persistence_llm.py -q` (or `make test` for everything).
