# llm — the only place a model provider is called

Hosted and local model clients behind one `LLMClient` interface, composed into resilient, hedged, recorded chains by `ModelChainBuilder`; local encoders; rate-limited HTTP; token metering; `make models` and `make model-check`.

- **Entry points:** `LLMClient`, `ModelChainBuilder` (`chain.py`), `RoleModelRouter` (`roles.py`), `create_llm_client`, local encoders (`local_*.py`), `metering.py`, `PromptBuilder` (`prompt.py`)
- **Configuration:** `llm:`
- **Design:** [LLD §A4](../../../design_docs/LLD.md#a4-llm) · [ADR P-19](../../../design_docs/ADR.md#p-19--every-model-call-goes-through-one-chain-builder)
