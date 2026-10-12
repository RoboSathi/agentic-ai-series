# security — identity, privacy, injection and moderation

RBAC and the agent's runtime policy; PII detection (Presidio over local spaCy), the provider boundary and the audited vault; prompt-injection and content-moderation screens, each deterministic first with a hosted guard and chat-model fallbacks.

- **Entry points:** `SecuritySuite`, `MessageScreener`, `auth/`, `pii/` (`PiiBoundary`, vault, `reveal`), `prompt_injection/`, `content_moderation/`, `GuardChain`
- **Configuration:** `security:`
- **Design:** [LLD §A12](../../../design_docs/LLD.md#a12-security) · [ADR P-15](../../../design_docs/ADR.md#p-15--pii-never-reaches-a-provider-tokens-are-reversible-only-on-audit)
