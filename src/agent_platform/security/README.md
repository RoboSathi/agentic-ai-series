# security — identity, privacy, injection and moderation

The security plane: who is acting and with what permissions (RBAC and
the agent's runtime policy), PII detection and redaction, prompt-injection
screening (deterministic phrases, delimiting untrusted data, tool-scope
limiting) and content moderation — each paired with an optional hosted
NVIDIA guard model and its fallbacks (chat models prompted as the same
classifier, asked in turn); the request fails closed only when every one
is unavailable.

## Vocabulary

- **AgentPolicy** — the agent's role, authority rank, permissions and per-action / window limits, read from its database
- **untrusted data** — user or tool text wrapped in delimiters so it is never read as instructions
- **guard verdict** — `allow`, `flagged` or `unavailable` from a hosted guard model
- **guard chain** — a guard and its fallbacks; the first verdict that is not `unavailable` wins, and says which model gave it

## Public interface

- `SecuritySuite` — one facade over every control (`suite.py`)
- Auth: `Authorizer`, `AgentPolicy`, `AgentPolicyRepository`, `Principal`, `Permission` (`auth/`)
- PII: `PIIFilterImpl` — detect, redact, minimise (`pii/`)
- Injection: `PromptInjectionGuard`, `HostedJailbreakGuard`, `ChatJailbreakGuard` (a chat fallback) (`prompt_injection/`)
- Moderation: `TermContentModerator`, `HostedContentSafetyGuard` (`content_moderation/`)
- `GuardChain` (`guard_chain.py`) — a guard and its fallbacks, asked in turn
- `post_json_with_poll` (`remote_http.py`) — hosted-guard HTTP, rate-limited

## Configuration

Section `security:` — `auth`, `pii`, `prompt_injection`, `content_moderation`, `guards`
(`enabled`, `fail_open_on_unavailable`, `jailbreak`, `content_safety` — each with `fallbacks`;
the jailbreak fallbacks' `classifier_prompt`, `verdict_key`, `score_key` — and `fallback_reason`). Whether each screening stage
exists: `features.prompt_injection`, `features.content_moderation`, `features.pii`.

## Seams

- An agent overlays its own suspicious phrases and allowed tool-call fields.

## Collaborators

- Used by the agent's screening node, `gatekeeper` (authority) and `memory` (redaction). Attacks against it are the eval security suite.

## Episodes

E02, E12, E30–E33, E36, E48b.

## Tests

Run with `./venv/bin/pytest tests/unit/test_security.py tests/test_hosted_guards.py tests/integration/test_e12_attacks.py -q` (or `make test` for everything).
