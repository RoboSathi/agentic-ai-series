# ADR ra-0016 — Platform features this agent does not compose (deviation)

## Context

An agent's overlay deep-merges over the platform's feature tree, so it cannot delete keys the platform declares (`deterministic_checks`, `gatekeeper`, `intent_classification`, `hallucination_guard`, `prompt_injection`, `content_moderation`, `context`).

## Decision

ra declares them **false**. It has no tool calls for a Gatekeeper to authorize. `/harness` lists them; moving them changes nothing for ra because its composition root never reads them.

**Amended 2026-10-02:** `prompt_injection` and `content_moderation` are now composed (with `pii`). They screen the *person's own message* through the platform `SecuritySuite` (`guardrails/input_screen.py`), before routing; the agent's own `injection_classifier` still screens fetched web pages. Each screen draws its own card, and a block is audited. Both are withheld in the offline fixture posture.

## Consequences

- `/harness` shows four inert keys for ra (`deterministic_checks`, `intent_classification`, `hallucination_guard`, `gatekeeper`). `context` (+ `rerank`, `failure_detection`) is composed since 2026-10-02: the direct answer's window goes through the platform's `ContextWindow`.
- Each message costs up to two hosted guard calls (jailbreak, content safety) against the NVIDIA rate-limit bucket; with `security.guards.fail_open_on_unavailable: false`, a guard outage blocks the turn.

## Alternatives rejected

- **Remove keys from the platform default** — would change csa's config surface.
