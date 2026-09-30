# ADR ra-0016 — Platform features this agent does not compose (deviation)

## Context

An agent's overlay deep-merges over the platform's feature tree, so it cannot delete keys the platform declares (`basic_sanity`, `gatekeeper`, `intent_classification`, `hallucination_guard`, `prompt_injection`, `content_moderation`, `context`).

## Decision

ra declares them **false**. It composes the platform's injection detector itself (as its own `injection_classifier` layer), and has no tool calls for a Gatekeeper to authorize. `/harness` lists them; moving them changes nothing for ra because its composition root never reads them.

## Consequences

- `/harness` shows seven inert keys for ra.

## Alternatives rejected

- **Remove keys from the platform default** — would change csa's config surface.
