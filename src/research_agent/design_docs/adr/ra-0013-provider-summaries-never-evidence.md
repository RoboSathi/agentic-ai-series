# ADR ra-0013 — Provider summaries are never evidence

## Context

Search providers write their own prose about results (Tavily's `answer`). It reads like a fact and cites nothing.

## Decision

The gateway keeps it as `SearchResponse.provider_summary`, shown as untrusted and **never turned into a page**; the verbatim-span rule then makes it impossible to cite (it is on no fetched page). The fixture world plants summaries that appear on no page.

## Consequences

- `provider_summaries_not_evidence` is a hard invariant.

## Alternatives rejected

- **Drop summaries entirely** — loses a useful display signal; keeping them visible and inert is the honest option.
