# ADR ra-0006 — Tavily behind an adapter

## Context

Search is the agent's most replaceable dependency and its most expensive to test. The API's shape (verified against Tavily's published reference: `POST /search`, Bearer auth, `{query, answer, results[{title, url, content, score, raw_content, published_date}], …}`) must not leak into the agent.

## Decision

`agent_platform.web.TavilyProvider` implements the `SearchProvider` Protocol behind `WebGateway`; the agent sees `SearchRequest → SearchResponse` only. The key is a bare lowercase env var (`tavily_api_key`). Live calls go through the platform `ResilientCaller` (bulkhead → breaker → retry → timeout) and the host-bucketed rate limiter. The adapter tolerates missing optional fields and keeps the provider's `answer` as `provider_summary` — display only (ra-0013).

## Consequences

- Replacing Tavily is one adapter.
- Record mode writes exactly what the adapter produced, so the recorded fixtures test the adapter too.

## Alternatives rejected

- **Call Tavily from the worker** — couples the agent to one vendor's JSON.
- **An SDK dependency** — adds a package for one POST.
