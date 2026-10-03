# ADR ra-0018 — Live by default, free-tier safe; the fixture world is for tests (amends ra-0015)

## Context

`make agent ra` defaulted to the offline demo (replayed Kelvari fixtures and
the reference model), so the console could only research a synthetic
world. The user has Tavily and NVIDIA keys on free tiers (Tavily: 1,000
calls a month) and wants a real research agent.

## Decision

- The agent's config is **live**: `web.mode: live`, `model_backend:
  routed`, every role on NVIDIA NIM. `config/profiles/demo.yaml` is the
  offline posture; `make agent ra PROFILE=demo` runs it. Tests and every
  `make eval* ra` / `make data ra` pin the demo profile (evals stay
  fixtures-only, ra-0009; the plugin refuses a live posture).
- **Two search providers behind a router** (commons `web/router.py`):
  DuckDuckGo (the `ddgs` package; free, no key, best effort) answers
  general searches first; Tavily answers time-bounded/news searches first
  and is the fallback when DuckDuckGo errors or finds nothing.
- **Free-tier guard**: the platform rate limiter gains a per-bucket
  monthly quota in its shared SQLite store — Tavily 900/month (a margin
  under 1,000). A spent quota fails fast, is never retried, falls back to
  DuckDuckGo, and — if nothing can search — ends the run `PARTIAL` with the
  reason in Coverage & Limits. Live searches are cached on disk for 24h
  and memoised within a run. Budgets are sized for the tiers: Research ≤8
  searches, Q&A ≤3; two concurrent workers against NVIDIA's 35 req/min.
- **Start-up**: a missing `nvidia_api_key` stops with a sentence; a missing
  `tavily_api_key` is a warning (DuckDuckGo only). The greeting and side
  panel show the month's Tavily usage.
- **Clock**: the contract's "now" is injected — real time live, the fixture
  world's `as_of` in the demo — so recency windows over fixed fixture dates
  stay reproducible.

## Consequences

- The live path cannot be exercised in CI (no keys, no network); it is
  covered by unit tests with mock transports and a scripted DuckDuckGo
  backend, and verified by the user.
- Replayed fixtures recorded without a time bound answer time-bounded
  requests (the recorded web has no clock; the worker filters dates itself).
- The fixture set re-records automatically whenever `world.yaml` changes.

## Alternatives rejected

- **Tavily only** — spends the metered quota on searches a free engine answers.
- **Scraping DuckDuckGo's HTML ourselves** — the maintained `ddgs` package
  is the smaller, safer dependency.
- **Keep demo as the default** — the console would stay a toy.
