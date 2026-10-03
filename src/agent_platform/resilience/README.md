# resilience — Nygard's patterns, implemented

Circuit breaker, retry, timeout, bulkhead, idempotency, saga, pub/sub, a
durable dead-letter queue, and a cross-process rate limiter — real
implementations, composed in one fixed order by `ResilientCaller`:
bulkhead → breaker → retry → timeout → operation.

## Vocabulary

- **idempotency key** — what makes a retry safe; retrying a side effect without one is refused
- **saga** — a multi-step action with a compensation per step, undone in reverse on failure
- **dead-letter queue** — durable quarantine for requests blocked before the model
- **rate-limit bucket** — a shared request budget for a set of hosts (e.g. `nvidia`)

## Public interface

- `ResilientCaller` (`composition.py`)
- `CircuitBreakerImpl`, `ExponentialBackoffRetry`, `AsyncTimeoutGuard`, `ResourcePoolBulkhead`, `SqliteIdempotencyGuard`, `CompensatingSaga`/`SagaWorkflow`, `InMemoryPubSub`, `DeadLetterQueue`
- Rate limiter (`rate_limiter/`): `SlidingWindowRateLimiter`, `SqliteRateWindowStore` (shared by every process) / `InMemoryRateWindowStore`, `RateLimitedTransport` (httpx) and `RateLimitedHttpx2Transport` (the OpenAI SDK), `get_rate_limiter()`

## Configuration

Section `resilience:` — one block per pattern (`circuit_breaker`, `retry`, `timeout`, `bulkhead`,
`idempotency`, `saga`, `dead_letter_queue`), each with its own `enabled`, and `rate_limit`:
`store_path`, `window_seconds`, `max_wait_seconds`, `lock_timeout_seconds`,
`default_retry_after_seconds`, `buckets.<name>.{max_requests, hosts}`. NVIDIA's free tier allows 40
requests/minute per account; the default bucket is 35. Override per run:
`RPM=38 make eval` or `RESILIENCE__RATE_LIMIT__BUCKETS__NVIDIA__MAX_REQUESTS=38`.

## Seams

- Each layer is typed by its Protocol, so an agent can hand in its own breaker or retry policy and keep the stacking order.
- The limiter counts every HTTP attempt (SDK retries included) and honours `Retry-After` on a 429 for every process.

## Collaborators

- Used by `llm`, `security` (hosted guards), `executor`, and the agent's refund saga.

## Episodes

E23, E35 (resilience as a control plane).

## Tests

Run with `./venv/bin/pytest tests/unit/test_resilience.py tests/unit/test_rate_limiter.py tests/test_dead_letter_queue.py -q` (or `make test` for everything).

## Status and limits

There is no single switch for the whole section: each pattern has its own `enabled`. (A top-level `resilience.enabled` used to exist and gated nothing; it was removed.)
