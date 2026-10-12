# resilience — Nygard's patterns, implemented

Circuit breaker, retry, timeout, bulkhead, idempotency, saga, hedge, admission, pub/sub, a durable dead-letter queue and a cross-process rate limiter, composed by `ResilientCaller` in one fixed order.

- **Entry points:** `ResilientCaller`, `CircuitBreakerImpl`, `ExponentialBackoffRetry`, `AsyncTimeoutGuard`, `ResourcePoolBulkhead`, `SqliteIdempotencyGuard`, `CompensatingSaga`, `HedgedCall`, `rate_limiter/`
- **Configuration:** `resilience:`
- **Design:** [LLD §A19](../../../design_docs/LLD.md#a19-resilience) · [ADR P-22](../../../design_docs/ADR.md#p-22--one-cross-process-rate-limit-per-provider)
