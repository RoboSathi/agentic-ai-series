"""The platform's one clock: timezone-aware UTC.

A layer-0 module so that any package, resilience included, can stamp a
time without importing `observability`, whose package import pulls in the
LLM adapter and, through it, resilience again.
"""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Timezone-aware now — never a naive timestamp in a trace or ledger."""
    return datetime.now(timezone.utc)
