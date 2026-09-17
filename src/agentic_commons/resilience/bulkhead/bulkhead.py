"""Bulkhead pattern stub for Phase 0."""

from contextlib import contextmanager
from typing import Generator

from agentic_commons.logging import log_call


class Bulkhead:
    """Bulkhead pattern stub."""

    @log_call("resilience")
    @contextmanager
    def acquire(self, pool: str) -> Generator[None, None, None]:
        """Phase 0 stub — no real bulkhead yet."""
        yield
