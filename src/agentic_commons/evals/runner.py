"""Eval runner stubs for Phase 0."""

from agentic_commons.evals.types import EvalResult
from agentic_commons.logging import log_call


class EvalRunner:
    """Eval runner interface stub."""

    @log_call("evals")
    def run(self, suite_name: str) -> EvalResult:
        """Phase 0 stub — no real evals yet."""
        return EvalResult(suite_name=suite_name, passed=0, failed=0, score=0.0)
