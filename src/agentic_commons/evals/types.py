"""Type definitions for evals module."""

from pydantic import BaseModel


class EvalResult(BaseModel):
    """Result of an eval run."""

    suite_name: str = ""
    passed: int = 0
    failed: int = 0
    score: float = 0.0
