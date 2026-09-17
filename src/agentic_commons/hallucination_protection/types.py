"""Type definitions for hallucination protection."""

from pydantic import BaseModel


class HallucinationVerdict(BaseModel):
    """Verdict on whether a claim is hallucinated."""

    is_hallucination: bool = False
    confidence: float = 0.0
