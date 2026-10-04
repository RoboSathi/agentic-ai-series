"""PII value types (LLD §12.2)."""

from typing import Literal

from pydantic import BaseModel

RedactionMode = Literal["mask", "tokenize", "remove"]


class PIISpan(BaseModel):
    """One detected span of personal data.

    Offsets, not just the matched text: redaction has to rewrite the
    original string, and searching for the text again would also hit
    coincidental occurrences elsewhere.
    """

    category: str
    start: int
    end: int
    text: str
    # Which recognizer found it, and how sure it was (a configured pattern is certain).
    detector: str = "pattern"
    score: float = 1.0

    def overlaps(self, other: "PIISpan") -> bool:
        """Whether two spans cover any of the same characters."""
        return self.start < other.end and other.start < self.end
