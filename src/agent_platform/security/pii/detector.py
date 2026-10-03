"""PII detection, redaction and minimization (E32, LLD §12.2).

Applied at three distinct surfaces, which is the part usually missed:
PII in *input* (minimize what a tool call carries), PII in *memory* (what
persists across sessions), and PII in *logs* (a mandatory pass before a
trace is written). Redacting only the visible response leaves the same
data in three places that outlive it.
"""

import hashlib
import re
from typing import Any, Protocol, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.logging import get_logger
from agent_platform.security.pii.types import PIISpan, RedactionMode


@runtime_checkable
class PIIDetector(Protocol):
    """Finds spans of personal data in text."""

    def detect(self, text: str) -> list[PIISpan]: ...


@runtime_checkable
class PIIFilter(Protocol):
    """Detection plus the three actions taken on what is found."""

    def detect(self, text: str) -> list[PIISpan]: ...

    def redact(self, text: str, mode: RedactionMode | None = None) -> str: ...

    def minimize(self, payload: dict[str, Any], allowed_fields: set[str]) -> dict[str, Any]: ...


class PatternPIIDetector:
    """Regex detection over the configured category patterns.

    Patterns live in config because they are jurisdiction- and
    deployment-specific: an identifier format that matters in one market
    is noise in another, and that is a policy decision, not a code change.
    """

    def __init__(self) -> None:
        settings = get_settings().security.pii
        self._patterns = {
            category: re.compile(pattern) for category, pattern in settings.patterns.items()
        }
        self._logger = get_logger("pii")

    def detect(self, text: str) -> list[PIISpan]:
        """Every match, de-overlapped so the longest span for a region wins."""
        found = [
            PIISpan(
                category=category,
                start=match.start(),
                end=match.end(),
                text=match.group(),
            )
            for category, pattern in self._patterns.items()
            for match in pattern.finditer(text)
        ]
        return _resolve_overlaps(found)


class NERPIIDetector:
    """Named-entity detection for what patterns cannot express.

    Names and addresses have no regex. Without a model this detector
    reports nothing and says so, rather than pretending coverage it does
    not have — a silently empty detector is worse than an absent one.
    """

    def __init__(self, model: PIIDetector | None = None) -> None:
        self._model = model
        self._enabled = get_settings().security.pii.ner_enabled
        self._logger = get_logger("pii")

    @property
    def available(self) -> bool:
        """Whether this detector can actually contribute anything."""
        return self._enabled and self._model is not None

    def detect(self, text: str) -> list[PIISpan]:
        """Delegate to the model when one is wired; otherwise contribute nothing."""
        if not self.available or self._model is None:
            return []
        return [span.model_copy(update={"detector": "ner"}) for span in self._model.detect(text)]


class PIIFilterImpl:
    """The filter used at all three surfaces (Facade + Strategy on mode).

    A filter that exists, filters. Switching PII protection off is the
    composition root declining to build one, not this object being asked
    to stand there doing nothing — a detector that is present but inert
    still appears in traces and still reads as a defence.
    """

    def __init__(self, detectors: list[PIIDetector] | None = None) -> None:
        settings = get_settings().security.pii
        self._detectors: list[PIIDetector] = detectors or [PatternPIIDetector(), NERPIIDetector()]
        self._mode: RedactionMode = settings.mode
        self._mask_token = settings.mask_token
        self._allowed_fields = set(settings.tool_call_allowed_fields)
        self._logger = get_logger("pii")

    def detect(self, text: str) -> list[PIISpan]:
        """Union of every detector's findings, de-overlapped."""
        return _resolve_overlaps(
            [span for detector in self._detectors for span in detector.detect(text)]
        )

    def redact(self, text: str, mode: RedactionMode | None = None) -> str:
        """Rewrite detected spans according to the redaction strategy.

        Spans are replaced right-to-left so that each replacement cannot
        shift the offsets of the ones not yet applied.
        """
        spans = self.detect(text)
        if not spans:
            return text
        strategy = mode or self._mode
        redacted = text
        for span in sorted(spans, key=lambda item: item.start, reverse=True):
            redacted = (
                redacted[: span.start] + self._replacement(span, strategy) + redacted[span.end :]
            )
        return redacted

    def minimize(
        self, payload: dict[str, Any], allowed_fields: set[str] | None = None
    ) -> dict[str, Any]:
        """Drop every field a tool call is not entitled to carry.

        An allow-list, not a deny-list: a new field added upstream is
        excluded until someone decides it belongs, which is the failure
        direction you want.
        """
        allowed = allowed_fields if allowed_fields is not None else self._allowed_fields
        dropped = set(payload) - allowed
        if dropped:
            self._logger.debug("Minimized tool-call payload", dropped=sorted(dropped))
        return {key: value for key, value in payload.items() if key in allowed}

    def _replacement(self, span: PIISpan, mode: RedactionMode) -> str:
        """The text that stands in for one redacted span."""
        if mode == "remove":
            return ""
        if mode == "tokenize":
            return f"{self._mask_token[:-1]}:{span.category}:{_stable_token(span.text)}]"
        return self._mask_token


def _stable_token(text: str) -> str:
    """A deterministic surrogate: the same value maps to the same token.

    Stability is what keeps tokenized data joinable — two records about
    the same customer still match — without the original ever persisting.
    """
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _resolve_overlaps(spans: list[PIISpan]) -> list[PIISpan]:
    """Keep the longest span wherever detections overlap.

    A card number also matches the phone pattern in part; masking the
    partial match would leave the rest of the card number in the clear.
    """
    ordered = sorted(spans, key=lambda span: (span.start, -(span.end - span.start)))
    kept: list[PIISpan] = []
    for span in ordered:
        if not any(span.overlaps(existing) for existing in kept):
            kept.append(span)
    return kept
