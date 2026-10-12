"""PII detection, redaction and minimization (E32, LLD §12.2).

Detection is Presidio (spaCy NER and recognizers, `presidio.py`) plus the configured
patterns, or the patterns alone. Redaction tokenizes into the audit vault (`vault.py`).

Applied at three distinct surfaces, which is the part usually missed:
PII in *input* (minimize what a tool call carries), PII in *memory* (what
persists across sessions), and PII in *logs* (a mandatory pass before a
trace is written). Redacting only the visible response leaves the same
data in three places that outlive it.
"""

import functools
import hashlib
import hmac
import re
import secrets
from typing import Any, NamedTuple, Protocol, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.core.primitives.pii_tokens import framed_token
from agent_platform.logging import get_logger
from agent_platform.security.pii.types import PIISpan, RedactionMode
from agent_platform.security.pii.vault import PIIVault, create_pii_vault


class _Default:
    """Marks "build the configured vault", distinct from an explicit None (no vault)."""


_DEFAULT = _Default()


# Length of the per-process token key, in bytes (an HMAC-SHA256 key of full strength).
_PROCESS_KEY_BYTES = 32


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


class RedactedSpan(NamedTuple):
    """One redaction as it may be shown and logged: no value, only its token and provenance."""

    category: str
    detector: str
    score: float
    start: int
    end: int
    token: str


class PIIFilterImpl:
    """The filter used at all three surfaces (Facade + Strategy on mode).

    A filter that exists, filters. Switching PII protection off is the
    composition root declining to build one, not this object being asked
    to stand there doing nothing — a detector that is present but inert
    still appears in traces and still reads as a defence.

    `tokenize` writes `[CATEGORY:token]` with a keyed token and keeps the
    value in the vault for audit; without a vault it masks instead.
    """

    def __init__(
        self,
        detectors: list[PIIDetector] | None = None,
        vault: PIIVault | None | _Default = _DEFAULT,
    ) -> None:
        settings = get_settings().security.pii
        self._detectors: list[PIIDetector] = detectors or _default_detectors()
        self._vault = create_pii_vault() if isinstance(vault, _Default) else vault
        self._mode: RedactionMode = settings.mode
        self._mask_token = settings.mask_token
        self._token_format = settings.token_format
        self._allowed_fields = set(settings.tool_call_allowed_fields)
        self._allow_list = {value.casefold() for value in settings.allow_list}
        self._logger = get_logger("pii")

    def detect(self, text: str) -> list[PIISpan]:
        """Union of every detector's findings, de-overlapped, less the allow-listed values.

        Filtered here, once, so every engine honours the list: a support
        address the agent gives out on purpose is not a customer's data.
        """
        return _resolve_overlaps(
            [
                span
                for detector in self._detectors
                for span in detector.detect(text)
                if span.text.casefold() not in self._allow_list
            ]
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

    @property
    def model(self) -> str:
        """The model behind detection ("" for patterns alone)."""
        return next(
            (str(getattr(d, "model", "")) for d in self._detectors if getattr(d, "model", "")), ""
        )

    @property
    def unavailable(self) -> str:
        """Why the model detector fell back to patterns ("" when it did not)."""
        return next(
            (
                str(getattr(d, "unavailable", ""))
                for d in self._detectors
                if getattr(d, "unavailable", "")
            ),
            "",
        )

    def restore(self, text: str) -> str:
        """`text` with each vault token put back to the value it stands for.

        For the one path where a person has approved text that is going to the customer.
        A token the vault does not know is left as it is, and without a vault nothing is.
        """
        vault = self._vault
        if vault is None:
            return text

        def value(match: re.Match[str]) -> str:
            revealed = vault.reveal(match.group(0))
            return revealed.value if revealed is not None else match.group(0)

        return framed_token().sub(value, text)

    def tokens(self, text: str) -> list[RedactedSpan]:
        """What redacting `text` replaces: each span's category, recognizer, score and token.

        For a card and a log line — never the value itself. Empty without a vault (masking).
        """
        if self._vault is None:
            return []
        return [
            RedactedSpan(
                category=span.category,
                detector=span.detector,
                score=span.score,
                start=span.start,
                end=span.end,
                token=self._vault.token(span.category, span.text),
            )
            for span in self.detect(text)
        ]

    def tokenize(self, text: str) -> tuple[str, dict[str, str]]:
        """Text with every span replaced by its token, and token → value for what was replaced.

        Always tokens, never a mask, because the caller (the provider boundary) must be able
        to put the values back into what comes back. With a vault the token is the vault's
        (and the value is kept for an audited reveal); without one it is keyed by a secret
        this process alone holds, so it is consistent within the process and opaque outside.
        """
        spans = self.detect(text)
        if not spans:
            return text, {}
        found: dict[str, str] = {}
        tokenized = text
        for span in sorted(spans, key=lambda item: item.start, reverse=True):
            token = self._token_format.format(category=span.category, token=self._token(span))
            found[token] = span.text
            tokenized = tokenized[: span.start] + token + tokenized[span.end :]
        return tokenized, found

    def _token(self, span: PIISpan) -> str:
        if self._vault is not None:
            token = self._vault.token(span.category, span.text)
            self._vault.keep(span.category, token, span.text)
            return token
        digest = hmac.new(_process_key(), f"{span.category}|{span.text}".encode(), hashlib.sha256)
        return digest.hexdigest()[: get_settings().security.pii.token_chars]

    def _replacement(self, span: PIISpan, mode: RedactionMode) -> str:
        """The text that stands in for one redacted span."""
        if mode == "remove":
            return ""
        if mode == "tokenize" and self._vault is not None:
            token = self._vault.token(span.category, span.text)
            self._vault.keep(span.category, token, span.text)
            return self._token_format.format(category=span.category, token=token)
        return self._mask_token


@functools.cache
def _process_key() -> bytes:
    """A secret for tokens when there is no vault: this process's alone, never stored."""
    return secrets.token_bytes(_PROCESS_KEY_BYTES)


def _default_detectors() -> list[PIIDetector]:
    """Presidio (the patterns as its fallback) when configured; the patterns alone otherwise.

    Building one never loads anything: the analyzer is shared, loaded once — in the background
    by a console that starts it early, or on the first message otherwise.
    """
    patterns = PatternPIIDetector()
    if get_settings().security.pii.engine != "presidio":
        return [patterns]
    from agent_platform.security.pii.presidio import PresidioPIIDetector

    return [PresidioPIIDetector(fallback=patterns)]


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
