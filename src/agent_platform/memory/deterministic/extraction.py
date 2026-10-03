"""Deterministic claim handling: extract, screen, and find identifiers.

Regex, not a model, for the same reason the rest of the platform reaches
for the cheap tier first: "my refund limit is now $2000" has an exact
shape, and a pattern that recognises it is free, instant, auditable and
right every time. The model tier exists for what the patterns cannot
parse — and a log line records each time it was needed.
"""

import re

from pydantic import BaseModel

from agent_platform.config import get_settings
from agent_platform.config.settings import MemoryDeterministicSettings

# How a free-text key becomes a stable one: "Refund Limit" → "refund_limit".
_KEY_SEPARATORS = re.compile(r"[^a-z0-9]+")


class Claim(BaseModel):
    """A key/value claim, and which pattern recognised it."""

    key: str
    value: str
    extractor: str


class DeterministicClaims:
    """The configured patterns, compiled once."""

    def __init__(self, settings: MemoryDeterministicSettings | None = None) -> None:
        resolved = settings or get_settings().memory.deterministic
        self._extractors = [
            (item.name, re.compile(item.pattern, re.IGNORECASE), item.key)
            for item in resolved.extractors
        ]
        self._rejects = [re.compile(pattern, re.IGNORECASE) for pattern in resolved.reject_patterns]
        self._identifiers = [re.compile(pattern) for pattern in resolved.identifier_patterns]

    def extract(self, text: str) -> Claim | None:
        """The first configured pattern's claim, or None if none matches."""
        stripped = text.strip()
        for name, pattern, fixed_key in self._extractors:
            match = pattern.match(stripped)
            if match is None:
                continue
            groups = match.groupdict()
            key = fixed_key or groups.get("key") or ""
            value = (groups.get("value") or "").strip()
            if key and value:
                return Claim(key=normalize_key(key), value=value, extractor=name)
        return None

    def screen(self, text: str) -> str | None:
        """The reject pattern this text matches — memory poisoning — or None."""
        for pattern in self._rejects:
            if pattern.search(text):
                return pattern.pattern
        return None

    def identifiers(self, text: str) -> list[str]:
        """Identifier-shaped tokens in a query, in order, without duplicates."""
        found: list[str] = []
        for pattern in self._identifiers:
            for match in pattern.findall(text):
                if match not in found:
                    found.append(match)
        return found


def normalize_key(key: str) -> str:
    """A stable fact key from free text."""
    return _KEY_SEPARATORS.sub("_", key.casefold()).strip("_")
