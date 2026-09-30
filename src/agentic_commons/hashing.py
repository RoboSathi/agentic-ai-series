"""Canonical hashing: one stable fingerprint for any JSON-shaped value.

It started life in the eval package, where a golden case needed a hash
that survives key reordering. Frozen artifacts need the same fingerprint,
and an artifact store depending on the eval framework would invert the
layering, so the function lives here and `evals.contracts` re-exports it.
"""

import hashlib
import json
from typing import Any


def canonical_hash(value: Any) -> str:
    """SHA-256 of a value's canonical JSON — stable across key order and runs."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def text_hash(text: str) -> str:
    """SHA-256 of a text exactly as given — for page bodies and spans."""
    return hashlib.sha256(text.encode()).hexdigest()


__all__ = ["canonical_hash", "text_hash"]
