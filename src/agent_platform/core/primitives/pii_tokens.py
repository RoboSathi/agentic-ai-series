"""What a PII token looks like, as this deployment writes it (`security.pii.token_format`).

One reading of the configured shape for everything that must recognise a token without
holding a detector: the hallucination check (a token's digits are not figures a reply
states) and the reply path that puts real values back once a person has approved it.
"""

import re
from functools import cache
from string import Formatter

from agent_platform.config import get_settings

_CATEGORY = r"[A-Za-z_]+"


@cache
def _shapes(token_format: str, chars: int) -> tuple[re.Pattern[str], re.Pattern[str]]:
    token = rf"[0-9a-f]{{{chars}}}"
    framed = "".join(
        re.escape(literal) + {"category": _CATEGORY, "token": token}.get(name or "", "")
        for literal, name, _, _ in Formatter().parse(token_format)
    )
    # A token the model wrote without its frame, told from a number by a letter in it.
    bare = rf"(?<![A-Za-z0-9])(?=[0-9]*[a-f]){token}(?![A-Za-z0-9])"
    return re.compile(framed, re.IGNORECASE), re.compile(bare, re.IGNORECASE)


def framed_token() -> re.Pattern[str]:
    """A token as written, with its frame ("[PERSON:3f9a1c2b7e10]")."""
    pii = get_settings().security.pii
    return _shapes(pii.token_format, pii.token_chars)[0]


def bare_token() -> re.Pattern[str]:
    """A token with its frame lost ("3f9a1c2b7e10")."""
    pii = get_settings().security.pii
    return _shapes(pii.token_format, pii.token_chars)[1]
