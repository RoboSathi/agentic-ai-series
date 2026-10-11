"""How many tokens a piece of text is — the one counter every size in the harness uses.

Sizes used to be `len(text) // chars_per_token`, written out in five places. That is a
fair guess for English prose and a poor one for JSON, order numbers and tool schemas,
which is most of what an agent sends. A model family's own tokenizer counts exactly
what its provider will bill; the character estimate stays as the fallback, and says so.

The tokenizer file is downloaded once into the repository, like every local model here,
and never on the hot path: the first count that finds it missing tries once, and a
failure latches to the estimate for the life of the process.
"""

import math
import threading
from functools import cache
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from agent_platform.config import get_settings
from agent_platform.config.settings import TokenizerSettings
from agent_platform.logging import get_logger


@runtime_checkable
class Tokenizer(Protocol):
    """Counts tokens. `exact` is False for an estimate, so a card can say which it shows."""

    @property
    def name(self) -> str: ...

    @property
    def exact(self) -> bool: ...

    def count(self, text: str) -> int: ...


class CharEstimateTokenizer:
    """Characters divided by a configured characters-per-token ratio, rounded up."""

    def __init__(self, chars_per_token: int | None = None) -> None:
        self._ratio = chars_per_token or get_settings().context.chars_per_token

    @property
    def name(self) -> str:
        return f"chars/{self._ratio}"

    @property
    def exact(self) -> bool:
        return False

    def count(self, text: str) -> int:
        return math.ceil(len(text) / self._ratio)


class LocalTokenizer:
    """A model family's `tokenizer.json`, loaded from the repository on first use.

    Degrades rather than raises: a file that is missing and cannot be fetched, or that
    does not parse, latches to the fallback once and is logged once.
    """

    def __init__(
        self,
        settings: TokenizerSettings | None = None,
        fallback: Tokenizer | None = None,
    ) -> None:
        self._settings = settings or get_settings().llm.tokenizer
        self._fallback = fallback or CharEstimateTokenizer()
        self._loaded: Any = None
        self._unavailable = not self._settings.enabled
        self._lock = threading.Lock()
        self._logger = get_logger("llm")

    @property
    def name(self) -> str:
        return self._fallback.name if self._unavailable else self._settings.model

    @property
    def exact(self) -> bool:
        return not self._unavailable and self._ready() is not None

    def count(self, text: str) -> int:
        if not text:
            return 0
        tokenizer = self._ready()
        if tokenizer is None:
            return self._fallback.count(text)
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    def _ready(self) -> Any:
        """The loaded tokenizer, loading it once; None once it has proved unavailable."""
        if self._unavailable or self._loaded is not None:
            return self._loaded
        with self._lock:
            if self._loaded is None and not self._unavailable:
                try:
                    self._loaded = self._load()
                except Exception as exc:
                    self._disable(exc)
        return self._loaded

    def _load(self) -> Any:
        from tokenizers import Tokenizer as RustTokenizer

        path = self._settings.resolved_local_dir / self._settings.file
        if not path.is_file():
            if not self._settings.download_if_missing:
                raise FileNotFoundError(f"tokenizer file missing at {path}")
            self._download(path)
        loaded = RustTokenizer.from_file(str(path))
        self._logger.info("Tokenizer loaded", model=self._settings.model, path=str(path))
        return loaded

    def _download(self, path: Path) -> None:
        from huggingface_hub import hf_hub_download

        self._logger.warning("Tokenizer missing; downloading", model=self._settings.model)
        path.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=self._settings.model, filename=self._settings.file, local_dir=str(path.parent)
        )

    def _disable(self, exc: Exception) -> None:
        self._unavailable = True
        self._logger.warning(
            "Tokenizer unavailable; counting by characters",
            model=self._settings.model,
            fallback=self._fallback.name,
            error=str(exc),
        )


@cache
def shared_tokenizer() -> Tokenizer:
    """The process's one tokenizer: the configured model family's, else the estimate."""
    return LocalTokenizer()


def count_tokens(text: str) -> int:
    """`text`'s size in tokens, by the shared tokenizer."""
    return shared_tokenizer().count(text)
