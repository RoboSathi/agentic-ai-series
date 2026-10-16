"""Personal data becomes a token where it enters the harness, and a value only where it leaves.

Redacting at each place something is written — a memory note, a card, the turn summary, the
review queue — left every new place one forgotten call away from writing a customer's name. So
values are tokenised at the source instead: the person's message as the agent receives it, a
tool's result as the Executor hands it back, a check's facts as the checker returns them. From
there on the harness — its state, its checkpoints, its memory, its logs, its cards — holds only
`[PERSON:3f9a1c2b7e10]`.

A value comes back in exactly two places: the arguments a tool is called with (it must look up
the real email), and the reply the person reads. Both restore from this module's map of token
→ value, held in memory and never written; the audited vault (`make pii-reveal`) is the only
persisted way back. A token the model made up is in no map and stays a token.

Tokens are the filter's own: an HMAC of the value, the same wherever and whenever the value is
met, so a reply and the evidence it is grounded on still agree, token for token.
"""

import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.core.primitives.pii_tokens import framed_token, token_category
from agent_platform.core.primitives.string_tree import map_strings
from agent_platform.security.pii.detector import PIIFilterImpl, RedactedSpan, rule_detectors
from agent_platform.security.pii.known_values import known_values

# Where an agent carries a message's admission (`Admitted.record()`) in its state, from where
# the message entered to the stage that draws its card.
ADMITTED_KEY = "pii_admitted"


class Admitted(BaseModel):
    """A text as it entered: tokenised, what was replaced (never a value), and how long it took.

    Plain data, so an agent can carry it in its state from where the message entered to the
    stage that draws its card.
    """

    text: str
    spans: list[RedactedSpan] = Field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def categories(self) -> list[str]:
        return sorted({span.category for span in self.spans})

    def record(self) -> dict[str, Any]:
        """What the admission found, without the text, for an agent's state."""
        return self.model_dump(mode="json", exclude={"text"})

    @classmethod
    def from_record(cls, text: str, record: dict[str, Any]) -> "Admitted":
        return cls.model_validate({**record, "text": text})


class PiiSource:
    """Tokenises what enters; restores what leaves for a tool or a person.

    `scrub` is the cheap pass for everything written (a log line, a card): the known values and
    the patterns, no model, so a value that reached a writer some other way still leaves as its
    token.
    """

    def __init__(
        self,
        pii: PIIFilterImpl | Callable[[], PIIFilterImpl],
        rules: PIIFilterImpl | None = None,
        *,
        max_tokens: int | None = None,
        cache_entries: int | None = None,
    ) -> None:
        settings = get_settings().security.pii.source
        # The full filter, or how to get the one in force: the token map outlives a change of
        # configuration, the entities detected do not.
        self._pii = pii if isinstance(pii, PIIFilterImpl) else None
        self._pii_of = pii if not isinstance(pii, PIIFilterImpl) else None
        self._rules = rules or PIIFilterImpl(detectors=rule_detectors())
        self._values: OrderedDict[str, str] = OrderedDict()
        self._cache: OrderedDict[tuple[int, int, str], tuple[str, dict[str, str]]] = OrderedDict()
        self._max_tokens = max_tokens or settings.max_tokens
        self._cache_entries = cache_entries or settings.cache_entries
        self._lock = threading.Lock()

    def admit(self, text: str) -> Admitted:
        """`text` tokenised, with each replacement as a card may show it."""
        if not text:
            return Admitted(text=text)
        started = time.perf_counter()
        tokenised, found, shown = self._full().tokenize_shown(text)
        self._remember(found)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return Admitted(text=tokenised, spans=shown, latency_ms=latency_ms)

    def tokenise(self, text: str) -> str:
        """`text` with every value the full filter finds replaced by its token."""
        return self._through(self._full(), text)

    def _full(self) -> PIIFilterImpl:
        if self._pii is not None:
            return self._pii
        assert self._pii_of is not None
        return self._pii_of()

    def tokenise_tree(self, value: Any) -> Any:
        """Every string in a payload tokenised; the shape is kept."""
        return map_strings(value, self.tokenise)

    def scrub(self, text: str) -> str:
        """`text` with every known value and pattern match tokenised — no model, so cheap."""
        return self._through(self._rules, text)

    def restore(self, text: str) -> str:
        """`text` with each token this process made put back; any other token stays as it is."""
        if not text or not self._values:
            return text
        return framed_token().sub(self._value_of, text)

    def restore_tree(self, value: Any) -> Any:
        return map_strings(value, self.restore)

    def _value_of(self, match: re.Match[str]) -> str:
        return self._values.get(match.group(0), match.group(0))

    def _through(self, pii: PIIFilterImpl, text: str) -> str:
        if not text:
            return text
        # Keyed by the known values' version too: a text read before a name was registered
        # must not keep tokenising to itself once it is.
        key = (known_values().version, id(pii), text)
        with self._lock:
            cached = self._cache.get(key)
        if cached is None:
            cached = pii.tokenize(text)
            with self._lock:
                self._cache[key] = cached
                if len(self._cache) > self._cache_entries:
                    self._cache.popitem(last=False)
        tokenised, found = cached
        self._remember(found)
        return tokenised

    def _remember(self, found: dict[str, str]) -> None:
        """Keep token → value for a restore, and each value as known, so every later pass —
        the cheap one included — finds it without the model."""
        if not found:
            return
        registry = known_values()
        with self._lock:
            for token, value in found.items():
                self._values[token] = value
                self._values.move_to_end(token)
                registry.register(token_category(token), value)
            while len(self._values) > self._max_tokens:
                self._values.popitem(last=False)


__all__ = ["ADMITTED_KEY", "Admitted", "PiiSource"]
