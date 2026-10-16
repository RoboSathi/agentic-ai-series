"""The PII source the running posture declares, for every place values enter, leave or are written.

The source (`source.py`) is used far from any composition root — the Executor, the logger, the
status feed, a review queue, the reply a person approved — and each must follow whether `pii`
is on *now*: `/harness` moves it hot. So they ask here, per call, rather than being told once:
with `pii` off every function returns its input untouched, and no detector is ever built.
"""

import functools
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from agent_platform.config import active_features, get_settings
from agent_platform.core.primitives.scrubbing import install_row_observer, install_scrubber
from agent_platform.security.pii.detector import PIIFilterImpl
from agent_platform.security.pii.known_values import register_rows
from agent_platform.security.pii.source import Admitted, PiiSource

# PII configurations whose filters are kept at once (one per agent overlay a process loads).
_CONFIGURATIONS = 8


def _filter() -> PIIFilterImpl:
    """The filter for the PII configuration in force: this agent's own entities."""
    pii = get_settings().security.pii
    return _filter_for(pii.engine, tuple(pii.presidio.entities))


@functools.lru_cache(maxsize=_CONFIGURATIONS)
def _filter_for(engine: str, entities: tuple[str, ...]) -> PIIFilterImpl:
    """Keyed by what it detects, so a process that loads another agent's configuration (a
    test suite composing each agent in turn) never detects with the first one's entities."""
    return PIIFilterImpl()


@functools.cache
def _source() -> PiiSource:
    """One source per process: one token map, whichever configuration's filter is in force."""
    return PiiSource(_filter)


def pii_on() -> bool:
    """Whether the running posture declares `pii`: what a console last set, else the config.

    Before any agent is composed there is no published posture, and "nothing published" must
    not read as "pii off": a log line written at start-up is written like any other.
    """
    return active_features(get_settings().features).is_on("pii")


def admit_if_on(text: str) -> Admitted:
    """A person's message as the agent should hold it: tokenised, with what was replaced."""
    return _source().admit(text) if pii_on() else Admitted(text=text)


def tokenise_if_on(text: str) -> str:
    """`text` tokenised at the source, when the posture declares `pii`."""
    return _source().tokenise(text) if pii_on() else text


def tokenise_tree_if_on(value: Any) -> Any:
    """Every string in a payload tokenised at the source (a tool's result, a check's facts)."""
    return _source().tokenise_tree(value) if pii_on() else value


def scrub_if_on(text: str) -> str:
    """The cheap pass before anything is written: known values and patterns, no model."""
    return _source().scrub(text) if pii_on() else text


def redact_if_on(text: str) -> str:
    """`text` with its personal data tokenised, when the posture declares `pii`."""
    return tokenise_if_on(text)


def restore_if_on(text: str) -> str:
    """`text` with its tokens put back: for a tool's arguments and the reply a person reads.

    This process's own map first; a token it never made (one queued before a restart) from
    the audit vault, when there is one.
    """
    if not pii_on():
        return text
    return _filter().restore(_source().restore(text))


def restore_tree_if_on(value: Any) -> Any:
    """Every string in a payload restored (a tool call's arguments)."""
    return _source().restore_tree(value) if pii_on() else value


def install_pii_scrubbing() -> None:
    """Make the cheap pass the one every writer applies — the logger and the status feed.

    Called by each agent's composition root. Installed whatever the posture: the pass reads
    `pii` per call, so `/harness pii off` turns it off without reinstalling anything.
    """
    install_scrubber(_written)
    install_row_observer(_rows_read)


def _rows_read(query: str, rows: list[dict[str, Any]]) -> None:
    if pii_on():
        register_rows(query, rows)


def _written(value: Any) -> Any:
    """A text or a record's fields, as they may be written: the same shape, every string in it
    scrubbed. An object a writer would print is printed here first, so it is scrubbed too."""
    return _walk_written(value) if pii_on() else value


def _walk_written(value: Any) -> Any:
    if isinstance(value, str):
        return _source().scrub(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {key: _walk_written(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_walk_written(item) for item in value]
    if isinstance(value, BaseModel):
        try:
            return _walk_written(value.model_dump(mode="json"))
        except ValueError:
            # A model holding what JSON cannot (a class): written as its scrubbed text.
            return _source().scrub(str(value))
    return _source().scrub(str(value))


__all__ = [
    "admit_if_on",
    "install_pii_scrubbing",
    "pii_on",
    "redact_if_on",
    "restore_if_on",
    "restore_tree_if_on",
    "scrub_if_on",
    "tokenise_if_on",
    "tokenise_tree_if_on",
]
