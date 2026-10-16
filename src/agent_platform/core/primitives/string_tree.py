"""Every string in a payload passed through one function, with the shape kept.

The payload is made plain JSON first. A graph node's input is the agent's state *model*, not a
dict, and a walk that only knew dicts and lists passed it — and every customer string inside
it — through untouched. Whatever type arrives, nothing reaches the walk that it cannot see into.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from typing import Any

from pydantic import BaseModel

StringMap = Callable[[str], str]


def map_strings(value: Any, apply: StringMap) -> Any:
    """`value` as plain JSON, with `apply` run on every string in it (keys are left alone)."""
    return _walk(json.loads(json.dumps(value, default=_plain)), apply)


def _plain(value: Any) -> Any:
    """A JSON-encodable stand-in for an object `json` cannot encode."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (set, frozenset)):
        return list(value)
    return str(value)


def _walk(value: Any, apply: StringMap) -> Any:
    if isinstance(value, str):
        return apply(value)
    if isinstance(value, dict):
        return {key: _walk(item, apply) for key, item in value.items()}
    if isinstance(value, list):
        return [_walk(item, apply) for item in value]
    return value


__all__ = ["StringMap", "map_strings"]
