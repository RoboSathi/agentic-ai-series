"""What a tool is called where people and models read it (ADR 0010).

A tool has one internal identity, its qualified name (`orders.process_refund`), used by
policy, audit, budgets and facts. It has one *model name*, the name the model is offered
and every card shows: configured (`tools.model_names`), else its bare name. The
`namespace__name` wire form is gone from everything a person or a model reads.
"""

from agent_platform.config import get_settings
from agent_platform.tools.types import WIRE_SEPARATOR

_QUALIFIED_SEPARATOR = "."
# Model name -> qualified name for every tool a registry has offered, so a card or an
# eval can name the internal identity behind a bare name no configuration aliases.
_OFFERED: dict[str, str] = {}


def remember_model_name(qualified: str, model_name: str) -> None:
    """Record the name a registry offers a tool under, for `qualified_tool_name`."""
    _OFFERED[model_name] = qualified


def configured_model_name(qualified: str, bare: str) -> str:
    """The model name a tool is registered under."""
    return get_settings().tools.model_names.get(qualified, bare)


def display_tool_name(name: str) -> str:
    """The name a card shows for any spelling of a tool: model, bare, wire or qualified."""
    names = get_settings().tools.model_names
    if name in names.values():
        return name
    qualified = name.replace(WIRE_SEPARATOR, _QUALIFIED_SEPARATOR)
    if qualified in names:
        return names[qualified]
    return qualified.rsplit(_QUALIFIED_SEPARATOR, maxsplit=1)[-1]


def qualified_tool_name(name: str) -> str:
    """The internal identity behind a model name, where config knows it; else as given."""
    for qualified, model_name in get_settings().tools.model_names.items():
        if model_name == name:
            return qualified
    if name in _OFFERED:
        return _OFFERED[name]
    return name.replace(WIRE_SEPARATOR, _QUALIFIED_SEPARATOR)
