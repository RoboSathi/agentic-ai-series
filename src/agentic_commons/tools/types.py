"""Tool value types (LLD §19).

`ToolCall` and `SurfaceClass` live here, at layer 2, rather than in
`gatekeeper` at layer 4: a tool declares what it touches and how far its
effects reach, and the Gatekeeper reads those declarations. Putting them
the other way round would invert the dependency and let the mediation
point define the vocabulary it is supposed to be checking.
"""

import re
from typing import Any, Awaitable, Callable, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

# Providers validate function names against this charset and reject the
# whole request if one fails — so the separator that namespaces a tool on
# the wire cannot be the dot we use in logs and policy.
WIRE_SEPARATOR = "__"
WIRE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# What kind of surface an action touches (E13). Authority to write to a
# Locked surface cannot be granted at runtime, by anyone.
SurfaceClass = Literal["locked", "editable", "append_only", "human_controlled"]

# How far an action's effects reach if it is wrong (the Three Gates, E15).
BlastRadius = Literal["single_record", "account", "tenant", "global"]


class ToolResult(BaseModel):
    """The outcome of invoking a tool.

    A failure is a value, not an exception: the agent has to be able to
    read what went wrong and choose differently, which it cannot do with
    a traceback that unwound the loop.
    """

    success: bool = False
    output: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


ToolHandler = Union[Callable[..., ToolResult], Callable[..., Awaitable[ToolResult]]]


class ToolDefinition(BaseModel):
    """A tool's identity, contract and governance properties.

    The governance fields are part of the definition rather than an
    external policy table so that a tool cannot be added without someone
    stating what it can break.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    namespace: str
    description: str
    args_schema: type[BaseModel]
    handler: ToolHandler
    surface: SurfaceClass = "editable"
    blast_radius: BlastRadius = "single_record"
    reversible: bool = True
    requires_authority_rank: int = 5

    @property
    def qualified_name(self) -> str:
        """`namespace.name` — how the tool is named in logs and policy."""
        return f"{self.namespace}.{self.name}"

    @property
    def wire_name(self) -> str:
        """`namespace__name` — how the tool is named to a provider.

        The dot cannot go on the wire. Providers validate function names
        against `[a-zA-Z0-9_-]` and reject the request outright, so a
        namespace separator that reads well in a log is not necessarily
        one that survives the API.
        """
        return f"{self.namespace}{WIRE_SEPARATOR}{self.name}"

    def to_openai_schema(self) -> dict[str, Any]:
        """Render as an OpenAI-compatible `tools=[...]` entry.

        The parameter schema is derived from the pydantic model, never
        hand-written: a hand-kept copy drifts from the handler signature
        and the drift shows up as a runtime TypeError.
        """
        return {
            "type": "function",
            "function": {
                "name": self.wire_name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }


class ToolCall(BaseModel):
    """A governed, typed invocation — what the Gatekeeper evaluates.

    Distinct from `llm.ModelToolCall`, which is only what came back over
    the wire. A wire message becomes a `ToolCall` once it has been matched
    to a registered tool and its arguments have been validated.
    """

    action: str
    params: dict[str, Any] = Field(default_factory=dict)
    surface: SurfaceClass = "editable"
    blast_radius: BlastRadius = "single_record"
    reversible: bool = True
    call_id: str = ""


class ToolSelection(BaseModel):
    """One tool offered to the model, with the score that put it there."""

    tool: str
    score: float
    reason: str = ""
