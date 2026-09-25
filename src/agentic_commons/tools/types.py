"""Typed tool contracts used by the common execution boundary."""

import re
from typing import Any, Awaitable, Callable, Literal, Protocol, Union, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

WIRE_SEPARATOR = "__"
WIRE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

SurfaceClass = Literal["locked", "editable", "append_only", "human_controlled"]


class ToolResult(BaseModel):
    """The outcome of invoking a tool."""

    success: bool = False
    output: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


ToolHandler = Union[Callable[..., ToolResult], Callable[..., Awaitable[ToolResult]]]


class ToolDefinition(BaseModel):
    """A tool's identity, contract and deterministic execution envelope."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    namespace: str
    description: str
    args_schema: type[BaseModel]
    handler: ToolHandler
    surface: SurfaceClass = "editable"
    reversible: bool = True
    side_effecting: bool = False
    requires_authority_rank: int = 5
    required_permission: str | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.namespace}.{self.name}"

    @property
    def wire_name(self) -> str:
        return f"{self.namespace}{WIRE_SEPARATOR}{self.name}"

    def to_openai_schema(self) -> dict[str, Any]:
        """Render an OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.wire_name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }


class ToolCall(BaseModel):
    """A validated call with its execution-governance metadata attached."""

    action: str
    params: dict[str, Any] = Field(default_factory=dict)
    surface: SurfaceClass = "editable"
    reversible: bool = True
    side_effecting: bool = False
    requires_authority_rank: int = 5
    required_permission: str | None = None
    call_id: str = ""


class ToolSelection(BaseModel):
    """One tool offered to the model, with the score that put it there."""

    tool: str
    score: float
    reason: str = ""


@runtime_checkable
class ToolSource(Protocol):
    """Somewhere tools come from.

    A tool is not always a Python function the agent wrote. It can be a
    definition fetched from a remote server, generated from a schema, or
    read out of a manifest. The registry does not care which: it cares
    that whatever arrives is a `ToolDefinition` carrying the governance
    envelope the Gatekeeper needs, because a tool that arrives without
    one cannot be authorized and therefore cannot be executed.

    This is the seam an MCP server enters through (see
    `agentic_commons.mcp`), and the reason `agentic_commons` needs no
    knowledge of any particular agent's tools: the agent owns its tools,
    the platform owns the door they come in by.
    """

    async def tools(self) -> list[ToolDefinition]:
        """Every tool this source offers, as governed definitions."""
        ...
