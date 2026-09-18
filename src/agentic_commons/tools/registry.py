"""Namespaced tool registry and invocation (LLD §19).

The registry owns three things a tool-calling loop otherwise scatters:
what tools exist, what they are allowed to break, and how a wire-level
model request becomes a validated call. Argument validation happening
*here* is what lets the Gatekeeper downstream reason about real values
rather than a JSON string it has to trust.
"""

import inspect
import json
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError

from agentic_commons.logging import get_logger, instrument, logged
from agentic_commons.tools.types import (
    WIRE_NAME_PATTERN,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


@runtime_checkable
class ToolRegistry(Protocol):
    """Registration and lookup over the available tools."""

    def register(self, tool: ToolDefinition) -> None: ...

    def get(self, name: str) -> ToolDefinition | None: ...

    def all_tools(self) -> list[ToolDefinition]: ...


@instrument("tools")
class NamespacedToolRegistry:
    """Tools keyed by `namespace.name`, with namespace-scoped exposure.

    Namespacing is not cosmetic. E22 measures the accuracy collapse that
    comes from exposing fifty flat tool names at once; scoping exposure
    to a namespace is half of the recovery, retrieval is the other half.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._logger = get_logger("tools")

    @logged("tools")
    def register(self, tool: ToolDefinition) -> None:
        """Register a tool under its qualified name, replacing any namesake.

        The wire name is validated here rather than at call time: a name
        a provider will reject should fail at startup, not halfway
        through a customer's first conversation.
        """
        if not WIRE_NAME_PATTERN.match(tool.wire_name):
            raise ValueError(
                f"Tool '{tool.wire_name}' is not a valid provider function name; "
                f"only letters, digits, underscores and dashes are allowed"
            )
        self._tools[tool.qualified_name] = tool

    def register_all(self, tools: list[ToolDefinition]) -> None:
        """Register a batch — how a subsystem contributes its whole toolset."""
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> ToolDefinition | None:
        """Look up by qualified name, falling back to a unique bare name.

        Three spellings resolve: the qualified name used in logs and
        policy, the wire name the provider was actually shown, and a bare
        name while it is unambiguous — models routinely drop the
        namespace. Once two namespaces share a bare name the lookup
        fails rather than guessing which was meant.
        """
        if name in self._tools:
            return self._tools[name]
        by_wire = next((tool for tool in self._tools.values() if tool.wire_name == name), None)
        if by_wire is not None:
            return by_wire
        matches = [tool for tool in self._tools.values() if tool.name == name]
        return matches[0] if len(matches) == 1 else None

    def all_tools(self) -> list[ToolDefinition]:
        """Every registered tool."""
        return list(self._tools.values())

    def in_namespace(self, namespace: str) -> list[ToolDefinition]:
        """Tools belonging to one namespace."""
        return [tool for tool in self._tools.values() if tool.namespace == namespace]

    def namespaces(self) -> list[str]:
        """Every registered namespace."""
        return sorted({tool.namespace for tool in self._tools.values()})

    def to_openai_tools(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """Schemas for the named tools, or for all of them when unfiltered."""
        selected = (
            [tool for name in names if (tool := self.get(name)) is not None]
            if names is not None
            else self.all_tools()
        )
        return [tool.to_openai_schema() for tool in selected]

    def build_call(self, name: str, raw_arguments: str | dict[str, Any]) -> ToolCall | None:
        """Turn a wire-level request into a validated, governed `ToolCall`.

        Returns None when the tool is unknown or its arguments do not
        validate — both are the model's error to see and recover from,
        not the harness's to crash on.
        """
        tool = self.get(name)
        if tool is None:
            self._logger.warning("Model asked for an unregistered tool", tool=name)
            return None
        try:
            params = (
                json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
            )
            validated = tool.args_schema.model_validate(params or {})
        except (ValidationError, ValueError) as exc:
            self._logger.warning("Tool arguments failed validation", tool=name, error=str(exc))
            return None
        return ToolCall(
            action=tool.qualified_name,
            params=validated.model_dump(),
            surface=tool.surface,
            blast_radius=tool.blast_radius,
            reversible=tool.reversible,
        )

    @logged("tools")
    async def invoke(self, call: ToolCall) -> ToolResult:
        """Execute a validated call, wrapping any escape into a failed result."""
        tool = self.get(call.action)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {call.action}")
        try:
            outcome = tool.handler(**call.params)
            return await outcome if inspect.isawaitable(outcome) else outcome
        except Exception as exc:
            self._logger.exception("Tool raised", tool=call.action)
            return ToolResult(success=False, error=f"Tool '{call.action}' failed: {exc}")
