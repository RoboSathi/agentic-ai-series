"""Namespaced tool registry and validation (LLD §19).

The registry owns three things a tool-calling loop otherwise scatters:
what tools exist, what they are allowed to break, and how a wire-level
model request becomes a validated call. Argument validation happening
*here* is what lets the Gatekeeper downstream reason about real values
rather than a JSON string it has to trust. Execution deliberately does not
belong here; the common Executor is the only execution boundary.
"""

import json
from typing import Any, Protocol, runtime_checkable

from agentic_commons.guardrails import PydanticValidator
from agentic_commons.logging import get_logger, instrument, logged
from agentic_commons.tools.types import (
    WIRE_NAME_PATTERN,
    ToolCall,
    ToolDefinition,
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

    def __init__(self, validator: PydanticValidator | None = None) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        # Injected so a caller can demand strict rejection where repair
        # would be wrong. The default follows config, which is
        # `auto_repair`: a model that fenced its JSON made a formatting
        # mistake, not a authorization one.
        self._validator = validator or PydanticValidator()
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

        The validation is `guardrails`', not a second implementation of
        it living here. That matters for more than tidiness: the
        guardrail knows how to *repair* mechanically malformed output —
        a code fence around the JSON, a trailing comma — and a private
        `model_validate` in this method quietly threw those calls away
        as invalid. It also means the FR-10 invariant ("no malformed
        output reaches a side-effecting executor") is enforced by the
        component the invariant is written about.

        Returns None when the tool is unknown or its arguments cannot be
        made to validate — both are the model's error to see and recover
        from, not the harness's to crash on.
        """
        tool = self.get(name)
        if tool is None:
            self._logger.warning("Model asked for an unregistered tool", tool=name)
            return None
        result = self._validator.validate(_as_json(raw_arguments), tool.args_schema)
        if not result.valid or result.value is None:
            self._logger.warning(
                "Tool arguments failed validation", tool=name, errors=result.errors
            )
            return None
        if result.repaired:
            self._logger.info("Repaired the model's tool arguments", tool=name)
        return ToolCall(
            action=tool.qualified_name,
            params=result.value,
            surface=tool.surface,
            blast_radius=tool.blast_radius,
            path_cardinality=tool.path_cardinality,
            certifiable=tool.certifiable,
            reversible=tool.reversible,
            side_effecting=tool.side_effecting,
            requires_authority_rank=tool.requires_authority_rank,
            required_permission=tool.required_permission,
        )



def _as_json(raw_arguments: str | dict[str, Any]) -> str:
    """The arguments as a JSON string, whichever way the caller had them.

    The guardrail validates text, because text is what a model produces
    and what can be malformed. A caller that already holds a dict gets it
    re-serialised rather than handed a second validation path.
    """
    if isinstance(raw_arguments, str):
        return raw_arguments or "{}"
    return json.dumps(raw_arguments)
