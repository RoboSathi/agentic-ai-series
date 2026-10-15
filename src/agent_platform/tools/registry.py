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

from pydantic import BaseModel, ConfigDict, Field

from agent_platform.guardrails import PydanticValidator
from agent_platform.logging import get_logger, instrument, logged
from agent_platform.tools.names import configured_model_name, remember_model_name
from agent_platform.tools.types import (
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


class ToolResolution(BaseModel):
    """What a wire-level request resolved to: a call, an unknown tool, or invalid arguments."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tool: ToolDefinition | None = None
    call: ToolCall | None = None
    unknown: bool = False
    errors: list[str] = Field(default_factory=list)


@instrument("tools")
class NamespacedToolRegistry:
    """Tools keyed by `namespace.name`, with namespace-scoped exposure.

    Namespacing is not cosmetic. Tool-selection accuracy collapses
    when fifty flat tool names are exposed at once; scoping exposure
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
        """Register a tool under its qualified name, offered under its model name.

        The model name is validated here rather than at call time: a name a
        provider will reject should fail at startup, not halfway through a
        customer's first conversation. Two tools offered under one name would
        leave the model's call ambiguous, so that is refused too.
        """
        named = self._named(tool)
        if not WIRE_NAME_PATTERN.match(named.model_name):
            raise ValueError(
                f"Tool '{named.model_name}' is not a valid provider function name; "
                f"only letters, digits, underscores and dashes are allowed"
            )
        self._tools[named.qualified_name] = named
        remember_model_name(named.qualified_name, named.model_name)

    def _named(self, tool: ToolDefinition) -> ToolDefinition:
        """The tool under the name it is offered by — never one another tool already has.

        A configured name that clashes is a configuration error. A bare name that clashes
        (two namespaces with a `lookup`) keeps its namespace, as the wire name, so both stay
        callable and neither call is ambiguous.
        """
        configured = configured_model_name(tool.qualified_name, "")
        name = configured or tool.alias or tool.name
        clash = self._by_model_name(name)
        if clash is None or clash.qualified_name == tool.qualified_name:
            return tool.model_copy(update={"alias": name})
        if configured:
            raise ValueError(
                f"Tools '{clash.qualified_name}' and '{tool.qualified_name}' would both be "
                f"offered as '{name}'"
            )
        return tool.model_copy(update={"alias": tool.wire_name})

    def register_all(self, tools: list[ToolDefinition]) -> None:
        """Register a batch — how a subsystem contributes its whole toolset."""
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> ToolDefinition | None:
        """Look up by qualified name, falling back to a unique bare name.

        Four spellings resolve: the qualified name used in logs and
        policy, the model name the provider was shown, the legacy wire
        name, and a bare
        name while it is unambiguous — models routinely drop the
        namespace. Once two namespaces share a bare name the lookup
        fails rather than guessing which was meant.
        """
        if name in self._tools:
            return self._tools[name]
        offered = self._by_model_name(name)
        if offered is not None:
            return offered
        by_wire = next((tool for tool in self._tools.values() if tool.wire_name == name), None)
        if by_wire is not None:
            return by_wire
        matches = [tool for tool in self._tools.values() if tool.name == name]
        return matches[0] if len(matches) == 1 else None

    def _by_model_name(self, name: str) -> ToolDefinition | None:
        return next((tool for tool in self._tools.values() if tool.model_name == name), None)

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

    def resolve(self, name: str, raw_arguments: str | dict[str, Any]) -> ToolResolution:
        """Turn a wire-level request into a validated, governed `ToolCall`, or say why not.

        The validation is `guardrails`', not a second implementation of
        it living here. That matters for more than tidiness: the
        guardrail knows how to *repair* mechanically malformed output —
        a code fence around the JSON, a trailing comma — and a private
        `model_validate` in this method quietly threw those calls away
        as invalid. It also means the FR-10 invariant ("no malformed
        output reaches a side-effecting executor") is enforced by the
        component the invariant is written about.

        An unknown tool and invalid arguments are told apart — both are the
        model's error to see and recover from, but they are different errors,
        and tool-call validity is measured by which one it made.
        """
        tool = self.get(name)
        if tool is None:
            self._logger.warning("Model asked for an unregistered tool", tool=name)
            return ToolResolution(unknown=True)
        result = self._validator.validate(_as_json(raw_arguments), tool.args_schema)
        if not result.valid or result.value is None:
            self._logger.warning(
                "Tool arguments failed validation", tool=name, errors=result.errors
            )
            return ToolResolution(tool=tool, errors=list(result.errors))
        if result.repaired:
            self._logger.info("Repaired the model's tool arguments", tool=name)
        call = ToolCall(
            action=tool.qualified_name,
            params=result.value,
            surface=tool.surface,
            reversible=tool.reversible,
            side_effecting=tool.side_effecting,
            voidable=tool.voidable,
            requires_authority_rank=tool.requires_authority_rank,
            required_permission=tool.required_permission,
        )
        return ToolResolution(tool=tool, call=call)


def _as_json(raw_arguments: str | dict[str, Any]) -> str:
    """The arguments as a JSON string, whichever way the caller had them.

    The guardrail validates text, because text is what a model produces
    and what can be malformed. A caller that already holds a dict gets it
    re-serialised rather than handed a second validation path.
    """
    if isinstance(raw_arguments, str):
        return raw_arguments or "{}"
    return json.dumps(raw_arguments)
