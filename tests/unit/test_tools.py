"""Typed tools, the namespaced registry and retrieval selection (LLD §19)."""

import pytest
from pydantic import BaseModel, Field

from agentic_commons.evals import create_scoring_cascade
from agentic_commons.tools import (
    NamespacedToolRegistry,
    ToolCall,
    ToolDefinition,
    ToolResult,
    ToolRetrievalSelector,
)


class OrderArgs(BaseModel):
    order_id: int = Field(description="the order")


def _handler(order_id: int) -> ToolResult:
    return ToolResult(success=True, output=f"order {order_id}")


async def _async_handler(order_id: int) -> ToolResult:
    return ToolResult(success=True, output=f"async order {order_id}")


def _definition(name: str, namespace: str = "orders", **overrides: object) -> ToolDefinition:
    fields: dict[str, object] = {
        "name": name,
        "namespace": namespace,
        "description": f"does {name}",
        "args_schema": OrderArgs,
        "handler": _handler,
    }
    fields.update(overrides)
    return ToolDefinition(**fields)  # type: ignore[arg-type]


@pytest.fixture
def registry() -> NamespacedToolRegistry:
    registry = NamespacedToolRegistry()
    registry.register(
        _definition("get_order_status", description="Look up the status of a customer order")
    )
    registry.register(
        _definition(
            "process_refund",
            namespace="billing",
            description="Issue a refund to a customer for an order",
            blast_radius="account",
            reversible=False,
        )
    )
    return registry


class TestToolDefinition:
    def test_qualified_name_is_namespaced(self) -> None:
        assert _definition("x").qualified_name == "orders.x"

    def test_the_schema_is_derived_from_the_model(self) -> None:
        """A hand-kept copy drifts from the handler signature."""
        schema = _definition("x").to_openai_schema()
        assert "order_id" in schema["function"]["parameters"]["properties"]

    def test_governance_properties_have_safe_defaults(self) -> None:
        tool = _definition("x")
        assert tool.surface == "editable" and tool.reversible is True


class TestRegistry:
    def test_looks_up_by_qualified_name(self, registry: NamespacedToolRegistry) -> None:
        assert registry.get("orders.get_order_status") is not None

    def test_resolves_an_unambiguous_bare_name(self, registry: NamespacedToolRegistry) -> None:
        """Models routinely drop the namespace they were shown."""
        assert registry.get("get_order_status") is not None

    def test_refuses_an_ambiguous_bare_name(self) -> None:
        """Guessing which namespace was meant is worse than failing."""
        registry = NamespacedToolRegistry()
        registry.register(_definition("lookup", namespace="a"))
        registry.register(_definition("lookup", namespace="b"))
        assert registry.get("lookup") is None

    def test_unknown_tools_return_none(self, registry: NamespacedToolRegistry) -> None:
        assert registry.get("nope") is None

    def test_lists_namespaces(self, registry: NamespacedToolRegistry) -> None:
        assert registry.namespaces() == ["billing", "orders"]

    def test_filters_by_namespace(self, registry: NamespacedToolRegistry) -> None:
        assert len(registry.in_namespace("billing")) == 1

    def test_builds_a_validated_call(self, registry: NamespacedToolRegistry) -> None:
        call = registry.build_call("orders.get_order_status", '{"order_id": 5}')
        assert call is not None and call.params == {"order_id": 5}

    def test_a_call_carries_the_tools_governance_properties(
        self, registry: NamespacedToolRegistry
    ) -> None:
        """Which is what lets the Gatekeeper reason about it."""
        call = registry.build_call("billing.process_refund", '{"order_id": 5}')
        assert call is not None and call.reversible is False

    def test_invalid_arguments_produce_no_call(self, registry: NamespacedToolRegistry) -> None:
        """The model's error to recover from, not the harness's to crash on."""
        assert registry.build_call("orders.get_order_status", "{}") is None

    def test_malformed_json_produces_no_call(self, registry: NamespacedToolRegistry) -> None:
        assert registry.build_call("orders.get_order_status", "not json") is None

    def test_an_unknown_tool_produces_no_call(self, registry: NamespacedToolRegistry) -> None:
        assert registry.build_call("nope", "{}") is None

    async def test_invokes_a_sync_handler(self, registry: NamespacedToolRegistry) -> None:
        call = registry.build_call("orders.get_order_status", '{"order_id": 7}')
        assert call is not None
        assert (await registry.invoke(call)).output == "order 7"

    async def test_invokes_an_async_handler(self) -> None:
        registry = NamespacedToolRegistry()
        registry.register(_definition("a", handler=_async_handler))
        call = registry.build_call("orders.a", '{"order_id": 1}')
        assert call is not None
        assert (await registry.invoke(call)).output == "async order 1"

    async def test_a_raising_handler_becomes_a_failed_result(self) -> None:
        def explode(order_id: int) -> ToolResult:
            raise RuntimeError("kaboom")

        registry = NamespacedToolRegistry()
        registry.register(_definition("bad", handler=explode))
        result = await registry.invoke(ToolCall(action="orders.bad", params={"order_id": 1}))
        assert result.success is False and "kaboom" in (result.error or "")

    async def test_invoking_an_unknown_tool_fails_cleanly(
        self, registry: NamespacedToolRegistry
    ) -> None:
        result = await registry.invoke(ToolCall(action="nope"))
        assert result.success is False

    def test_renders_only_the_named_tools(self, registry: NamespacedToolRegistry) -> None:
        schemas = registry.to_openai_tools(["orders.get_order_status"])
        assert len(schemas) == 1


class TestRetrievalSelection:
    async def test_ranks_the_relevant_tool_first(self, registry: NamespacedToolRegistry) -> None:
        selector = ToolRetrievalSelector(registry, create_scoring_cascade())
        selected = await selector.select("I want my money back for order 10001")
        assert selected[0].qualified_name == "billing.process_refund"

    async def test_an_exactly_named_tool_short_circuits_ranking(
        self, registry: NamespacedToolRegistry
    ) -> None:
        selector = ToolRetrievalSelector(registry, create_scoring_cascade())
        scored = await selector.select_scored("please run orders.get_order_status")
        assert scored[0].reason == "named in the task"

    async def test_selection_respects_k(self, registry: NamespacedToolRegistry) -> None:
        selector = ToolRetrievalSelector(registry, create_scoring_cascade())
        assert len(await selector.select("refund", k=1)) == 1

    async def test_namespace_scoping_narrows_the_candidates(
        self, registry: NamespacedToolRegistry
    ) -> None:
        """Half of E22's recovery; retrieval is the other half."""
        selector = ToolRetrievalSelector(registry, create_scoring_cascade())
        selected = await selector.select("anything", namespace="billing")
        assert all(tool.namespace == "billing" for tool in selected)

    async def test_metrics_are_recorded_for_every_selection(
        self, registry: NamespacedToolRegistry
    ) -> None:
        selector = ToolRetrievalSelector(registry, create_scoring_cascade())
        await selector.select("refund")
        metrics = selector.last_metrics
        assert metrics is not None and metrics.tools_available == 2

    async def test_selection_shrinks_the_exposed_token_budget(self) -> None:
        """E22's headline: fifty tools exposed is not a neutral choice."""
        registry = NamespacedToolRegistry()
        for index in range(50):
            registry.register(_definition(f"tool_{index}", description=f"does thing {index}"))
        registry.register(
            _definition(
                "process_refund", namespace="billing", description="Issue a customer refund"
            )
        )
        selector = ToolRetrievalSelector(registry, create_scoring_cascade())
        await selector.select("refund the customer their money", k=5)
        metrics = selector.last_metrics
        assert metrics is not None
        assert metrics.tools_exposed <= 5 and metrics.tools_available == 51

    async def test_without_a_ranker_it_exposes_the_registry(
        self, registry: NamespacedToolRegistry
    ) -> None:
        assert len(await ToolRetrievalSelector(registry).select("anything")) == 2
