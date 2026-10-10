"""What a tool is called where a model or a person reads it (ADR 0010).

The model is offered a model name — configured, else the bare name — never
`namespace__name`. Policy, audit and evals keep the qualified name.
"""

from collections.abc import Iterator

import pytest
from pydantic import BaseModel

from agent_platform.config import get_settings
from agent_platform.observability.tool_fields import tool_request_fields
from agent_platform.tools import NamespacedToolRegistry
from agent_platform.tools.names import display_tool_name, qualified_tool_name
from agent_platform.tools.types import ToolDefinition, ToolResult


class _Args(BaseModel):
    order_id: int


def _tool(name: str, namespace: str = "orders") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        namespace=namespace,
        description=name,
        args_schema=_Args,
        handler=lambda order_id: ToolResult(success=True),
    )


@pytest.fixture
def renamed() -> Iterator[None]:
    """One configured verb: `orders.get_status` is offered as `look_up_order`."""
    tools = get_settings().tools
    original = dict(tools.model_names)
    tools.model_names.clear()
    tools.model_names.update({"orders.get_status": "look_up_order"})
    yield
    tools.model_names.clear()
    tools.model_names.update(original)


@pytest.fixture
def registry(renamed: None) -> NamespacedToolRegistry:
    tools = NamespacedToolRegistry()
    tools.register_all([_tool("get_status"), _tool("list_all")])
    return tools


def _offered(registry: NamespacedToolRegistry) -> list[str]:
    return [schema["function"]["name"] for schema in registry.to_openai_tools()]


class TestWhatTheModelIsOffered:
    def test_a_configured_verb_and_an_unprefixed_bare_name(
        self, registry: NamespacedToolRegistry
    ) -> None:
        assert _offered(registry) == ["look_up_order", "list_all"]

    @pytest.mark.parametrize(
        "spelling", ["look_up_order", "orders.get_status", "orders__get_status", "get_status"]
    )
    def test_every_spelling_resolves_to_one_tool(
        self, registry: NamespacedToolRegistry, spelling: str
    ) -> None:
        tool = registry.get(spelling)
        assert tool is not None and tool.qualified_name == "orders.get_status"

    def test_a_configured_name_two_tools_share_is_refused(
        self, registry: NamespacedToolRegistry
    ) -> None:
        get_settings().tools.model_names["other.get_status"] = "look_up_order"
        with pytest.raises(ValueError, match="look_up_order"):
            registry.register(_tool("get_status", namespace="other"))

    def test_a_shared_bare_name_keeps_its_namespace(self, registry: NamespacedToolRegistry) -> None:
        registry.register(_tool("list_all", namespace="other"))
        assert _offered(registry)[-1] == "other__list_all"


class TestWhatPeopleRead:
    @pytest.mark.parametrize("spelling", ["orders.get_status", "orders__get_status"])
    def test_cards_show_the_model_name(self, renamed: None, spelling: str) -> None:
        assert display_tool_name(spelling) == "look_up_order"

    def test_an_unconfigured_tool_shows_its_bare_name(self, renamed: None) -> None:
        assert display_tool_name("orders__list_all") == "list_all"

    def test_the_qualified_name_is_resolved_back(self, renamed: None) -> None:
        assert qualified_tool_name("look_up_order") == "orders.get_status"
        fields = tool_request_fields("look_up_order", '{"order_id": 1}')
        assert fields["tool"] == "look_up_order"
        assert fields["qualified_tool"] == "orders.get_status"
