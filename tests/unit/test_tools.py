"""Unit tests for the real tool registry."""

from agentic_commons.tools.registry import Tool, ToolRegistry
from agentic_commons.tools.types import ToolResult


def _echo_tool() -> Tool:
    return Tool(
        name="echo",
        description="Echoes back its input.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        func=lambda text: ToolResult(success=True, output=text),
    )


def test_register_and_invoke_a_tool() -> None:
    registry = ToolRegistry()
    registry.register(_echo_tool())

    result = registry.invoke("echo", {"text": "hello"})

    assert result.success is True
    assert result.output == "hello"


def test_invoke_unknown_tool_fails_gracefully() -> None:
    registry = ToolRegistry()

    result = registry.invoke("nonexistent", {})

    assert result.success is False
    assert "nonexistent" in result.output


def test_invoke_wraps_exceptions_from_the_tool() -> None:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="boom",
            description="Always raises.",
            parameters={"type": "object", "properties": {}},
            func=lambda: (_ for _ in ()).throw(ValueError("kaboom")),
        )
    )

    result = registry.invoke("boom", {})

    assert result.success is False
    assert "kaboom" in result.output


def test_to_openai_tools_renders_registered_tools() -> None:
    registry = ToolRegistry()
    registry.register(_echo_tool())

    schemas = registry.to_openai_tools()

    assert schemas == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echoes back its input.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }
    ]


def test_to_openai_tools_empty_when_nothing_registered() -> None:
    registry = ToolRegistry()
    assert registry.to_openai_tools() == []
