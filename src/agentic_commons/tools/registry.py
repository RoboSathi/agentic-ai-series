"""Tool registry: register callables and expose them to an LLM as tool schemas."""

from dataclasses import dataclass
from typing import Any, Callable

from agentic_commons.logging import log_call
from agentic_commons.tools.types import ToolResult


@dataclass
class Tool:
    """A single LLM-callable tool: its schema plus the real implementation."""

    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., ToolResult]

    def to_openai_schema(self) -> dict[str, Any]:
        """Render this tool as an OpenAI-compatible `tools=[...]` entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Registry of tools an LLM can call, and the mechanism to invoke them."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    @log_call("tools")
    def register(self, tool: Tool) -> None:
        """Register a tool, making it callable by name and visible to the LLM."""
        self._tools[tool.name] = tool

    @log_call("tools")
    def invoke(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Call a registered tool by name, wrapping any exception into a failed result."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, output=f"Unknown tool: {name}")
        try:
            return tool.func(**args)
        except Exception as exc:
            return ToolResult(success=False, output=f"Tool '{name}' failed: {exc}")

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Render every registered tool as an OpenAI-compatible tools=[...] list."""
        return [tool.to_openai_schema() for tool in self._tools.values()]
