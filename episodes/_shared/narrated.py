"""The narrated episode shape — being retired (LLD §22).

This is how every episode was written before the harness could be
configured down: a script that hand-wires the two or three subsystems it
wants to talk about and narrates the result in prose.

It is being replaced by `runner.py`, where an episode runs the *real*
customer support agent and its own `config.yaml` decides which subsystems
are live. The reason for the change is that this shape could only ever
*describe* the claim the series makes — that an agent is a model plus a
harness — while the new one demonstrates it: turn a subsystem off in
config and its steps disappear from the trace.

Episodes still on this shape keep working unchanged. Delete this module
when the last one has moved.
"""

import asyncio
from typing import Any, Callable, Coroutine

from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentic_commons.llm.client import LLMClient, create_llm_client
from agentic_commons.llm.types import (
    CompletionRequest,
    CompletionResult,
    HealthCheckResult,
    TokenUsage,
)


class EpisodeOutput(BaseModel):
    """What an episode produced when it ran."""

    episode: str
    title: str
    principle: str = ""
    lines: list[str] = Field(default_factory=list)
    artifact: dict[str, Any] = Field(default_factory=dict)

    def say(self, line: str) -> None:
        """Record one observation for the transcript."""
        self.lines.append(line)


class OfflineLLM(LLMClient):
    """A scripted stand-in so an episode runs without a live endpoint.

    Episodes exist to show harness behaviour. Making them depend on a
    model would mean the demonstration fails when a key is missing or a
    rate limit is hit — for reasons that have nothing to do with the
    thing being demonstrated.
    """

    def __init__(self, replies: list[str] | None = None) -> None:
        self._replies = replies or ["(offline stand-in reply)"]
        self._index = 0

    @property
    def tier(self) -> str:
        return "offline"

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(healthy=True)

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        reply = self._replies[min(self._index, len(self._replies) - 1)]
        self._index += 1
        return CompletionResult(
            content=reply,
            usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            tier="offline",
        )


def episode_llm(replies: list[str] | None = None, live: bool = False) -> LLMClient:
    """A model for an episode: live only when explicitly asked for."""
    return create_llm_client() if live else OfflineLLM(replies)


def render(output: EpisodeOutput, console: Console | None = None) -> None:
    """Print an episode's transcript and its exit artifact."""
    console = console or Console()
    console.print()
    console.print(
        Panel(
            f"[bold]{output.title}[/bold]\n[dim]{output.principle}[/dim]",
            title=f"[bright_cyan]{output.episode}[/bright_cyan]",
            border_style="bright_cyan",
        )
    )
    for line in output.lines:
        console.print(f"  {line}")
    if output.artifact:
        table = Table(title="Exit artifact", title_style="bold green", show_header=False, box=None)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="white")
        for key, value in output.artifact.items():
            table.add_row(key, _format(value))
        console.print()
        console.print(table)
    console.print()


def _format(value: Any) -> str:
    """Render an artifact value compactly enough to read at a glance."""
    if isinstance(value, list):
        return "\n".join(str(item) for item in value) if value else "(none)"
    if isinstance(value, dict):
        return "\n".join(f"{k}: {v}" for k, v in value.items()) or "(none)"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main(run: Callable[[], Coroutine[Any, Any, EpisodeOutput]]) -> None:
    """Entry point shared by every episode's `__main__` block."""
    render(asyncio.run(run()))
