"""Optional host hooks and console effects: greeting, status block, cancel, forwarded turns.

A host that implements none of them (customer support) gets the console
exactly as before; one that implements them gets each behaviour.
"""

import asyncio
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

from agentic_commons.commands import CommandResult, ConsoleEffect
from agentic_commons.commands.types import Command
from agentic_commons.config import FeatureSet
from agentic_commons.host import CancellableHost, GreetingHost, StatusHost
from agentic_commons.observability import StageReporter
from agentic_commons.tui.app import AgenticTuiApp


class _Agent:
    def __init__(self) -> None:
        self.turns: list[str] = []

    async def handle(self, query: str, session_id: str) -> Any:
        self.turns.append(query)
        await asyncio.sleep(0)
        return SimpleNamespace(response=f"echo {query}")


class _Forward:
    name, summary, usage = "go", "forward a turn", "/go"

    async def run(self, args: list[str]) -> CommandResult:
        return CommandResult(effect=ConsoleEffect(agent_turn="/go now"))


class _Host:
    def __init__(self) -> None:
        self.agent = _Agent()
        self.cancelled = 0

    def title(self) -> str:
        return "Hooked"

    def placeholder(self) -> str:
        return "ask"

    def prepare(self) -> None:
        return None

    def build(self, reporter: StageReporter, features: FeatureSet | None = None) -> Any:
        return self.agent

    def commands(self) -> list[Command]:
        return [_Forward()]

    def stage_labels(self) -> Mapping[str, str]:
        return {}

    def greeting(self) -> str:
        return "Hello from the host"

    def status_lines(self, session_id: str) -> tuple[Sequence[str], Sequence[str]]:
        return [f"status for {session_id}"], [f"detail for {session_id}"]

    def cancel_key(self) -> str:
        return "ctrl+x"

    def cancel(self) -> bool:
        self.cancelled += 1
        return True


class TestOptionalHooks:
    def test_the_protocols_recognise_the_host(self) -> None:
        host = _Host()
        assert isinstance(host, GreetingHost) and isinstance(host, StatusHost)
        assert isinstance(host, CancellableHost)

    async def test_greeting_panel_cancel_and_forwarded_turns(self) -> None:
        host = _Host()
        app = AgenticTuiApp(host=host)
        async with app.run_test(size=(120, 30)) as pilot:
            await pilot.pause()
            assert "Hello from the host" in app._panel.as_text()
            app._refresh_status()
            assert "status for" in str(app._status.render())
            assert "detail for" not in str(app._status.render())
            await pilot.click("#status-line")
            assert "detail for" in str(app._status.render())
            await pilot.press("ctrl+x")
            await pilot.pause()
            assert host.cancelled == 1
            await app._run_command("/go")
            for _ in range(20):
                await pilot.pause()
                if host.agent.turns:
                    break
            assert host.agent.turns == ["/go now"]

    async def test_a_host_without_hooks_keeps_the_old_console(self) -> None:
        app = AgenticTuiApp()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            assert app._status_host is None
            assert "verify your account" in app._panel.as_text()
            app._refresh_status()
            app.action_cancel_run()
