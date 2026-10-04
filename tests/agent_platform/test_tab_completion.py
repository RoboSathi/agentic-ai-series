"""Tab completion for slash commands: complete as far as matches agree, then cycle.

And argument completion is positional: a command is told the words already
typed, so `/harness <path> <Tab>` offers the two switches and nothing else.
"""

from agent_platform.commands import CommandRegistry
from agent_platform.commands.types import CommandResult
from agent_platform.config import get_settings, set_active_features
from agent_platform.tui.completion import TabCompleter


class _Stub:
    """A command with a closed set in first position only."""

    group = "Test"
    usage = ""
    summary = ""

    def __init__(self, name: str, first: list[str]) -> None:
        self.name = name
        self._first = first

    def argument_completions(self, args: list[str]) -> list[str]:
        return [] if args else list(self._first)

    async def run(self, args: list[str]) -> CommandResult:
        return CommandResult()


def _registry() -> CommandRegistry:
    registry = CommandRegistry([])
    registry.register(_Stub("void", ["10441", "10860"]))
    registry.register(_Stub("voice", []))
    registry.register(_Stub("settle", ["10441", "all"]))
    return registry


class TestTabCompleter:
    def test_it_completes_as_far_as_every_match_agrees(self) -> None:
        assert TabCompleter(_registry()).complete("/vo") == "/voi"

    def test_pressed_again_it_cycles_through_the_matches(self) -> None:
        completer = TabCompleter(_registry())
        shared = completer.complete("/vo")
        assert shared is not None
        first = completer.complete(shared)
        second = completer.complete(first or "")
        assert {first, second} == {"/voice", "/void"}

    def test_shift_tab_cycles_backwards(self) -> None:
        completer = TabCompleter(_registry())
        shared = completer.complete("/vo")
        forward = completer.complete(shared or "")
        back = completer.complete(forward or "", step=-1)
        assert back != forward

    def test_a_single_match_completes_with_room_for_the_next_word(self) -> None:
        assert TabCompleter(_registry()).complete("/se") == "/settle "

    def test_an_argument_completes_from_the_commands_closed_set(self) -> None:
        completer = TabCompleter(_registry())
        assert completer.complete("/settle a") == "/settle all "
        assert completer.candidates("/void ") == ["/void 10441", "/void 10860"]

    def test_free_text_is_left_alone(self) -> None:
        assert TabCompleter(_registry()).complete("refund my order") is None


class TestPositionalArguments:
    def test_the_command_is_told_what_was_already_typed(self) -> None:
        registry = _registry()
        assert registry.argument_completions("/void 108") == ["10860"]
        # Second position: the stub offers nothing there.
        assert registry.argument_completions("/void 10860 ") == []

    def test_harness_offers_switches_only_after_a_path(self, restore_settings: None) -> None:
        # What a console has after building its agent: the declared feature posture.
        set_active_features(get_settings().features)
        registry = CommandRegistry()
        first = registry.argument_completions("/harness ")
        assert "gatekeeper" in first and "off" in first
        assert registry.argument_completions("/harness gatekeeper ") == ["on", "off"]
        assert registry.argument_completions("/harness off ") == []
