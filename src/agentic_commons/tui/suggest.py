"""Inline completion for slash commands (LLD §20).

Textual offers the completion as ghost text after the cursor, which is
the behaviour a Claude Code user already has in their fingers: type `/a`,
see `/audit`, press → to take it.

The command word always completes. An argument completes only when the
command declares a closed set for it — `/harness` does, because a feature
path is chosen from a fixed list. Order ids, statuses and config sections
do not: those are values, and suggesting one would be guessing at what
the operator meant rather than saving them typing.
"""

from textual.suggester import Suggester

from agentic_commons.commands import PREFIX, CommandRegistry


class CommandSuggester(Suggester):
    """Completes a partially typed command from the registry."""

    def __init__(self, registry: CommandRegistry) -> None:
        # Commands are lower-case, so a case-insensitive match means
        # `/AUDIT` still completes rather than silently offering nothing.
        super().__init__(case_sensitive=False)
        self._registry = registry

    async def get_suggestion(self, value: str) -> str | None:
        """The first completion matching what has been typed so far."""
        if not value.startswith(PREFIX):
            return None
        if " " not in value:
            matches = self._registry.completions(value)
            return f"{PREFIX}{matches[0]}" if matches else None
        matches = self._registry.argument_completions(value)
        if not matches:
            return None
        typed = value.rsplit(" ", maxsplit=1)[0]
        return f"{typed} {matches[0]}"
