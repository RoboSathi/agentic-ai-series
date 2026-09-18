"""Inline completion for slash commands (LLD §20).

Textual offers the completion as ghost text after the cursor, which is
the behaviour a Claude Code user already has in their fingers: type `/a`,
see `/audit`, press → to take it.

Only the command word completes. Arguments are values — order ids,
statuses, config sections — and suggesting one would be guessing at what
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
        """The first command matching what has been typed so far."""
        if not value.startswith(PREFIX) or " " in value:
            return None
        matches = self._registry.completions(value)
        return f"{PREFIX}{matches[0]}" if matches else None
