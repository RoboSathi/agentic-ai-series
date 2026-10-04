"""Tab completion for slash commands, the way a shell does it (LLD §20).

Tab completes as far as every match agrees (the longest common prefix);
pressed again on its own result it cycles through the matches, and
Shift+Tab cycles back. A single match completes outright, with a trailing
space so the next argument can be typed at once.

What completes is the registry's answer, unchanged: command words always,
and an argument only where the command declares a closed set for that
position. The inline ghost text (`suggest.py`) still shows the first match;
this is the same candidates, made reachable.
"""

from os.path import commonprefix

from agent_platform.commands import PREFIX, CommandRegistry

_SPACE = " "


class TabCompleter:
    """Completes a partly typed command line, cycling through matches on repeat."""

    def __init__(self, registry: CommandRegistry) -> None:
        self._registry = registry
        self._cycle: list[str] = []
        self._index = -1
        self._last = ""

    def complete(self, value: str, step: int = 1) -> str | None:
        """The line after one Tab (step 1) or Shift+Tab (step -1); None when nothing completes."""
        if not value.startswith(PREFIX):
            return None
        if self._cycle and value == self._last:
            self._index = (self._index + step) % len(self._cycle)
            return self._remember(self._cycle[self._index])
        candidates = self.candidates(value)
        self._cycle, self._index = candidates, -1
        if not candidates:
            return None
        if len(candidates) == 1:
            # Done with this word: the next Tab completes the next argument.
            self._cycle = []
            return self._remember(candidates[0] + _SPACE)
        shared = commonprefix(candidates)
        if len(shared) > len(value):
            return self._remember(shared)
        self._index = 0 if step > 0 else len(candidates) - 1
        return self._remember(candidates[self._index])

    def candidates(self, value: str) -> list[str]:
        """Every full line the typed text could complete to."""
        if _SPACE not in value:
            return [f"{PREFIX}{name}" for name in self._registry.completions(value)]
        typed = value.rsplit(_SPACE, maxsplit=1)[0]
        return [f"{typed}{_SPACE}{match}" for match in self._registry.argument_completions(value)]

    def _remember(self, line: str) -> str:
        self._last = line
        return line
