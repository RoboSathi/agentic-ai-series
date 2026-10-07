"""Commands that exercise or describe the harness itself (LLD §20)."""

from agent_platform.commands.types import CommandError, CommandResult, ConsoleEffect
from agent_platform.config import active_features, set_active_features
from agent_platform.config.features import SEPARATOR


class HarnessCommand:
    """The harness, and every switch on it (E01).

    The series' first claim is that an agent is a model plus a harness,
    and that nearly everything worth having in production is a property
    of the second half. The claim is cheap to assert and awkward to
    show — so this makes it switchable: ask for a refund, turn the
    harness off, ask for the identical refund.

    *The harness is the set of declared features* — the controls you can
    remove and still get an agent that runs, just a worse or more
    dangerous one. Everything else the agent is made of (the control
    loop and its budgets, resilience, observability, audit, logging,
    persistence, tool calling, schema validation) is always on, because
    removing it does not produce a reckless agent, it produces a broken
    program. That line is why this command no longer reads a
    hand-maintained list: the list *is* the feature tree.

    Every form is a hot reload. Nothing is re-read from disk and nothing
    is remembered — `off` withholds every feature, `on` grants every
    feature, and a named path moves exactly one. A restart returns to
    whatever the agent's config declared. A parent moves its children
    with it; a child is named dotted (`memory.working`) or in words
    (`memory working`).

    All of them rebuild the agent: the composition root decides what to
    wire, so an agent already running was built under the old answer.
    """

    group = "Control"
    name = "harness"
    summary = "Turn the whole harness, or one feature, off and on"
    usage = "/harness [on|off] | /harness <feature> [<child>] on|off"

    def argument_completions(self, args: list[str]) -> list[str]:
        """A feature path or a bulk on/off; then a parent's children, then the switch."""
        switches = ["on", "off"]
        features = active_features()
        if not args:
            return [*features.paths(), *switches]
        if args[-1] in switches:
            return []
        children = [p for p in features.child(args[0]).paths() if SEPARATOR not in p]
        return [*children, *switches] if len(args) == 1 else switches

    async def run(self, args: list[str]) -> CommandResult:
        if not args:
            return self._report()
        if len(args) == 1:
            return self._bulk(args[0])
        return self._one(SEPARATOR.join(args[:-1]), args[-1])

    def _bulk(self, word: str) -> CommandResult:
        """Move every feature at once — the E01 gesture."""
        value = _state(word)
        set_active_features(active_features().with_all(value))
        return self._applied(f"Harness {word.lower()}")

    def _one(self, path: str, word: str) -> CommandResult:
        """Move one feature, or one sub-feature, leaving the rest alone."""
        value = _state(word)
        features = active_features()
        try:
            set_active_features(features.with_override(path, value))
        except KeyError:
            raise CommandError(
                f"No feature '{path}'. Try one of: {', '.join(features.paths())}"
            ) from None
        return self._applied(f"{path} {word.lower()}")

    def _applied(self, title: str) -> CommandResult:
        """Report the new posture and ask the console to rebuild."""
        result = self._report()
        result.title = title
        result.line("")
        result.line(
            "The model, the system prompt and the tools are unchanged."
        )
        result.effect = ConsoleEffect(rebuild_agent=True)
        return result

    def _report(self) -> CommandResult:
        """Every feature and whether it currently stands between the model
        and execution."""
        features = active_features()
        result = CommandResult(title="Harness")
        result.headers = ["feature", "state"]
        for path in features.paths():
            live = features.is_on(path)
            indent = "  " * path.count(".")
            result.row(
                f"{indent}{path.rsplit('.', maxsplit=1)[-1]}",
                "on" if live else "off",
                style="" if live else "bright_black",
            )
        return result


def _state(word: str) -> bool:
    """Read 'on' or 'off', and refuse anything else."""
    wanted = word.lower()
    if wanted not in {"on", "off"}:
        raise CommandError(f"Say 'on' or 'off', not {word!r}")
    return wanted == "on"
