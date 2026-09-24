"""Commands that exercise or describe the harness itself (LLD §20)."""

from agentic_commons.commands.types import CommandError, CommandResult, ConsoleEffect
from agentic_commons.config import active_features, get_settings, set_active_features
from agentic_commons.evals import (
    TIER_ORDER,
    Claim,
    ComponentEvalRunner,
    EvalCase,
    create_scoring_cascade,
    load_manifest,
)
from agentic_commons.tools import ToolRegistry


class EvalCommand:
    """Run the eval manifest, or show what it declares."""

    name = "eval"
    summary = "Run the component evals, or show the manifest"
    usage = "/eval [manifest|run]"

    async def run(self, args: list[str]) -> CommandResult:
        manifest = load_manifest()
        if args and args[0] == "manifest":
            result = CommandResult(title=f"Eval manifest — {len(manifest.entries)} entries")
            result.headers = ["component", "level", "scorer", "threshold"]
            for entry in manifest.entries:
                result.row(
                    entry.component,
                    entry.level,
                    entry.scorer,
                    f"{entry.regression_threshold:.0%}",
                )
            return result

        cases = [
            EvalCase(case_id="exact", component="evals.scoring", input="", expected="42"),
            EvalCase(
                case_id="grounded",
                component="evals.scoring",
                input="",
                evidence=["the refund was approved today"],
            ),
            EvalCase(
                case_id="negated",
                component="evals.scoring",
                input="",
                evidence=["the refund was denied"],
            ),
        ]
        answers = {"exact": "42", "grounded": "refund approved", "negated": "refund approved"}

        async def system(case: EvalCase) -> str:
            return answers[case.case_id]

        report = await ComponentEvalRunner().run(cases, system)
        result = CommandResult(title=f"Component evals — {report.pass_rate:.0%} pass rate")
        result.headers = ["case", "result", "tier", "why"]
        for outcome in report.outcomes:
            result.row(
                outcome.case_id,
                "pass" if outcome.passed else "fail",
                outcome.tier,
                outcome.detail[:44],
                style="" if outcome.passed else "yellow",
            )
        result.line(
            f"nollm_scorer_ratio {report.nollm_scorer_ratio:.0%} — the share settled "
            f"without a model"
        )
        result.line("The 'negated' case must not pass: it is the trap similarity falls into.")
        return result


class CascadeCommand:
    """Resolve a claim through the NOLLM cascade and report the tier."""

    name = "cascade"
    summary = "Resolve a claim through the scoring cascade"
    usage = "/cascade <claim> -- <evidence>"

    async def run(self, args: list[str]) -> CommandResult:
        result = CommandResult(title="NOLLM cascade")
        if not args:
            result.line("Tiers, cheapest first:")
            for position, tier in enumerate(TIER_ORDER, start=1):
                result.line(f"  {position}. {tier}")
            result.line("")
            result.line(f"Usage: {self.usage}")
            return result

        text = " ".join(args)
        claim, _, evidence = text.partition("--")
        cascade = create_scoring_cascade()
        verdict = await cascade.resolve(
            Claim(text=claim.strip(), evidence=[evidence.strip()] if evidence.strip() else [])
        )
        result.headers = ["field", "value"]
        result.row("claim", claim.strip())
        result.row("evidence", evidence.strip() or "(none)")
        result.row("verdict", verdict.label)
        result.row("tier", verdict.tier)
        result.row("score", f"{verdict.score:.3f}")
        result.row("why", verdict.rationale)
        return result


class ToolsCommand:
    """Show the registered tools and their governance envelope.

    Reads the registry of the agent that is actually running, rather
    than constructing a fresh copy of one product's toolset. Building
    its own meant this command could only ever describe a single agent,
    and could describe it wrongly: a tool added at composition time, or
    withheld because a feature was off, would not appear in a set built
    from scratch here.
    """

    name = "tools"
    summary = "Registered tools and what each may break"
    usage = "/tools"

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self._registry = registry

    def bind(self, registry: ToolRegistry | None) -> None:
        """Attach the running agent's registry, or detach on rebuild."""
        self._registry = registry

    async def run(self, args: list[str]) -> CommandResult:
        if self._registry is None:
            return CommandResult(title="Tools", ok=False).line(
                "No live agent is attached to this console."
            )
        definitions = self._registry.all_tools()
        automation = get_settings().gatekeeper.automation

        result = CommandResult(title=f"{len(definitions)} registered tools")
        result.headers = ["tool", "surface", "blast radius", "reversible", "sheridan"]
        for tool in definitions:
            level = automation.levels.get(tool.name, automation.human_approval_below)
            result.row(
                tool.qualified_name,
                tool.surface,
                tool.blast_radius,
                "yes" if tool.reversible else "no",
                str(level),
                style="yellow" if not tool.reversible else "",
            )
        result.line("Surface and blast radius are declared by the tool; the Gatekeeper reads them.")
        return result


class GatekeeperCommand:
    """Show the authority hierarchy and the surface policy."""

    name = "gatekeeper"
    summary = "Authority hierarchy, surfaces and thresholds"
    usage = "/gatekeeper"

    async def run(self, args: list[str]) -> CommandResult:
        settings = get_settings().gatekeeper
        result = CommandResult(title="Gatekeeper")
        result.headers = ["rank", "source", "may grant"]
        for rank, source in enumerate(settings.authority_hierarchy, start=1):
            grants = rank <= settings.authority_grant_max_rank
            result.row(
                str(rank), source, "yes" if grants else "no", style="" if grants else "bright_black"
            )
        result.line("")
        result.line("Surfaces:")
        for action, surface in settings.surface_policy.surfaces.items():
            result.line(f"  {action:<24} {surface}")
        result.line("")
        result.line("Cumulative thresholds:")
        for action_class, threshold in settings.ledger.thresholds_usd.items():
            result.line(
                f"  {action_class:<24} ${threshold:,.0f} per {settings.ledger.window_hours}h"
            )
        return result


class PatternsCommand:
    """Show the WF/AG pattern catalogue."""

    name = "patterns"
    summary = "The workflow/agentic pattern catalogue"
    usage = "/patterns [WF|AG]"

    async def run(self, args: list[str]) -> CommandResult:
        from design_patterns import ALL_PATTERNS, by_family

        family = args[0].upper() if args else ""
        cards = by_family(family) if family in {"WF", "AG"} else ALL_PATTERNS
        result = CommandResult(title=f"{len(cards)} patterns")
        result.headers = ["pattern", "family", "depth", "use when"]
        for card in cards:
            result.row(
                card.name,
                card.family,
                card.depth_episode,
                card.use_when[:48],
                style="yellow" if not card.gates_pass else "",
            )
        result.line("Rows in yellow are the ones the Three Gates say need a human in the line.")
        return result


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
    whatever the agent's config declared.

    All of them rebuild the agent: the composition root decides what to
    wire, so an agent already running was built under the old answer.
    """

    name = "harness"
    summary = "Turn the whole harness, or one feature, off and on"
    usage = "/harness [on|off] | /harness <feature> [on|off]"

    def argument_completions(self) -> list[str]:
        """Every feature path, plus the two words that move one."""
        return [*active_features().paths(), "on", "off"]

    async def run(self, args: list[str]) -> CommandResult:
        if not args:
            return self._report()
        if len(args) == 1:
            return self._bulk(args[0])
        return self._one(args[0], args[1])

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
            "The model, the prompt and the tools are unchanged. Ask the same question again."
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
