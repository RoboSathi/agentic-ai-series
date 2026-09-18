"""Commands that exercise or describe the harness itself (LLD §20)."""

from agentic_commons.commands.types import CommandResult
from agentic_commons.config import get_settings
from agentic_commons.evals import (
    TIER_ORDER,
    Claim,
    ComponentEvalRunner,
    EvalCase,
    create_scoring_cascade,
    load_manifest,
)


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
    """Show the agent's registered tools and their governance."""

    name = "tools"
    summary = "Registered tools and what each may break"
    usage = "/tools"

    async def run(self, args: list[str]) -> CommandResult:
        from agentic_commons.persistence import SqliteConnectionPool
        from customer_support_agent.core.refund_saga import RefundSaga
        from customer_support_agent.core.repositories import (
            HitlQueueRepository,
            OrdersRepository,
            RefundLedgerRepository,
        )
        from customer_support_agent.core.tools import SupportTools

        pool = SqliteConnectionPool()
        orders = OrdersRepository(pool)
        refunds = RefundLedgerRepository(pool)
        tools = SupportTools(
            orders, HitlQueueRepository(pool), refunds, RefundSaga(orders, refunds)
        )
        automation = get_settings().gatekeeper.automation

        result = CommandResult(title=f"{len(tools.definitions())} registered tools")
        result.headers = ["tool", "surface", "blast radius", "reversible", "sheridan"]
        for tool in tools.definitions():
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
