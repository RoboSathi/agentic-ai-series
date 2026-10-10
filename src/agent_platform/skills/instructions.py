"""The agent profile and one turn's procedures, as the system prompt carries them.

`AgentInstructions` is the whole interface an agent sees. `select` picks this turn's
procedures without a model call; an agent that runs a procedure's prefetch reads does so
next; `announce` records the turn once (one PROCEDURE card and one audit entry, and nothing
at all when no procedure was used); `render` writes the prompt text. `compose` does all three
for an agent with nothing to do in between.

Procedures are synced at build time, and the build refuses what would slow or mislead the
model: a profile and procedures longer than the agent's prompt budget, a procedure naming a
tool the agent does not offer, or a prefetch of a tool with a side effect.
"""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from agent_platform.audit.ledger import AuditLedger
from agent_platform.audit.types import AuditEntry
from agent_platform.config.settings import SkillSettings
from agent_platform.logging import get_logger
from agent_platform.logging.trace_context import get_current_trace_id
from agent_platform.memory.procedural.memory import ProceduralStore
from agent_platform.memory.write_pipeline import MemoryReporter
from agent_platform.skills.catalog import SkillCatalog
from agent_platform.skills.document import (
    SkillDocumentError,
    load_profile,
    load_skill_documents,
)
from agent_platform.skills.selector import SkillSelector
from agent_platform.skills.types import (
    ActivatedSkill,
    Activation,
    AgentProfile,
    Instructions,
    Skill,
    SkillSyncReport,
)
from agent_platform.tools.registry import ToolRegistry


@runtime_checkable
class AgentInstructions(Protocol):
    """What an agent's prompt builder asks of its profile and procedures."""

    @property
    def catalog(self) -> SkillCatalog | None: ...

    def profile_section(self) -> str: ...

    def select(self, signals: frozenset[str], query: str) -> Activation: ...

    def announce(
        self, activation: Activation, *, prefetched: Sequence[str] = (), offered: Sequence[str] = ()
    ) -> None: ...

    def render(self, activation: Activation) -> str: ...

    def compose(self, signals: frozenset[str], query: str) -> Instructions: ...


class InstructionBuilder:
    """The agent profile plus selected procedures; the profile alone without a catalog."""

    def __init__(
        self,
        profile: AgentProfile,
        settings: SkillSettings,
        *,
        catalog: SkillCatalog | None = None,
        audit: AuditLedger | None = None,
        reporter: MemoryReporter | None = None,
    ) -> None:
        self._profile = profile
        self._settings = settings
        self._catalog = catalog
        self._selector = SkillSelector(settings)
        self._audit = audit
        self._reporter = reporter
        self._announced: tuple[object, ...] | None = None

    @property
    def catalog(self) -> SkillCatalog | None:
        return self._catalog

    def profile_section(self) -> str:
        return self._profile.text.strip()

    def select(self, signals: frozenset[str], query: str) -> Activation:
        """This turn's procedures, chosen without a model call (none without a catalog)."""
        if self._catalog is None:
            return Activation(profile=self._profile)
        chosen = self._selector.select(self._catalog.active(), signals, query)
        return Activation(profile=self._profile, skills=chosen)

    def render(self, activation: Activation) -> str:
        """The profile, then each procedure's sent sections under its heading."""
        render = self._settings.render
        sections = [self.profile_section()]
        for item in activation.skills:
            heading = render.procedure_heading.format(name=item.skill.name)
            sections.append(render.separator.join([heading, item.skill.document.prompt]))
        return render.separator.join(sections)

    def compose(self, signals: frozenset[str], query: str) -> Instructions:
        """Select, announce and render in one step."""
        activation = self.select(signals, query)
        self.announce(activation)
        return Instructions(text=self.render(activation), activation=activation)

    def announce(
        self, activation: Activation, *, prefetched: Sequence[str] = (), offered: Sequence[str] = ()
    ) -> None:
        """One PROCEDURE card and one audit entry per turn — and nothing when none was used."""
        if not activation.skills:
            return
        refs = tuple(item.skill.ref(self._settings.render.ref) for item in activation.skills)
        key = (get_current_trace_id(), refs)
        if key == self._announced:
            return
        self._announced = key
        self._show(activation, list(prefetched), list(offered))
        self._record(activation)

    def _show(self, activation: Activation, prefetched: list[str], offered: list[str]) -> None:
        if self._reporter is None:
            return
        words = self._settings.observability
        summary = words.joiner.join(_line(item, words.activated) for item in activation.skills)
        # Prefetch details belong in the expanded Procedure card, not its compact heading.
        summary += words.offered.format(tools=", ".join(offered)) if offered else words.offered_none
        self._reporter.emit(
            words.activation_stage,
            "completed",
            event_kind="memory",
            summary=summary,
            fields={
                "procedures": [_card_fields(item) for item in activation.skills],
                # Drawn before any of them runs: what the harness reads before the model is
                # asked, and what the model may then call (a read already made is not offered).
                "harness_reads_first": prefetched,
                "model_may_call": offered,
            },
        )

    def _record(self, activation: Activation) -> None:
        if self._audit is None:
            return
        render, audit = self._settings.render, self._settings.audit
        refs = [item.skill.ref(render.ref) for item in activation.skills]
        self._audit.record(
            AuditEntry(
                trace_id=get_current_trace_id() or audit.activation_action,
                actor=audit.actor,
                action=audit.activation_action,
                authority_context={
                    "profile": render.profile_ref.format(digest=activation.profile.digest),
                    "procedures": refs,
                    "reasons": {
                        item.skill.ref(render.ref): item.reason for item in activation.skills
                    },
                },
                decision="allow",
                rationale=", ".join(refs),
            )
        )


def _line(item: ActivatedSkill, template: str) -> str:
    skill = item.skill
    return template.format(name=skill.name, version=skill.version, reason=item.reason)


def _card_fields(item: ActivatedSkill) -> dict[str, Any]:
    skill = item.skill
    return {
        "procedure": f"{skill.name} v{skill.version}",
        "chosen_because": item.reason,
        "steps": list(skill.document.steps),
    }


def build_agent_instructions(
    settings: SkillSettings,
    *,
    store: ProceduralStore | None,
    audit: AuditLedger | None,
    reporter: MemoryReporter | None,
    tools: ToolRegistry | None = None,
) -> InstructionBuilder:
    """The composition root's one call: the profile always, procedures if memory has them.

    `store` is None when the agent's posture withholds procedural memory; the agent then
    runs on its profile alone, with nothing synced. `tools`, when the agent's tools are
    known at build time, is what every procedure's `tools` and `prefetch` are checked
    against.
    """
    profile = load_profile(settings)
    if store is None:
        return InstructionBuilder(profile, settings)
    catalog = SkillCatalog(store, settings, audit=audit)
    report = catalog.sync(load_skill_documents(settings))
    active = catalog.active()
    _check_always(active, settings)
    _check_budget(profile, active, settings)
    if tools is not None:
        _check_tools(active, tools, settings)
    _show_sync(report, settings, reporter)
    return InstructionBuilder(profile, settings, catalog=catalog, audit=audit, reporter=reporter)


def render_profile(settings: SkillSettings) -> str:
    """The profile as the prompt carries it, without building anything else."""
    return InstructionBuilder(load_profile(settings), settings).profile_section()


def _check_always(active: list[Skill], settings: SkillSettings) -> None:
    """More `always` skills than fit one prompt would silently drop one."""
    always = [s.name for s in active if settings.always_trigger in s.document.triggers]
    if len(always) > settings.selection.max_active:
        reason = f"{len(always)} '{settings.always_trigger}' skills exceed selection.max_active"
        raise SkillDocumentError(settings.skills_dir, reason)


def _check_budget(profile: AgentProfile, active: list[Skill], settings: SkillSettings) -> None:
    """The profile and the largest procedures that could join it must fit the budget."""
    budget = settings.max_prompt_chars
    if not budget:
        return
    sizes = sorted((len(s.document.prompt) for s in active), reverse=True)
    worst = len(profile.text) + sum(sizes[: settings.selection.max_active])
    if worst > budget:
        reason = f"profile and procedures reach {worst} chars, over max_prompt_chars {budget}"
        raise SkillDocumentError(settings.skills_dir, reason)


def _check_tools(active: list[Skill], tools: ToolRegistry, settings: SkillSettings) -> None:
    """Every tool a procedure names is offered by that name; a prefetch only reads."""
    for skill in active:
        for name in (*skill.document.tools, *skill.document.prefetch):
            tool = tools.get(name)
            if tool is None or tool.model_name != name:
                raise SkillDocumentError(skill.name, f"names '{name}', which is not offered")
        for name in skill.document.prefetch:
            tool = tools.get(name)
            if tool is not None and tool.side_effecting:
                raise SkillDocumentError(skill.name, f"prefetches '{name}', which has an effect")


def _show_sync(
    report: SkillSyncReport, settings: SkillSettings, reporter: MemoryReporter | None
) -> None:
    """A card when the sync changed something; one log line when it did not."""
    words = settings.observability
    if not report.changed:
        get_logger("memory").info(words.unchanged, procedures=len(report.unchanged))
        return
    if reporter is None:
        return
    counts = {key: len(names) for key, names in report.model_dump().items()}
    reporter.emit(
        words.sync_stage,
        "completed",
        event_kind="memory",
        summary=words.synced.format(**counts),
        fields=report.model_dump(),
    )
