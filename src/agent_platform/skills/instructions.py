"""The profile and one turn's procedures, as the system prompt carries them.

`AgentInstructions` is the whole interface an agent sees: the profile on
its own (for a screen that must not see per-turn procedures), or the
profile composed with whatever procedures this turn selected. Behind it,
procedures are synced at build time, selected without a model call, and
recorded — one card and one audit entry per turn, however many times a
tool loop composes the same prompt.
"""

from typing import Any, Protocol, runtime_checkable

from agent_platform.audit.ledger import AuditLedger
from agent_platform.audit.types import AuditEntry
from agent_platform.config.settings import SkillSettings
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
from agent_platform.skills.types import Activation, AgentProfile, Instructions, SkillSyncReport


@runtime_checkable
class AgentInstructions(Protocol):
    """What an agent's prompt builder asks of its profile and procedures."""

    @property
    def catalog(self) -> SkillCatalog | None: ...

    def profile_section(self) -> str: ...

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

    def compose(self, signals: frozenset[str], query: str) -> Instructions:
        """This turn's instructions: the profile, then each selected procedure."""
        if self._catalog is None:
            bare = Activation(profile=self._profile)
            return Instructions(text=self.profile_section(), activation=bare)
        chosen = self._selector.select(self._catalog.active(), signals, query)
        activation = Activation(profile=self._profile, skills=chosen)
        self._announce(activation, signals, query)
        return Instructions(text=self._render(activation), activation=activation)

    def _render(self, activation: Activation) -> str:
        render = self._settings.render
        sections = [self.profile_section()]
        for item in activation.skills:
            heading = render.procedure_heading.format(name=item.skill.name)
            sections.append(render.separator.join([heading, item.skill.document.body]))
        return render.separator.join(sections)

    def _announce(self, activation: Activation, signals: frozenset[str], query: str) -> None:
        """One card and one audit entry per turn — a tool loop composes many times."""
        key = (get_current_trace_id(), signals, query)
        if key == self._announced:
            return
        self._announced = key
        fields = self._fields(activation, signals)
        self._show(activation, fields)
        self._record(fields)

    def _fields(self, activation: Activation, signals: frozenset[str]) -> dict[str, Any]:
        render = self._settings.render
        refs = [item.skill.ref(render.ref) for item in activation.skills]
        return {
            "profile": render.profile_ref.format(digest=activation.profile.digest),
            "procedures": refs,
            "reasons": {item.skill.ref(render.ref): item.reason for item in activation.skills},
            "signals": sorted(signals),
        }

    def _show(self, activation: Activation, fields: dict[str, Any]) -> None:
        if self._reporter is None:
            return
        words = self._settings.observability
        refs = fields["procedures"]
        summary = (
            words.activated.format(count=len(activation.skills), refs=", ".join(refs))
            if activation.skills
            else words.none_active
        )
        self._reporter.emit(
            words.activation_stage,
            "completed",
            event_kind="memory",
            summary=summary,
            fields=fields,
            quiet=not activation.skills,
        )

    def _record(self, fields: dict[str, Any]) -> None:
        if self._audit is None:
            return
        audit = self._settings.audit
        self._audit.record(
            AuditEntry(
                trace_id=get_current_trace_id() or audit.activation_action,
                actor=audit.actor,
                action=audit.activation_action,
                authority_context=fields,
                decision="allow",
                rationale=", ".join(fields["procedures"]) or str(fields["profile"]),
            )
        )


def build_agent_instructions(
    settings: SkillSettings,
    *,
    store: ProceduralStore | None,
    audit: AuditLedger | None,
    reporter: MemoryReporter | None,
) -> InstructionBuilder:
    """The composition root's one call: profile always, procedures if memory has them.

    `store` is None when the agent's posture withholds procedural memory;
    the agent then runs on its profile alone, with nothing synced.
    """
    profile = load_profile(settings)
    if store is None:
        return InstructionBuilder(profile, settings)
    catalog = SkillCatalog(store, settings, audit=audit)
    report = catalog.sync(load_skill_documents(settings))
    _check_always(catalog, settings)
    _show_sync(report, settings, reporter)
    return InstructionBuilder(profile, settings, catalog=catalog, audit=audit, reporter=reporter)


def render_profile(settings: SkillSettings) -> str:
    """The profile as the prompt carries it, without building anything else."""
    return InstructionBuilder(load_profile(settings), settings).profile_section()


def _check_always(catalog: SkillCatalog, settings: SkillSettings) -> None:
    """More `always` skills than fit one prompt would silently drop one."""
    always = [s.name for s in catalog.active() if settings.always_trigger in s.document.triggers]
    if len(always) > settings.selection.max_active:
        reason = f"{len(always)} '{settings.always_trigger}' skills exceed selection.max_active"
        raise SkillDocumentError(settings.skills_dir, reason)


def _show_sync(
    report: SkillSyncReport, settings: SkillSettings, reporter: MemoryReporter | None
) -> None:
    if reporter is None:
        return
    words = settings.observability
    counts = {key: len(names) for key, names in report.model_dump().items()}
    reporter.emit(
        words.sync_stage,
        "completed",
        event_kind="memory",
        summary=words.synced.format(**counts),
        fields=report.model_dump(),
        quiet=True,
    )
