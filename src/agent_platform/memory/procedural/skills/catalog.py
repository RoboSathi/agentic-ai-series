"""Authored skills as versions in procedural memory (ADR 0010).

The SKILL.md in git is the source of truth; procedural memory is what
runs, and what keeps every version. Sync is a deployment step, not a
principal's claim, so it writes to the store directly rather than through
the governed write pipeline — and records each change on the audit trail.
It compares against the *latest* version, not the active one: an
operator who rolled a skill back keeps that rollback across a restart
until the file itself changes again.
"""

from collections.abc import Sequence

from agent_platform.audit.ledger import AuditLedger
from agent_platform.audit.types import AuditEntry
from agent_platform.config.settings import SkillSettings
from agent_platform.logging import get_logger
from agent_platform.logging.trace_context import get_current_trace_id
from agent_platform.memory.procedural.skills.document import parse_stored
from agent_platform.memory.procedural.skills.types import (
    AUTHORED,
    LEARNED,
    Skill,
    SkillDocument,
    SkillSyncReport,
)
from agent_platform.memory.procedural.store import SkillStore
from agent_platform.memory.types import MemoryFact, Provenance


class SkillCatalog:
    """The agent's skills: authored skills and approved lessons."""

    def __init__(
        self,
        store: SkillStore,
        settings: SkillSettings,
        *,
        audit: AuditLedger | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._audit = audit
        self._parsed: dict[tuple[str, int], Skill] = {}
        self._logger = get_logger("memory")

    def sync(self, documents: Sequence[SkillDocument]) -> SkillSyncReport:
        """Bring procedural memory in line with the authored files."""
        report = SkillSyncReport()
        for document in documents:
            getattr(report, self._sync_one(document)).append(document.name)
        report.retired.extend(self._retire_missing({d.name for d in documents}))
        self._logger.info("Synced skills", **report.model_dump())
        return report

    def active(self) -> list[Skill]:
        """Every selectable skill's active version, by name."""
        found = (self.show(name) for name in self._store.names())
        return [skill for skill in found if skill is not None]

    def show(self, name: str) -> Skill | None:
        """A skill's active version, if it has one and may be selected."""
        fact = self._store.get(name)
        if fact is None or fact.category not in self._settings.selectable_artifact_types:
            return None
        return self._skill(fact)

    def history(self, name: str) -> list[Skill]:
        """Every version of a skill, newest first."""
        return [self._skill(fact) for fact in self._store.history(name)]

    def _sync_one(self, document: SkillDocument) -> str:
        """Which report list a document belongs on, writing if it changed."""
        latest = self._store.history(document.name)[:1]
        if not latest:
            self._write(document, "added")
            return "added"
        if latest[0].value != document.raw:
            self._write(document, "updated")
            return "updated"
        if self._store.get(document.name) is None:
            self._restore(document.name, _version(latest[0]))
            return "added"
        return "unchanged"

    def _write(self, document: SkillDocument, change: str) -> None:
        kind = self._settings.authored_artifact_type
        fact = MemoryFact(
            key=document.name,
            value=document.raw,
            memory_type="procedural",
            category=kind,
            metadata={"artifact_type": kind, "rationale": document.description},
            provenance=self._provenance(document.name),
        )
        written = self._store.write(fact, fact.provenance)
        self._record(document.name, _version(written.fact) if written.fact else 0, change)

    def _restore(self, name: str, version: int) -> None:
        self._store.rollback(name, version)
        self._record(name, version, "added")

    def _retire_missing(self, present: set[str]) -> list[str]:
        """Retire authored skills whose files are gone; lessons are left alone."""
        retired: list[str] = []
        for name in self._store.names():
            fact = self._store.get(name)
            if name in present or fact is None:
                continue
            if fact.category == self._settings.authored_artifact_type:
                self._store.retire(name)
                self._record(name, _version(fact), "retired")
                retired.append(name)
        return retired

    def _provenance(self, name: str) -> Provenance:
        source = "/".join([self._settings.skills_dir, name, self._settings.skill_file])
        return Provenance(source=source, kind="verified_system_of_record")

    def _skill(self, fact: MemoryFact) -> Skill:
        key = (fact.key, _version(fact))
        if key not in self._parsed:
            self._parsed[key] = Skill(
                document=parse_stored(fact.value, fact.key, self._settings),
                version=key[1],
                source=AUTHORED
                if fact.category == self._settings.authored_artifact_type
                else LEARNED,
            )
        return self._parsed[key]

    def _record(self, name: str, version: int, change: str) -> None:
        if self._audit is None:
            return
        audit = self._settings.audit
        ref = self._settings.render.ref.format(name=name, version=version)
        self._audit.record(
            AuditEntry(
                trace_id=get_current_trace_id() or audit.sync_action,
                actor=audit.actor,
                action=audit.sync_action,
                authority_context={"skill": ref, "change": change},
                decision="allow",
                rationale=f"{ref} {change}",
            )
        )


def _version(fact: MemoryFact) -> int:
    return int(fact.metadata.get("version", 0))
