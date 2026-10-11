"""A dream: consolidating an agent's dream events into long-term memory, on demand.

The turn-time path writes what one turn can judge. A dream reads whole
sessions afterwards — Letta's sleep-time idea, run when an operator asks
(`make dream <agent>`, `/dream`) — and asks the agent's memory profile
what is worth keeping:

  * a **summary** per session (episodic) that supersedes its per-turn
    records, so recall returns one meaningful entry, not ten;
  * **facts** (semantic), reconciled against what is already believed or
    already waiting for a curator (mem0's ADD / UPDATE / NOOP);
  * **skill candidates** (procedural) from signals that recur across
    sessions, which governance sends to a curator rather than activating.

Every write goes through the governed pipeline via the facade, redacted
first. Processed events are moved, not deleted, and pruned only after the
retention window — so a dream is idempotent and auditable.
"""

from datetime import timedelta
from typing import TYPE_CHECKING

from agent_platform.config.settings import DreamSettings
from agent_platform.logging import get_logger, instrument
from agent_platform.memory.dream.consolidators import FactCandidate, SkillCandidate
from agent_platform.memory.dream.events import DreamEvent
from agent_platform.memory.dream.inbox import FileDreamInbox
from agent_platform.memory.dream.report import DreamEntry, DreamReport, ReconcileOp, plural
from agent_platform.memory.dream.summaries import SUMMARY_CATEGORY
from agent_platform.memory.facade import MemoryFacade
from agent_platform.memory.types import MemoryCommand, MemoryPrincipal, WriteResult
from agent_platform.observability.types import utc_now

if TYPE_CHECKING:
    # Typing only: the profile's defaults are built from this package.
    from agent_platform.memory.profile import MemoryProfile

_SUMMARY_KEY = "session_summary"


@instrument("memory")
class Dream:
    """Runs dreams over one agent's inbox, with that agent's memory profile."""

    def __init__(
        self,
        inbox: FileDreamInbox,
        memory: MemoryFacade,
        profile: "MemoryProfile",
        settings: DreamSettings,
    ) -> None:
        self._inbox = inbox
        self._memory = memory
        self._profile = profile
        self._settings = settings
        self._logger = get_logger("memory")

    @property
    def inbox(self) -> FileDreamInbox:
        return self._inbox

    async def dream(self) -> DreamReport:
        """Consolidate everything pending; returns what changed."""
        started = utc_now()
        sessions = self._inbox.unconsolidated()
        report = DreamReport(
            started_at=started,
            events=sum(len(events) for events in sessions.values()),
            sessions=len(sessions),
        )
        for session, events in sessions.items():
            await self._consolidate_session(session, events, report)
            self._inbox.mark_consolidated([event.event_id for event in events], started)
        self._propose_skills(sessions, report)
        report.pruned = self._inbox.prune(started - timedelta(days=self._settings.retention_days))
        if self._memory.episodic is not None:
            self._memory.episodic.write_index(SUMMARY_CATEGORY)
        return report

    async def _consolidate_session(
        self, session: str, events: list[DreamEvent], report: DreamReport
    ) -> None:
        subject = events[0].subject_id or self._subject_from_log(session)
        wrote = self._facts(session, subject, events, report)
        if self._memory.episodic is None:
            return
        summary = await self._profile.episodic.summarize(events)
        if summary is None:
            if not wrote:
                report.skipped.append(session)
            return
        if summary.intent_kept is False:
            report.notes.append(
                f"session {session}: generated intent sentence failed the NLI check; dropped"
            )
        self._summary(session, subject, summary.text, report)

    def _summary(self, session: str, subject: str, text: str, report: DreamReport) -> None:
        """Write the summary, then retire the per-turn records (and older summaries) it replaces."""
        episodic = self._memory.episodic
        assert episodic is not None
        replaced = [fact.key for fact in episodic.events(session)]
        version = self._summary_count(session) + 1
        # A summary is what happened, rebuilt from recorded results — an
        # event, not a claim — so it takes the event path: authorized,
        # redacted, but not held back by confidence routing.
        result = self._memory.record_event(
            text,
            source=self._settings.actor,
            session_id=session,
            principal=self._principal(subject),
            category=SUMMARY_CATEGORY,
            key=f"{_SUMMARY_KEY}_{version}",
        )
        op = _op(result)
        report.add(
            DreamEntry(
                store="Episodic", op=op, label=f'summary "{text}"', subject=subject, session=session
            )
        )
        if op == "ADD" and result.fact is not None and replaced:
            episodic.supersede(session, replaced, result.fact.key, result.fact.provenance)
            report.add(
                DreamEntry(
                    store="Episodic",
                    op="UPDATE",
                    label=f"superseded {plural(len(replaced), 'earlier record')}",
                    subject=subject,
                    session=session,
                )
            )

    def _facts(
        self, session: str, subject: str, events: list[DreamEvent], report: DreamReport
    ) -> bool:
        """The profile's facts for this session, reconciled against current belief."""
        if self._memory.semantic is None:
            return False
        wrote = False
        waiting = self._pending_facts()
        for candidate in self._profile.semantic.facts(session, subject, events):
            why = self._already(candidate, waiting)
            if why:
                label = f"{candidate.key} = {candidate.value} ({why})"
                report.add(_fact_entry("NOOP", label, candidate, session))
                continue
            result = self._memory.submit(self._fact_command(candidate, session))
            op = _op(result)
            wrote = wrote or op in ("ADD", "UPDATE", "QUEUED")
            label = _fact_label(result, candidate.key, candidate.value)
            report.add(_fact_entry(op, label, candidate, session))
        return wrote

    def _already(self, candidate: FactCandidate, waiting: set[tuple[str, str, str]]) -> str:
        """Why a candidate needs no write ("" when it does)."""
        semantic = self._memory.semantic
        current = semantic.latest(candidate.key, candidate.subject_id) if semantic else None
        if current is not None and current.value == candidate.value:
            return "already known"
        if (candidate.key, candidate.value, candidate.subject_id) in waiting:
            return "already awaiting a curator"
        return ""

    def _fact_command(self, candidate: FactCandidate, session: str) -> MemoryCommand:
        subject = candidate.subject_id
        principal = MemoryPrincipal(
            actor=f"{candidate.role}-{subject}" if subject else self._settings.actor,
            role=candidate.role,
            subject_id=subject,
        )
        if candidate.text:
            # A person's own words: the pattern tier extracts the claim, as at turn time.
            return MemoryCommand(
                principal=principal,
                operation="create",
                text=candidate.text,
                session_id=session,
                require_pattern=True,
            )
        return MemoryCommand(
            principal=principal,
            operation="create",
            key=candidate.key,
            value=candidate.value,
            category=candidate.category,
            reason=candidate.reason,
            session_id=session,
        )

    def _pending_facts(self) -> set[tuple[str, str, str]]:
        """(key, value, subject) of every write already waiting on a curator."""
        return {
            (result.fact.key, result.fact.value, result.fact.subject_id)
            for _, result in self._memory.pipeline.pending_with_ids()
            if result.fact is not None
        }

    def _propose_skills(self, sessions: dict[str, list[DreamEvent]], report: DreamReport) -> None:
        """Recurring signals become skill candidates, handed to governance."""
        pending = {
            result.fact.key for _, result in self._memory.pipeline.pending_with_ids() if result.fact
        }
        active = self._memory.procedural
        for candidate in self._profile.skills.candidates(sessions):
            if candidate.name in pending:
                label = f"{candidate.name} (already awaiting a curator)"
                report.add(DreamEntry(store="Procedural", op="NOOP", label=label))
                continue
            if active is not None and active.get(candidate.name) is not None:
                label = f"{candidate.name} (already a skill)"
                report.add(DreamEntry(store="Procedural", op="NOOP", label=label))
                continue
            result = self._memory.submit(self._skill_command(candidate))
            label = f"{candidate.name} — {candidate.signal.kind}: {candidate.signal.value}"
            report.add(DreamEntry(store="Procedural", op=_op(result), label=label))

    def _skill_command(self, candidate: SkillCandidate) -> MemoryCommand:
        return MemoryCommand(
            principal=self._principal(""),
            operation="create",
            memory_type="procedural",
            key=candidate.name,
            value=candidate.document,
            category=self._settings.learned_skill.artifact_type,
            reason=f"{candidate.signal.kind} recurred in {candidate.occurrences} sessions",
        )

    def _principal(self, subject: str) -> MemoryPrincipal:
        return MemoryPrincipal(actor=self._settings.actor, role="agent", subject_id=subject)

    def _subject_from_log(self, session: str) -> str:
        """For events recorded before turn events carried a subject."""
        if self._memory.episodic is None:
            return ""
        events = self._memory.episodic.events(session, include_superseded=True)
        return next((fact.subject_id for fact in events if fact.subject_id), "")

    def _summary_count(self, session: str) -> int:
        episodic = self._memory.episodic
        if episodic is None:
            return 0
        return sum(
            1
            for fact in episodic.events(session, include_superseded=True)
            if fact.category == SUMMARY_CATEGORY
        )


def _op(result: WriteResult) -> ReconcileOp:
    """The reconcile op a pipeline outcome amounts to."""
    if result.written:
        return "UPDATE" if result.fact is not None and result.fact.version > 1 else "ADD"
    if result.route == "refused" and result.curator_id is None:
        return "REFUSED"
    if result.curator_id is not None or result.route in ("curator_queue", "refused"):
        return "QUEUED"
    return "NOOP"


def _fact_label(result: WriteResult, key: str, value: str) -> str:
    base = f"{key} = {value}"
    return base if result.written else f"{base} ({result.reason})"


def _fact_entry(op: ReconcileOp, label: str, candidate: FactCandidate, session: str) -> DreamEntry:
    return DreamEntry(
        store="Semantic", op=op, label=label, subject=candidate.subject_id, session=session
    )
