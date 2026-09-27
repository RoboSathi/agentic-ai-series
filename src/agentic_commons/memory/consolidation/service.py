"""A dream: consolidating the dream inbox into long-term memory, on demand.

The turn-time path writes what it can judge from one message. A dream
reads whole sessions afterwards — Letta's sleep-time idea, run when an
operator asks (`make dream`, `/dream`) — and decides what is worth
keeping:

  * a **session summary** (episodic) that supersedes the session's
    per-turn events, so recall returns one meaningful record, not ten
    "query: …" lines;
  * **customer facts** (semantic) the pattern tier recognises, reconciled
    against what is already believed (mem0's ADD / UPDATE / NOOP);
  * **procedure candidates** from refusals that recur across sessions,
    which governance sends to a curator rather than activating.

Every write goes through the governed pipeline via the facade, redacted
first. Processed events are marked, not deleted, and pruned only after
the retention window — so a dream is idempotent and auditable.
"""

from datetime import timedelta

from agentic_commons.config.settings import DreamSettings
from agentic_commons.logging import get_logger, instrument
from agentic_commons.memory.consolidation.procedures import ProcedureCandidate, recurring_refusals
from agentic_commons.memory.consolidation.report import (
    DreamEntry,
    DreamReport,
    ReconcileOp,
    plural,
)
from agentic_commons.memory.consolidation.summaries import SessionSummarizer
from agentic_commons.memory.deterministic import DeterministicClaims
from agentic_commons.memory.episodic.memory import EpisodicStore
from agentic_commons.memory.events import DreamInbox, MemoryEvent
from agentic_commons.memory.facade import MemoryFacade
from agentic_commons.memory.inspect import SUMMARY_CATEGORY
from agentic_commons.memory.types import MemoryCommand, MemoryPrincipal, WriteResult
from agentic_commons.observability.types import utc_now

_PROCEDURE_CATEGORY = "how_to"
_SUMMARY_KEY = "session_summary"


@instrument("memory")
class ConsolidationService:
    """Runs dreams over one agent's inbox."""

    def __init__(
        self,
        inbox: DreamInbox,
        memory: MemoryFacade,
        episodic: EpisodicStore,
        summarizer: SessionSummarizer,
        settings: DreamSettings,
        claims: DeterministicClaims | None = None,
    ) -> None:
        self._inbox = inbox
        self._memory = memory
        self._episodic = episodic
        self._summarizer = summarizer
        self._settings = settings
        self._claims = claims or DeterministicClaims()
        self._logger = get_logger("memory")

    async def dream(self) -> DreamReport:
        """Consolidate everything unconsolidated; returns what changed."""
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
        self._propose_procedures(sessions, report)
        report.pruned = self._inbox.prune(started - timedelta(days=self._settings.retention_days))
        return report

    async def _consolidate_session(
        self, session: str, events: list[MemoryEvent], report: DreamReport
    ) -> None:
        subject = events[0].subject_id or self._subject_from_log(session)
        wrote = self._facts(session, subject, events, report)
        summary = await self._summarizer.summarize(events)
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
        """Write the summary, then retire the per-turn events (and older summaries) it replaces."""
        replaced = [fact.key for fact in self._episodic.events(session)]
        version = self._summary_count(session) + 1
        # A summary is what happened, rebuilt from recorded tool results —
        # an event, not a claim — so it takes the event path: authorized,
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
            self._episodic.supersede(session, replaced, result.fact.key, result.fact.provenance)
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
        self, session: str, subject: str, events: list[MemoryEvent], report: DreamReport
    ) -> bool:
        """Customer facts the pattern tier recognises, reconciled against current belief."""
        wrote = False
        semantic = self._memory.semantic
        waiting = self._pending_facts()
        for event in events:
            claim = self._claims.extract(event.user_input)
            if claim is None or not subject:
                continue
            current = semantic.latest(claim.key, subject) if semantic is not None else None
            known = current is not None and current.value == claim.value
            if known or (claim.key, claim.value, subject) in waiting:
                why = "already known" if known else "already awaiting a curator"
                report.add(
                    DreamEntry(
                        store="Semantic",
                        op="NOOP",
                        label=f"{claim.key} = {claim.value} ({why})",
                        subject=subject,
                        session=session,
                    )
                )
                continue
            result = self._memory.submit(
                MemoryCommand(
                    principal=MemoryPrincipal(
                        actor=f"customer-{subject}", role="customer", subject_id=subject
                    ),
                    operation="create",
                    text=event.user_input,
                    session_id=session,
                    require_pattern=True,
                )
            )
            op = _op(result)
            wrote = wrote or op in ("ADD", "UPDATE", "QUEUED")
            report.add(
                DreamEntry(
                    store="Semantic",
                    op=op,
                    label=_fact_label(result, claim.key, claim.value),
                    subject=subject,
                    session=session,
                )
            )
        return wrote

    def _pending_facts(self) -> set[tuple[str, str, str]]:
        """(key, value, subject) of every write already waiting on a curator."""
        return {
            (result.fact.key, result.fact.value, result.fact.subject_id)
            for _, result in self._memory.pipeline.pending_with_ids()
            if result.fact is not None
        }

    def _propose_procedures(
        self, sessions: dict[str, list[MemoryEvent]], report: DreamReport
    ) -> None:
        """Recurring refusals become procedure candidates, handed to governance."""
        pending = {
            result.fact.key for _, result in self._memory.pipeline.pending_with_ids() if result.fact
        }
        for candidate in recurring_refusals(
            sessions,
            self._settings.refusal_markers,
            self._settings.procedure_threshold,
            self._settings.procedure_key_chars,
        ):
            if candidate.key in pending:
                report.add(
                    DreamEntry(
                        store="Procedural",
                        op="NOOP",
                        label=f"{candidate.key} (already awaiting a curator)",
                    )
                )
                continue
            result = self._memory.submit(self._procedure_command(candidate))
            report.add(
                DreamEntry(
                    store="Procedural", op=_op(result), label=f'candidate "{candidate.value}"'
                )
            )

    def _procedure_command(self, candidate: ProcedureCandidate) -> MemoryCommand:
        return MemoryCommand(
            principal=self._principal(""),
            operation="create",
            memory_type="procedural",
            key=candidate.key,
            value=candidate.value,
            category=_PROCEDURE_CATEGORY,
            reason=f"refused in {candidate.sessions} sessions",
        )

    def _principal(self, subject: str) -> MemoryPrincipal:
        return MemoryPrincipal(actor=self._settings.actor, role="agent", subject_id=subject)

    def _subject_from_log(self, session: str) -> str:
        """For events recorded before turn events carried a subject."""
        events = self._episodic.events(session, include_superseded=True)
        return next((fact.subject_id for fact in events if fact.subject_id), "")

    def _summary_count(self, session: str) -> int:
        return sum(
            1
            for fact in self._episodic.events(session, include_superseded=True)
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
