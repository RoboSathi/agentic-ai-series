"""What a finished research run leaves for the dream (ADR ra-0023).

Reflection inside a run fixes that run. What carries to the next run is
what its critics found — a criterion the audit failed, the reasons pages
were rejected, a limit the run hit — recorded here, in code, from the
run's own verdicts and never by a model. The run's dream event carries
them as signals, scoped to the run's kind (`archetype:<kind>`), and
`make dream ra` decides what they amount to: a summary of the run, facts
about sources, and — once a finding recurs across runs — a skill a
curator may approve for the next run of that kind.
"""

from collections import Counter

from pydantic import BaseModel, Field

from agent_platform.memory.dream import DreamEvent, DreamEventSink, DreamSignal
from agent_platform.verification.acceptance import CheckVerdict
from research_agent.config.settings import ResearchMemorySettings
from research_agent.domain.rubric import Contract

# Where a budget stop names the node it happened in: "<node>: <what ran out>".
_NODE_SEPARATOR = ": "


class RunOutcome(BaseModel):
    """What a finished run's critics found."""

    contract: Contract
    status: str
    failed: list[CheckVerdict] = Field(default_factory=list)
    rejected_codes: dict[str, str] = Field(default_factory=dict)
    budget_stops: list[str] = Field(default_factory=list)
    audit_scores: list[float] = Field(default_factory=list)
    revisions: int = 0


class RunEventWriter:
    """Writes one dream event per finished run."""

    def __init__(
        self, sink: DreamEventSink, agent_id: str, settings: ResearchMemorySettings
    ) -> None:
        self._sink = sink
        self._agent = agent_id
        self._settings = settings

    def record(self, outcome: RunOutcome) -> DreamEvent:
        """Emit the run's event and return it."""
        brief = outcome.contract.brief
        event = DreamEvent(
            agent_id=self._agent,
            session_id=outcome.contract.contract_id,
            kind=self._settings.run_event_kind,
            user_input=brief.topic,
            assistant_output=outcome.status,
            signals=self.signals(outcome),
            scope=[self.scope(brief.archetype)],
            outcome={
                "topic": brief.topic,
                "archetype": brief.archetype,
                "status": outcome.status,
                "audit_scores": outcome.audit_scores,
                "revisions": outcome.revisions,
                "failed": [_criterion(v, outcome) for v in outcome.failed],
                "rejected": outcome.rejected_codes,
                "budget_stops": outcome.budget_stops,
            },
        )
        self._sink.emit(event)
        return event

    def scope(self, archetype: str) -> str:
        """A run kind as the skill trigger a learned skill applies under."""
        return self._settings.scope.format(archetype=archetype)

    def signals(self, outcome: RunOutcome) -> list[DreamSignal]:
        """One signal per finding: each failed criterion, rejection code and limit hit."""
        kinds, scope = self._settings.signals, self.scope(outcome.contract.brief.archetype)
        found = [
            DreamSignal(kind=kinds.criterion_failed, value=_criterion(v, outcome), scope=scope)
            for v in outcome.failed
        ]
        found += [
            DreamSignal(kind=kinds.source_rejected, value=code, scope=scope)
            for code in Counter(outcome.rejected_codes.values())
        ]
        limits = dict.fromkeys(stop.split(_NODE_SEPARATOR)[-1] for stop in outcome.budget_stops)
        found += [DreamSignal(kind=kinds.budget_stop, value=limit, scope=scope) for limit in limits]
        return found


def _criterion(verdict: CheckVerdict, outcome: RunOutcome) -> str:
    item = outcome.contract.rubric.item(verdict.item_id)
    return item.text if item is not None else verdict.name
