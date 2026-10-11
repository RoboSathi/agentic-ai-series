"""How the research agent's dream reads a run (ADR ra-0023).

Two of the three questions a dream asks are answered here, for run
events; everything else falls through to the platform's defaults:

  * **Episodic** — a run is summarised from its own record (kind, topic,
    status, the audit's scores, what failed), never by a model.
  * **Semantic** — the sources a run could not use become facts about
    those sources ("pages on example.com were rejected as paywalled"),
    recorded as the system's observation rather than an inference: the
    fetcher's rejection code is what happened, not an opinion.

Skills come from the default recurring-signal learner: a criterion that
fails in enough runs of one kind becomes a skill for that kind.
"""

from collections import defaultdict
from urllib.parse import urlparse

from agent_platform.memory.dream import (
    DreamEvent,
    EpisodicConsolidator,
    FactCandidate,
    SessionSummary,
)
from research_agent.config.settings import ResearchMemorySettings


class RunSummary:
    """A run's summary from its recorded outcome; other events go to the default."""

    def __init__(self, settings: ResearchMemorySettings, fallback: EpisodicConsolidator) -> None:
        self._settings = settings
        self._fallback = fallback

    async def summarize(self, events: list[DreamEvent]) -> SessionSummary | None:
        runs = [event for event in events if event.kind == self._settings.run_event_kind]
        if not runs:
            return await self._fallback.summarize(events)
        lines = [self._line(event) for event in runs]
        return SessionSummary(text=" ".join(lines), outcomes=lines)

    def _line(self, event: DreamEvent) -> str:
        outcome, words = event.outcome, self._settings
        scores = [float(score) for score in outcome.get("audit_scores", [])]
        trajectory = " → ".join(f"{score:.2f}" for score in scores) or words.summary_none
        failed = "; ".join(str(item) for item in outcome.get("failed", [])) or words.summary_none
        return words.summary.format(
            kind=str(outcome.get("archetype", "")).replace("_", " "),
            topic=outcome.get("topic", event.user_input),
            status=outcome.get("status", event.assistant_output),
            scores=trajectory,
            revisions=outcome.get("revisions", 0),
            failed=failed,
        )


class SourceReliability:
    """Facts about the sources runs could not use, by domain."""

    def __init__(self, settings: ResearchMemorySettings) -> None:
        self._settings = settings

    def facts(self, session: str, subject: str, events: list[DreamEvent]) -> list[FactCandidate]:
        codes: dict[str, set[str]] = defaultdict(set)
        for event in events:
            if event.kind != self._settings.run_event_kind:
                continue
            for url, code in dict(event.outcome.get("rejected", {})).items():
                domain = urlparse(str(url)).netloc
                if domain and code:
                    codes[domain].add(str(code))
        words = self._settings
        return [
            FactCandidate(
                role="system_of_record",
                key=words.source_fact_key.format(domain=domain),
                value=words.source_fact_value.format(codes=", ".join(sorted(found))),
                category=words.source_fact_category,
                reason=session,
            )
            for domain, found in sorted(codes.items())
        ]
