"""Earlier research as a context source: pointers to archived runs that bear on a question.

A run whose topic shares enough words with the question is offered ("open it with /show"),
never as fact — a pointer is not evidence, so its chunk is `memory`, never `retrieved`, and
nothing it says can ground a claim.
"""

import re

from agentic_commons.context.types import Chunk, Task
from agentic_commons.logging import instrument
from research_agent.archive.store import ResearchArchive, RunIndexRow
from research_agent.config.settings import AnswerContextSettings, AnswerMessages

_WORD = re.compile(r"\w+")
SOURCE = "research_archive"


@instrument("research_agent")
class EarlierRunsSource:
    """`CandidateSource` over the archive's run index."""

    def __init__(
        self, archive: ResearchArchive, settings: AnswerContextSettings, messages: AnswerMessages
    ) -> None:
        self._archive = archive
        self._settings = settings
        self._messages = messages

    async def candidates(self, task: Task) -> list[Chunk]:
        block = self.block(self.matching(task.query))
        return [Chunk(content=block, kind="memory", source=SOURCE)] if block else []

    def matching(self, question: str) -> list[RunIndexRow]:
        """Archived runs whose topic shares enough words with the question, best match first."""
        if not self._settings.recall_limit:
            return []
        wanted = self._words(question)
        scored = [
            (len(wanted & self._words(f"{row.topic} {row.archetype}")), row)
            for row in self._archive.history(self._settings.recall_scan)
        ]
        ranked = sorted(
            (item for item in scored if item[0] >= self._settings.min_overlap),
            key=lambda item: -item[0],
        )
        return [row for _, row in ranked[: self._settings.recall_limit]]

    def block(self, runs: list[RunIndexRow]) -> str:
        """The runs as the prompt shows them: a heading, then one pointer a line."""
        if not runs:
            return ""
        notes = self._messages
        lines = [
            notes.earlier_line.format(run_id=r.run_id, status=r.status, topic=r.topic) for r in runs
        ]
        return "\n".join([notes.earlier_heading, *lines])

    def _words(self, text: str) -> set[str]:
        floor = self._settings.min_word_chars
        return {word for word in _WORD.findall(text.casefold()) if len(word) >= floor}


__all__ = ["EarlierRunsSource"]
