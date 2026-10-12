"""Reflection on the SQL: does what the model ran answer what was asked?

The server's walls decide what SQL may run (`QueryGuard`), and DuckDB says when a name or
a type is wrong; both come back to the model as errors it fixes. Neither can say whether
a query that ran answers the question: the wrong view, a filter the question did not ask
for, a count where a distinct count was meant, a rate whose denominator is off. The
figures check cannot either — a figure faithfully copied from the wrong query is
"supported". So once the model answers, one more call reads the question, the SQL, a
few rows and the draft, and says accept or revise (Reflexion's critic). A revise goes
back to the model as a reviewer's finding, a bounded number of times a turn; the loop
governor's budgets still hold every step.

The critic is a quality check, not a safety control: a call that fails, times out or
answers out of form accepts, and the card says the answer went on unreviewed.
"""

import asyncio
import json
import time
from typing import Literal

from pydantic import BaseModel, Field

from agent_platform.interface.presentation.text_tree import as_structure
from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import CompletionRequest, Message
from agent_platform.logging import describe_error, get_logger
from agent_platform.observability.step_card import ModelCall
from ops_agent.config.settings import SqlReviewSettings
from ops_agent.domain import QueryRecord

Verdict = Literal["accept", "revise"]
_VERDICT, _ISSUES, _HINT, _RATIONALE = "verdict", "issues", "hint", "rationale"


class SqlReview(BaseModel):
    """What the critic concluded about the turn's SQL, and the call that concluded it."""

    verdict: Verdict
    issues: list[str] = Field(default_factory=list)
    hint: str = ""
    rationale: str = ""
    call: ModelCall | None = None
    # What the call cost, for the turn's budget (the loop governor counts every model call).
    cost_usd: float = 0.0
    # Why the critic could not judge (it then accepts); "" when it judged.
    failed: str = ""

    @property
    def revise(self) -> bool:
        return self.verdict == "revise"


class SqlReviewer:
    """Asks one model call whether the turn's SQL answers the question (Reflexion's critic)."""

    def __init__(self, llm: LLMClient, settings: SqlReviewSettings) -> None:
        self._llm = llm
        self._settings = settings
        self._logger = get_logger("ops_agent")

    def reviewed(self, queries: list[QueryRecord]) -> list[QueryRecord]:
        """The queries whose SQL the model wrote (the only ones worth reviewing)."""
        return [query for query in queries if query.tool in self._settings.reviewed_tools]

    async def review(self, question: str, queries: list[QueryRecord], draft: str) -> SqlReview:
        """Accept or revise, with the issues found; accept when the critic cannot judge."""
        request = self._request(question, queries, draft)
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self._settings.timeout_seconds):
                result = await self._llm.complete(request)
        except Exception as exc:  # noqa: BLE001 - a quality check fails open, and says so
            error = describe_error(exc)
            self._logger.warning("SQL review failed", error=error)
            return SqlReview(verdict="accept", failed=error)
        call = ModelCall.of(self._settings.role, result, time.perf_counter() - started)
        review = self._parsed(result.content, call).model_copy(update={"cost_usd": result.cost_usd})
        self._logger.info("SQL reviewed", **review.model_dump(exclude={"call"}))
        return review

    def revise_message(self, review: SqlReview) -> Message:
        """The finding, as the model reads it on its way back to fix the query."""
        text = self._settings.revise_message.format(
            issues="; ".join(review.issues) or review.rationale, hint=review.hint
        )
        return Message(role="user", content=text)

    def _request(self, question: str, queries: list[QueryRecord], draft: str) -> CompletionRequest:
        settings = self._settings
        blocks = "\n\n".join(self._block(number, query) for number, query in enumerate(queries, 1))
        content = settings.request.format(question=question, draft=draft, queries=blocks)
        return CompletionRequest(
            messages=[
                Message(role="system", content=settings.instructions),
                Message(role="user", content=content),
            ],
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    def _block(self, number: int, query: QueryRecord) -> str:
        settings = self._settings
        return settings.query_block.format(
            number=number,
            rows=query.row_count,
            truncated=settings.truncated_mark if query.truncated else "",
            sql=query.sql.strip(),
            columns=", ".join(query.columns),
            sample=json.dumps(query.rows[: settings.sample_rows], default=str),
        )

    def _parsed(self, content: str, call: ModelCall) -> SqlReview:
        """The critic's JSON; out of form, an accept that says it could not judge."""
        answer = as_structure(content)
        verdict = answer.get(_VERDICT) if isinstance(answer, dict) else None
        if verdict not in (self._settings.accept, self._settings.revise):
            return SqlReview(verdict="accept", call=call, failed=self._settings.card.unparsed)
        issues = answer.get(_ISSUES) or []
        return SqlReview(
            verdict="revise" if verdict == self._settings.revise else "accept",
            issues=[str(issue) for issue in issues] if isinstance(issues, list) else [str(issues)],
            hint=str(answer.get(_HINT) or ""),
            rationale=str(answer.get(_RATIONALE) or ""),
            call=call,
        )


__all__ = ["SqlReview", "SqlReviewer"]
