"""Promptfoo grader provider: `llm-rubric` assertions judged by the platform judge.

An agent's `promptfooconfig.yaml` names a one-line `grader.py` that re-exports
`call_api` from here, under `defaultTest.options.provider` — so model-graded
assertions use the platform judge (`judge:`), its grading rules and the
shared rate-limit budget, instead of Promptfoo's default (another vendor's
account). The verdict is validated against Promptfoo's grading shape; a
judge that cannot answer abstains, which Promptfoo records as an error,
never as a pass.
"""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_platform.judge import JudgeCall, ModelJudge
from agent_platform.llm.types import Message

Role = Literal["system", "user", "assistant"]
_ROLES: dict[str, Role] = {"system": "system", "user": "user", "assistant": "assistant"}


class RubricGrade(BaseModel):
    """What Promptfoo expects a grader to return for `llm-rubric`."""

    reason: str
    passed: bool = Field(alias="pass")
    score: float

    model_config = ConfigDict(populate_by_name=True)


class GradeOutcome(BaseModel):
    """The grade, or why there is none."""

    grade: RubricGrade | None = None
    call: JudgeCall = Field(default_factory=JudgeCall)
    error: str = ""


class PromptfooRubricJudge(ModelJudge[str, RubricGrade, GradeOutcome]):
    """Promptfoo's rendered rubric prompt in, its grading JSON out."""

    def __init__(self) -> None:
        super().__init__(RubricGrade, subsystem="evals")

    def render(self, request: str) -> list[Message]:
        """Promptfoo sends a rubric as a JSON chat array, or as plain text."""
        try:
            chat = json.loads(request)
        except ValueError:
            chat = None
        if not isinstance(chat, list):
            return [Message(role="user", content=request)]
        return [
            Message(role=_ROLES[item["role"]], content=str(item.get("content", "")))
            for item in chat
            if isinstance(item, dict) and item.get("role") in _ROLES
        ]

    def verdict(self, request: str, answer: RubricGrade, call: JudgeCall) -> GradeOutcome:
        return GradeOutcome(grade=answer, call=call)

    def abstain(self, request: str, reason: str) -> GradeOutcome:
        return GradeOutcome(error=reason)


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Promptfoo's provider entry point."""
    import asyncio

    outcome = asyncio.run(PromptfooRubricJudge().judge(prompt))
    if outcome.grade is None:
        return {"error": outcome.error}
    return {
        "output": outcome.grade.model_dump_json(by_alias=True),
        "tokenUsage": {"numRequests": outcome.call.calls},
        "cost": outcome.call.cost_usd,
    }
