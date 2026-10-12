"""One layout for every step card: what was asked, the model's part, the decision, the evidence.

A step that used a model used to draw two cards — its own, and an LLM CALL card for the call —
and every step laid its card out its own way, so the same question ("why did it decide that?")
had a different answer in a different place on each. A step card is one card per step, its
sections always in the configured order (`observability.step_card.sections`) and an empty one
left out:

    Asked      what the step was given to decide
    Model      which model answered, its tokens and time, and the backup if one answered
    Reasoning  the model's reasoning, or the rationale its answer states
    Response   the model's answer, as a tree of its fields
    Decision   what the step concluded, and why
    Evidence   what it rested on, and where that came from

It is a narrative card (`event_kind="narrative"`), so the console draws it with no renderer
of its own. The full request and response are in `llm.log`; the card is what to read.
"""

import json
from typing import Any, Self

from pydantic import BaseModel

from agent_platform.config import get_settings
from agent_platform.interface.presentation.text_tree import value_lines
from agent_platform.llm.types import CompletionResult

# A narrative card line: plain text, or `{"text", "tone"}` (`tui.tone_styles`).
Line = str | dict[str, str]

ASKED = "asked"
MODEL = "model"
REASONING = "reasoning"
RESPONSE = "response"
DECISION = "decision"
EVIDENCE = "evidence"


class ModelCall(BaseModel):
    """What one model call contributed to a step."""

    role: str
    model: str
    tokens: int
    seconds: float
    reasoning: str = ""
    response: str = ""
    finish_reason: str = ""
    # The model that answered when it was not the role's own (a backup in its chain).
    backup: str = ""
    # How many calls this stands for (a repaired plan, a judge per criterion).
    calls: int = 1

    @classmethod
    def combined(cls, calls: list["ModelCall"]) -> "ModelCall | None":
        """Several calls of one step as one: tokens and time summed, the last one's answer."""
        if not calls:
            return None
        last = calls[-1]
        return last.model_copy(
            update={
                "tokens": sum(call.tokens for call in calls),
                "seconds": sum(call.seconds for call in calls),
                "calls": sum(call.calls for call in calls),
                "backup": next((call.backup for call in reversed(calls) if call.backup), ""),
            }
        )

    @classmethod
    def of(
        cls, role: str, result: CompletionResult, seconds: float, primary: str = ""
    ) -> "ModelCall":
        """A completed call, its reasoning taken from the provider or else from the answer."""
        answered = result.model or primary
        stated, response = _split_rationale(result.content)
        return cls(
            role=role,
            model=primary or answered,
            tokens=result.usage.total_tokens,
            seconds=seconds,
            reasoning=result.reasoning or stated,
            # A rationale shown as the reasoning is not shown again in the response.
            response=result.content if result.reasoning else response,
            finish_reason=result.finish_reason or "",
            backup=answered if primary and answered and answered != primary else "",
        )


class StepCard:
    """Builds a step card's sections, in the configured order (Builder)."""

    def __init__(self) -> None:
        self._sections: dict[str, list[Line]] = {}

    def asked(self, *lines: Line) -> Self:
        return self._add(ASKED, lines)

    def model(self, call: ModelCall | None, *, response: bool = True) -> Self:
        """The model's part of the step: who answered, its reasoning, its answer.

        `response=False` leaves the raw answer out (it is in `llm.log`), for an answer too long
        for a card; the step then writes its own summary with `response(...)`.
        """
        if call is None:
            return self
        words = get_settings().observability.step_card
        line = words.model_line.format(
            role=call.role.replace("_", " ").upper(),
            model=call.model,
            tokens=f"{call.tokens:,}",
            seconds=f"{call.seconds:.1f}",
        ) + (words.calls_suffix.format(calls=call.calls) if call.calls > 1 else "")
        backup = [words.backup_line.format(backup=call.backup)] if call.backup else []
        self._add(MODEL, (line, *backup))
        self._add(REASONING, tuple(value_lines(call.reasoning)) if call.reasoning else ())
        if response and call.response:
            self._add(RESPONSE, tuple(value_lines(call.response)))
        return self

    def model_notes(self, *lines: Line) -> Self:
        """More about the model's part: which models, and how independent of each other."""
        return self._add(MODEL, lines)

    def reasoning(self, *lines: Line) -> Self:
        return self._add(REASONING, lines)

    def response(self, *lines: Line) -> Self:
        return self._add(RESPONSE, lines)

    def decided_in_code(self, by: str, *reasoning: Line) -> Self:
        """No model was asked: say what decided instead, and on what grounds."""
        line = get_settings().observability.step_card.no_model_line.format(by=by)
        self._add(MODEL, (line,))
        return self._add(REASONING, reasoning)

    def decision(self, *lines: Line) -> Self:
        return self._add(DECISION, lines)

    def evidence(self, *lines: Line) -> Self:
        return self._add(EVIDENCE, lines)

    def fields(self) -> dict[str, list[Line]]:
        """The card's sections under their configured titles, in order, empty ones left out."""
        titles = get_settings().observability.step_card.sections
        return {
            title: self._sections[key] for key, title in titles.items() if self._sections.get(key)
        }

    def _add(self, key: str, lines: tuple[Line, ...]) -> Self:
        self._sections.setdefault(key, []).extend(line for line in lines if line)
        return self


def _split_rationale(content: str) -> tuple[str, str]:
    """The rationale a structured answer states for itself, and the answer without it."""
    try:
        parsed: Any = json.loads(content)
    except (TypeError, ValueError):
        return "", content
    if not isinstance(parsed, dict):
        return "", content
    fields = get_settings().observability.step_card.rationale_fields
    name = next((name for name in fields if parsed.get(name)), None)
    if name is None:
        return "", content
    rest = {key: value for key, value in parsed.items() if key != name}
    return str(parsed[name]), json.dumps(rest) if rest else ""


__all__ = ["ModelCall", "StepCard"]
