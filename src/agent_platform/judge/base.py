"""The LLM judge's skeleton: every judge asks, validates, and abstains the same way.

A judge is a model asked to decide something about someone else's output.
What it is asked differs per judge; how it is asked must not, or each judge
grows its own JSON parsing, its own error handling and its own notion of
"could not decide". So the skeleton lives here (Template Method):

    judge(request)
      └── _answer(request)  → (validated answer, what the call cost)   hook
      └── verdict(request, answer, call)                               hook
      └── on any failure:  abstain(request, reason)                    hook

`LlmJudge` fixes that order and the failure rule: a judge that could not
answer — every model in its chain down, or an answer that failed its schema
after the guardrails' bounded repair — *abstains*. It never returns a
silent zero, because a zero is a verdict and an outage is not.

`ModelJudge` is the common case: rendered messages, the platform judge
model (`judge:`), the shared grading rules prepended. An agent whose judge
must run through its own model router (say, for its own budget accounting)
subclasses `LlmJudge` and supplies `_answer` itself.
"""

from abc import ABC, abstractmethod
from functools import cache
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel

from agent_platform.config import get_settings
from agent_platform.config.settings import JudgeSectionSettings
from agent_platform.guardrails.guards import PydanticValidator
from agent_platform.judge.client import create_judge_client
from agent_platform.llm.client import LLMClient
from agent_platform.llm.recording import RecordingLLMClient
from agent_platform.llm.types import CompletionRequest, Message
from agent_platform.logging import describe_error, get_logger

RequestT = TypeVar("RequestT")
AnswerT = TypeVar("AnswerT", bound=BaseModel)
VerdictT = TypeVar("VerdictT")

_RULES = Path(__file__).resolve().parent / "prompts" / "judge_rules.md"


class JudgeAnswerError(ValueError):
    """The judge answered, but not in its schema, even after repair."""


class JudgeCall(BaseModel):
    """What one verdict cost, and which model gave it (a backup counts as itself)."""

    model: str = ""
    calls: int = 0
    cost_usd: float = 0.0


class LlmJudge(ABC, Generic[RequestT, AnswerT, VerdictT]):
    """Ask, then decide; abstain on any failure. Subclasses say what is asked and meant."""

    def __init__(self, subsystem: str = "judge") -> None:
        self._logger = get_logger(subsystem)
        # The models that answered since `models_used` was last read.
        self._used: set[str] = set()

    async def judge(self, request: RequestT) -> VerdictT:
        """The verdict — or, when no answer could be had, an abstention."""
        try:
            answer, call = await self._answer(request)
        except Exception as exc:
            self._logger.warning("Judge abstained", error=describe_error(exc))
            return self.abstain(request, describe_error(exc))
        if call.model:
            self._used.add(call.model)
        return self.verdict(request, answer, call)

    def models_used(self) -> set[str]:
        """The models that judged since the last read, and forget them."""
        used, self._used = self._used, set()
        return used

    @abstractmethod
    async def _answer(self, request: RequestT) -> tuple[AnswerT, JudgeCall]:
        """Ask the model and return its validated answer; raise when there is none."""

    @abstractmethod
    def verdict(self, request: RequestT, answer: AnswerT, call: JudgeCall) -> VerdictT:
        """What the answer means, in the caller's terms."""

    @abstractmethod
    def abstain(self, request: RequestT, reason: str) -> VerdictT:
        """The caller's "not judged" — never a pass, never a silent fail."""


class ModelJudge(LlmJudge[RequestT, AnswerT, VerdictT]):
    """A judge on the platform judge model: render messages, the rest is shared."""

    def __init__(
        self,
        schema: type[AnswerT],
        client: LLMClient | None = None,
        settings: JudgeSectionSettings | None = None,
        subsystem: str = "judge",
    ) -> None:
        super().__init__(subsystem)
        self._schema = schema
        self._settings = settings or get_settings().judge
        self._client = client

    @abstractmethod
    def render(self, request: RequestT) -> list[Message]:
        """The messages that ask this judge's question (the grading rules are prepended)."""

    async def _answer(self, request: RequestT) -> tuple[AnswerT, JudgeCall]:
        if self._client is None:
            self._client = create_judge_client(self._settings)
        return await ask_validated(self._client, self.render(request), self._schema, self._settings)


async def ask_validated(
    client: LLMClient,
    messages: list[Message],
    schema: type[AnswerT],
    settings: JudgeSectionSettings | None = None,
) -> tuple[AnswerT, JudgeCall]:
    """One judge call with the grading rules first, validated against `schema`."""
    judge = settings or get_settings().judge
    recording = RecordingLLMClient(client)
    result = await recording.complete(
        CompletionRequest(
            messages=with_rules(messages),
            temperature=judge.temperature,
            max_tokens=judge.max_tokens,
        )
    )
    validation = PydanticValidator(mode=judge.validation_mode).validate(result.content, schema)
    if not validation.valid or validation.value is None:
        raise JudgeAnswerError(f"{schema.__name__}: {'; '.join(validation.errors)}")
    call = JudgeCall(
        model=result.model or client.model, calls=recording.calls, cost_usd=recording.cost_usd
    )
    return schema.model_validate(validation.value), call


async def ask_text(
    client: LLMClient, prompt: str, settings: JudgeSectionSettings | None = None
) -> tuple[str, JudgeCall]:
    """One free-text judge call (a framework that parses the answer itself)."""
    judge = settings or get_settings().judge
    recording = RecordingLLMClient(client)
    result = await recording.complete(
        CompletionRequest(
            messages=with_rules([Message(role="user", content=prompt)]),
            temperature=judge.temperature,
            max_tokens=judge.max_tokens,
        )
    )
    call = JudgeCall(
        model=result.model or client.model, calls=recording.calls, cost_usd=recording.cost_usd
    )
    return result.content, call


def with_rules(messages: list[Message]) -> list[Message]:
    """The shared grading rules as the first system message, ahead of the judge's own."""
    return [Message(role="system", content=judge_rules()), *messages]


@cache
def judge_rules() -> str:
    """The calibration rules every platform judge is held to (`prompts/judge_rules.md`)."""
    return _RULES.read_text(encoding="utf-8").strip()
