"""The provider boundary (security/pii/boundary.py): values leave as tokens, come back as values.

A scripted inner client records exactly what the provider would have
received. The pattern detector stands in for the NER model: what is under
test is the boundary, not detection.
"""

import re

from agent_platform.llm.client import LLMClient
from agent_platform.llm.types import (
    CompletionRequest,
    CompletionResult,
    HealthCheckResult,
    Message,
    ModelToolCall,
)
from agent_platform.security.pii.boundary import PiiBoundary
from agent_platform.security.pii.detector import PatternPIIDetector, PIIFilterImpl

_EMAIL = "ada@example.com"
_TOKEN = re.compile(r"\[EMAIL_ADDRESS:[0-9a-f]+\]")


class _Provider(LLMClient):
    """Answers by echoing the first token it was sent (or a canned line)."""

    def __init__(self, answer: str | None = None) -> None:
        self.sent: list[CompletionRequest] = []
        self._answer = answer

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        self.sent.append(request)
        tokens = _TOKEN.findall(" ".join(m.content for m in request.messages))
        echoed = self._answer or (f"We wrote to {tokens[0]}." if tokens else "Nothing.")
        call = ModelToolCall(id="c1", name="email", arguments=f'{{"to": "{echoed}"}}')
        return CompletionResult(content=echoed, tool_calls=[call])

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult(healthy=True, provider="test", model="m")

    @property
    def tier(self) -> str:
        return "test"


def _boundary(provider: _Provider) -> PiiBoundary:
    return PiiBoundary(provider, PIIFilterImpl(detectors=[PatternPIIDetector()], vault=None))


def _request(*contents: str) -> CompletionRequest:
    roles = ["system", "user", "tool"]
    return CompletionRequest(
        messages=[Message(role=roles[i % 3], content=c) for i, c in enumerate(contents)]  # type: ignore[arg-type]
    )


async def test_the_provider_never_receives_the_value() -> None:
    provider = _Provider()
    await _boundary(provider).complete(
        _request("Customer: ada@example.com", "hi", f'{{"email": "{_EMAIL}"}}')
    )
    [sent] = provider.sent
    assert all(_EMAIL not in m.content for m in sent.messages)
    assert all(_TOKEN.search(m.content) for m in (sent.messages[0], sent.messages[2]))


async def test_the_answer_and_tool_arguments_get_the_value_back() -> None:
    result = await _boundary(_Provider()).complete(_request(f"Customer: {_EMAIL}"))
    assert result.content == f"We wrote to {_EMAIL}."
    assert result.tool_calls is not None and _EMAIL in result.tool_calls[0].arguments


async def test_a_token_this_request_did_not_send_stays_a_token() -> None:
    stranger = "[EMAIL_ADDRESS:0123456789ab]"
    result = await _boundary(_Provider(answer=f"Mail {stranger}")).complete(
        _request(f"Customer: {_EMAIL}")
    )
    assert result.content == f"Mail {stranger}"


async def test_the_same_value_gets_the_same_token_on_every_call() -> None:
    provider = _Provider()
    boundary = _boundary(provider)
    for _ in range(2):
        await boundary.complete(_request(f"Customer: {_EMAIL}"))
    first, second = (_TOKEN.findall(r.messages[0].content) for r in provider.sent)
    assert first == second


async def test_text_without_personal_data_passes_untouched() -> None:
    provider = _Provider(answer="ok")
    result = await _boundary(provider).complete(_request("What is the status of order 10001?"))
    assert provider.sent[0].messages[0].content == "What is the status of order 10001?"
    assert result.content == "ok"
