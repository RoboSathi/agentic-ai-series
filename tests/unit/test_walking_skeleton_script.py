"""Test the walking skeleton script."""

from typing import Iterator

from agentic_commons.llm.types import (
    HealthCheckResult,
    LLMResponse,
    Message,
    StreamedResponse,
)
from customer_support_agent.dev.walking_skeleton import main


class _FakeLLMClient:
    """Fake LLMClient so the walking skeleton never hits the network in tests."""

    def __init__(self, stream_chunks: list[str] | None = None) -> None:
        self._stream_chunks = stream_chunks or ["[fake ", "streamed ", "response]"]

    def generate(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        return LLMResponse(content="[fake response]", model="fake")

    def generate_stream(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> StreamedResponse:
        response = LLMResponse(model="fake")

        def _chunks() -> Iterator[str]:
            for chunk in self._stream_chunks:
                response.content += chunk
                yield chunk

        return StreamedResponse(chunks=_chunks(), response=response)

    def health_check(self) -> HealthCheckResult:
        raise NotImplementedError


def test_walking_skeleton_main_runs() -> None:
    """Test that walking skeleton main function runs."""
    main(llm_client=_FakeLLMClient())


def test_walking_skeleton_uses_query_as_ticket_issue() -> None:
    """Test that a caller-supplied query replaces the hardcoded ticket issue."""
    fake = _FakeLLMClient()
    result = main(query="My custom refund question", llm_client=fake)
    assert result == "[fake response]"


def test_walking_skeleton_streams_when_callback_given() -> None:
    """Test that on_llm_chunk receives every delta and the final text is complete."""
    fake = _FakeLLMClient(stream_chunks=["Hello", ", ", "world!"])
    received: list[str] = []

    result = main(
        query="stream me", llm_client=fake, on_llm_chunk=received.append
    )

    assert received == ["Hello", ", ", "world!"]
    assert result == "Hello, world!"


def test_walking_skeleton_imports() -> None:
    """Test that walking skeleton imports work."""
    from agentic_commons.audit.log import AuditLog
    from agentic_commons.audit.types import AuditEvent
    from agentic_commons.gatekeeper.gatekeeper import Gatekeeper
    from agentic_commons.gatekeeper.types import Action
    from agentic_commons.guardrails.guards import PydanticGuard
    from agentic_commons.hallucination_protection.checker import HallucinationChecker
    from agentic_commons.llm.client import LLMClient
    from agentic_commons.memory.episodic.memory import EpisodicMemory
    from agentic_commons.memory.procedural.memory import ProceduralMemory
    from agentic_commons.memory.semantic.memory import SemanticMemory
    from agentic_commons.memory.working.memory import WorkingMemory
    from agentic_commons.security.content_moderation.moderator import ContentModerator
    from agentic_commons.security.pii.detector import PIIDetector
    from agentic_commons.security.prompt_injection.detector import InjectionDetector
    from agentic_commons.tools.registry import ToolRegistry

    assert AuditLog is not None
    assert AuditEvent is not None
    assert Gatekeeper is not None
    assert Action is not None
    assert PydanticGuard is not None
    assert HallucinationChecker is not None
    assert LLMClient is not None
    assert EpisodicMemory is not None
    assert ProceduralMemory is not None
    assert SemanticMemory is not None
    assert WorkingMemory is not None
    assert ContentModerator is not None
    assert PIIDetector is not None
    assert InjectionDetector is not None
    assert ToolRegistry is not None
