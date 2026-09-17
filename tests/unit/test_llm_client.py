"""Unit tests for the OpenAI-compatible LLM client adapter."""

from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, omit

from agentic_commons.config.settings import LLMSettings
from agentic_commons.llm.client import (
    LLMClient,
    OpenAICompatibleClient,
    create_llm_client,
)
from agentic_commons.llm.types import Message, ToolCall


def _fake_completion(
    content: str = "hello",
    model: str = "test-model",
    finish_reason: str = "stop",
    total_tokens: int = 42,
    tool_calls: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    """Build a duck-typed stand-in for openai's ChatCompletion response."""
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(total_tokens=total_tokens),
    )


def _fake_tool_call(
    call_id: str = "call-1", name: str = "get_order_status", arguments: str = "{}"
) -> SimpleNamespace:
    """Build a duck-typed stand-in for an openai response tool call."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _FakeOpenAI:
    """Duck-typed stand-in for openai.OpenAI, recording calls made to it."""

    def __init__(self, completion: SimpleNamespace | None = None) -> None:
        self.create_calls: list[dict[str, object]] = []
        self.models_list_calls = 0
        self._completion = completion or _fake_completion()
        self._models_error: Exception | None = None

        fake_client = self

        class _Completions:
            def create(self, **kwargs: object) -> SimpleNamespace:
                fake_client.create_calls.append(kwargs)
                return fake_client._completion

        class _Chat:
            completions = _Completions()

        class _Models:
            def list(self) -> None:
                fake_client.models_list_calls += 1
                if fake_client._models_error is not None:
                    raise fake_client._models_error

        self.chat = _Chat()
        self.models = _Models()

    def fail_models_list_with(self, error: Exception) -> None:
        self._models_error = error


def _settings(**overrides: object) -> LLMSettings:
    return LLMSettings(
        provider="local",
        base_url="http://fake/v1",
        model="fake-model",
        timeout_seconds=30,
        max_tokens=99,
        **overrides,  # type: ignore[arg-type]
    )


def test_generate_sends_model_and_messages() -> None:
    fake = _FakeOpenAI()
    client = OpenAICompatibleClient(settings=_settings(), client=fake)  # type: ignore[arg-type]

    response = client.generate([Message(role="user", content="hi")])

    assert fake.create_calls[0]["model"] == "fake-model"
    assert fake.create_calls[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert response.content == "hello"
    assert response.model == "test-model"
    assert response.tokens_used == 42
    assert response.finish_reason == "stop"


def test_generate_defaults_max_tokens_from_settings_and_omits_temperature() -> None:
    """max_tokens always has a bound (from config), temperature is left to the API."""
    fake = _FakeOpenAI()
    client = OpenAICompatibleClient(settings=_settings(), client=fake)  # type: ignore[arg-type]

    client.generate([Message(role="user", content="hi")])

    assert fake.create_calls[0]["temperature"] is omit
    assert fake.create_calls[0]["max_tokens"] == 99


def test_generate_forwards_temperature_and_max_tokens() -> None:
    fake = _FakeOpenAI()
    client = OpenAICompatibleClient(settings=_settings(), client=fake)  # type: ignore[arg-type]

    client.generate([Message(role="user", content="hi")], temperature=0.2, max_tokens=64)

    assert fake.create_calls[0]["temperature"] == 0.2
    assert fake.create_calls[0]["max_tokens"] == 64


def test_generate_omits_tools_by_default() -> None:
    fake = _FakeOpenAI()
    client = OpenAICompatibleClient(settings=_settings(), client=fake)  # type: ignore[arg-type]

    client.generate([Message(role="user", content="hi")])

    assert fake.create_calls[0]["tools"] is omit


def test_generate_forwards_tools() -> None:
    fake = _FakeOpenAI()
    client = OpenAICompatibleClient(settings=_settings(), client=fake)  # type: ignore[arg-type]
    tools = [{"type": "function", "function": {"name": "get_order_status"}}]

    client.generate([Message(role="user", content="hi")], tools=tools)

    assert fake.create_calls[0]["tools"] == tools


def test_generate_extracts_tool_calls_from_response() -> None:
    fake = _FakeOpenAI(
        completion=_fake_completion(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _fake_tool_call(
                    call_id="call-1",
                    name="get_order_status",
                    arguments='{"order_id": "12345"}',
                )
            ],
        )
    )
    client = OpenAICompatibleClient(settings=_settings(), client=fake)  # type: ignore[arg-type]

    response = client.generate([Message(role="user", content="status of order 12345?")])

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls == [
        ToolCall(id="call-1", name="get_order_status", arguments='{"order_id": "12345"}')
    ]


def test_generate_tool_calls_none_when_absent() -> None:
    fake = _FakeOpenAI()
    client = OpenAICompatibleClient(settings=_settings(), client=fake)  # type: ignore[arg-type]

    response = client.generate([Message(role="user", content="hi")])

    assert response.tool_calls is None


def test_to_openai_message_for_assistant_with_tool_calls() -> None:
    from agentic_commons.llm.client import _to_openai_message

    message = Message(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="call-1", name="get_order_status", arguments='{"order_id": "1"}')],
    )

    result = _to_openai_message(message)

    assert result == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "get_order_status", "arguments": '{"order_id": "1"}'},
            }
        ],
    }


def test_to_openai_message_for_tool_role() -> None:
    from agentic_commons.llm.client import _to_openai_message

    message = Message(role="tool", content="order is shipped", tool_call_id="call-1")

    result = _to_openai_message(message)

    assert result == {
        "role": "tool",
        "content": "order is shipped",
        "tool_call_id": "call-1",
    }


def test_health_check_success() -> None:
    fake = _FakeOpenAI()
    client = OpenAICompatibleClient(settings=_settings(), client=fake)  # type: ignore[arg-type]

    result = client.health_check()

    assert result.healthy is True
    assert result.error is None
    assert fake.models_list_calls == 1


def test_health_check_reports_connection_failure() -> None:
    fake = _FakeOpenAI()
    fake.fail_models_list_with(
        APIConnectionError(request=httpx.Request("GET", "http://fake/v1/models"))
    )
    client = OpenAICompatibleClient(settings=_settings(), client=fake)  # type: ignore[arg-type]

    result = client.health_check()

    assert result.healthy is False
    assert result.error is not None


def test_openai_compatible_client_satisfies_llm_client_protocol() -> None:
    client = OpenAICompatibleClient(settings=_settings(), client=_FakeOpenAI())  # type: ignore[arg-type]
    assert isinstance(client, LLMClient)


def test_create_llm_client_uses_given_settings() -> None:
    settings = LLMSettings(provider="local", model="explicit-model")

    client = create_llm_client(settings)

    assert isinstance(client, OpenAICompatibleClient)
    assert client._settings is settings


@pytest.mark.parametrize("provider", ["nim", "local"])
def test_create_llm_client_builds_real_openai_client_without_network_call(
    provider: str,
) -> None:
    """Constructing a client must not make any network calls."""
    settings = LLMSettings(provider=provider)  # type: ignore[arg-type]

    client = create_llm_client(settings)

    assert isinstance(client, OpenAICompatibleClient)
