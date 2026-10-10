"""LangSmith export: one root run per turn, redacted, and never in a turn's way.

The SDK is real; only the client is a mock, so what is asserted is what the
SDK would have sent — no network, no model.
"""

import time
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from agent_platform.config import get_settings
from agent_platform.config.settings import LangSmithSettings
from agent_platform.llm.metering import record_usage
from agent_platform.llm.types import TokenUsage
from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter
from agent_platform.observability.langsmith import (
    LangSmithExporter,
    NullTraceExporter,
    _routing_free_client,
    build_trace_exporter,
    current_trace_url,
    is_routing_run,
    redact_tree,
    tool_run,
    traced_tool,
    turn_trace,
)

AGENT = "agent-under-test"


def _settings(**changes: Any) -> LangSmithSettings:
    return get_settings().observability.langsmith.model_copy(update=changes)


def _calls(client: MagicMock, method: str) -> list[dict[str, Any]]:
    return [call.kwargs for call in client.method_calls if call[0] == method]


def _exporter(client: MagicMock) -> LangSmithExporter:
    return LangSmithExporter(client, _settings(), AGENT)


def _redact(text: str) -> str:
    return text.replace("ada@example.com", "<EMAIL>")


class TestBuild:
    def test_withheld_export_is_the_no_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(get_settings().observability, "langsmith", _settings(enabled=False))
        assert isinstance(build_trace_exporter(AGENT, lambda: _redact), NullTraceExporter)

    def test_no_key_is_the_no_op(self) -> None:
        # The session fixture has removed every configured key variable.
        assert isinstance(build_trace_exporter(AGENT, lambda: _redact), NullTraceExporter)

    def test_the_redactor_is_not_built_when_nothing_is_exported(self) -> None:
        def redactor() -> Any:
            raise AssertionError("a withheld exporter built the PII model")

        build_trace_exporter(AGENT, redactor)

    def test_every_payload_is_redacted_before_it_leaves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = MagicMock()
        # Export is off by default; this test is about what an enabled one sends.
        monkeypatch.setattr(get_settings().observability.langsmith, "enabled", True)
        monkeypatch.setenv(_settings().key_env_vars[0], "key")
        monkeypatch.setattr(
            "agent_platform.observability.langsmith._routing_free_client", lambda: client
        )
        build_trace_exporter(AGENT, lambda: _redact)
        hooks = client.call_args.kwargs
        payload = {"query": "refund ada@example.com"}
        assert hooks["hide_inputs"](payload) == {"query": "refund <EMAIL>"}
        assert hooks["hide_outputs"](payload) == {"query": "refund <EMAIL>"}

    def test_the_first_set_key_variable_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        first, second = _settings().key_env_vars[:2]
        monkeypatch.setenv(second, "second")
        assert _settings().api_key() == "second"
        monkeypatch.setenv(first, "first")
        assert _settings().api_key() == "first"


class TestTurn:
    def test_the_root_run_id_is_the_turns_trace_id(self) -> None:
        client = MagicMock()
        trace_id = uuid.uuid4().hex
        with _exporter(client).turn(trace_id, {"query": "q"}, {}):
            pass
        assert _calls(client, "create_run")[0]["id"] == uuid.UUID(hex=trace_id)

    def test_each_agent_has_its_own_project(self) -> None:
        client = MagicMock()
        with _exporter(client).turn(uuid.uuid4().hex, {"query": "q"}, {}):
            pass
        assert _calls(client, "create_run")[0]["session_name"] == _settings().project.format(
            agent_id=AGENT
        )

    def test_what_the_turn_produced_is_recorded(self) -> None:
        client = MagicMock()
        with _exporter(client).turn(uuid.uuid4().hex, {"query": "q"}, {}) as root:
            root.finish({"response": "done"})
        assert _calls(client, "update_run")[0]["outputs"] == {"response": "done"}

    def test_a_tool_inside_the_turn_is_its_child(self) -> None:
        client = MagicMock()
        trace_id = uuid.uuid4().hex
        with _exporter(client).turn(trace_id, {"query": "q"}, {}):
            tool_run("lookup_order", lambda order: order)("10010")
        tool = next(run for run in _calls(client, "create_run") if run["name"] == "lookup_order")
        assert tool["trace_id"] == uuid.UUID(hex=trace_id)

    def test_a_decorated_method_is_a_tool_run_named_for_itself(self) -> None:
        class Gateway:
            @traced_tool
            def search(self, query: str) -> str:
                return query

        client = MagicMock()
        with _exporter(client).turn(uuid.uuid4().hex, {"query": "q"}, {}):
            Gateway().search("agents")
        runs = {run["name"]: run for run in _calls(client, "create_run")}
        assert runs["search"]["run_type"] == "tool"

    def test_outside_a_turn_a_tool_is_not_exported(self) -> None:
        assert tool_run("lookup_order", lambda order: order)("10010") == "10010"


class TestSampling:
    def test_a_sampled_out_turn_sends_nothing_and_has_no_link(self) -> None:
        client = MagicMock()
        exporter = LangSmithExporter(client, _settings(sample_rate=0.0), AGENT)
        trace_id = uuid.uuid4().hex
        with exporter.turn(trace_id, {"query": "q"}, {}) as root:
            root.annotate({"intent": "x"})
            assert exporter.url_for(trace_id) is None
        assert _calls(client, "create_run") == []


class TestLink:
    def test_no_link_until_the_project_resolves(self) -> None:
        assert _exporter(MagicMock()).url_for(uuid.uuid4().hex) is None

    def test_the_link_names_workspace_project_and_run(self) -> None:
        client = MagicMock()
        client.create_project.return_value = MagicMock(tenant_id="t-1", id="p-1")
        exporter = _exporter(client)
        trace_id = uuid.uuid4().hex
        with exporter.turn(trace_id, {"query": "q"}, {}):
            pass
        deadline = time.monotonic() + 2
        while exporter.url_for(trace_id) is None and time.monotonic() < deadline:
            time.sleep(0.01)
        url = exporter.url_for(trace_id) or ""
        assert "t-1" in url and "p-1" in url and str(uuid.UUID(hex=trace_id)) in url

    def test_a_failed_lookup_leaves_no_link_and_no_error(self) -> None:
        client = MagicMock()
        client.create_project.side_effect = RuntimeError("offline")
        exporter = _exporter(client)
        trace_id = uuid.uuid4().hex
        with exporter.turn(trace_id, {"query": "q"}, {}):
            pass
        assert exporter.url_for(trace_id) is None


class TestTurnTrace:
    def _linked(self) -> Any:
        exporter = MagicMock(wraps=NullTraceExporter())
        exporter.url_for.return_value = "https://example.invalid/r/1"
        return exporter

    def test_the_link_is_known_only_inside_the_turn(self) -> None:
        exporter = self._linked()
        with turn_trace(exporter, agent=AGENT, session="s", inputs={}):
            assert current_trace_url() == "https://example.invalid/r/1"
        assert current_trace_url() is None

    def test_the_turn_summary_carries_the_link(self) -> None:
        feed = InMemoryLiveStatusFeed(max_events=0)
        reporter = StageReporter(feed, "r")
        with turn_trace(self._linked(), agent=AGENT, session="s", inputs={}):
            reporter.begin_turn(session="s")
            reporter.end_turn()
        assert feed.events[-1].fields[_settings().url_field] == "https://example.invalid/r/1"

    def test_no_link_no_field(self) -> None:
        feed = InMemoryLiveStatusFeed(max_events=0)
        reporter = StageReporter(feed, "r")
        with turn_trace(NullTraceExporter(), agent=AGENT, session="s", inputs={}):
            reporter.begin_turn(session="s")
            reporter.end_turn()
        assert _settings().url_field not in feed.events[-1].fields

    def test_the_summary_counts_model_calls_from_the_meter(self) -> None:
        feed = InMemoryLiveStatusFeed(max_events=0)
        reporter = StageReporter(feed, "r")
        reporter.begin_turn(session="s")
        record_usage(TokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))
        reporter.emit("observation", "completed", event_kind="tool_result", fields={"tool": "t"})
        reporter.end_turn()
        record_usage(TokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10))
        fields = feed.events[-1].fields
        assert (fields["model_calls"], fields["tokens"], fields["tools"]) == (1, 10, "t")

    def test_a_failing_turn_still_raises(self) -> None:
        with pytest.raises(RuntimeError):
            with turn_trace(NullTraceExporter(), agent=AGENT, session="s", inputs={}):
                raise RuntimeError("boom")


class _State(BaseModel):
    query: str
    history: list[str]


@pytest.mark.parametrize("full", [True, False])
def test_a_step_is_sent_as_what_it_changed_unless_full_state_is_asked(full: bool) -> None:
    client = _routing_free_client().__new__(_routing_free_client())
    client._full_node_state, client._dropped = full, set()
    sent: list[dict[str, Any]] = []
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("langsmith.Client.create_run", lambda self, *a, **k: sent.append(k))
        step = {"metadata": {"langgraph_node": "reason"}}
        client.create_run(name="reason", run_type="chain", inputs={"query": "q"}, extra=step)
        client.create_run(name="_after_reason", run_type="chain", inputs={}, extra=step, id="r")
        patch.setattr("langsmith.Client.update_run", lambda self, run_id, **k: sent.append(k))
        client.update_run("n", name="reason", run_type="chain", inputs={"query": "q"}, extra=step)
    assert [run["inputs"] for run in sent] == (
        [{"query": "q"}, {"query": "q"}] if full else [{}, None]
    )
    assert client.update_run("r") is None


def test_a_routing_edge_is_not_a_step() -> None:
    edge = {"metadata": {"langgraph_node": "reason"}}
    assert is_routing_run("_after_reason", "chain", edge)
    assert not is_routing_run("reason", "chain", edge)
    assert not is_routing_run("ChatOpenAI", "llm", edge)
    assert not is_routing_run("LangGraph", "chain", {"metadata": {}})


def test_the_turn_carries_its_thread_and_versions() -> None:
    client = MagicMock()
    with turn_trace(_exporter(client), agent=AGENT, session="s-1", inputs={}):
        pass
    metadata = _calls(client, "create_run")[0]["extra"]["metadata"]
    assert metadata[_settings().thread_key] == "s-1"
    assert metadata["policy_version"] == get_settings().audit.policy_version


def test_redaction_sees_inside_a_state_model() -> None:
    """A graph node's input is the agent's state model, not a dict."""
    payload = {"input": _State(query="I am ada@example.com", history=["ada@example.com"])}
    assert redact_tree(payload, _redact) == {
        "input": {"query": "I am <EMAIL>", "history": ["<EMAIL>"]}
    }


def test_redaction_reaches_every_nested_string() -> None:
    payload = {
        "messages": [{"content": "mail ada@example.com"}],
        "n": 3,
        "tags": ("ada@example.com",),
    }
    assert redact_tree(payload, _redact) == {
        "messages": [{"content": "mail <EMAIL>"}],
        "n": 3,
        "tags": ["<EMAIL>"],
    }
