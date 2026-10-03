"""`python -m agent_platform.evals` and the Promptfoo provider entry points."""

from typing import Any

import pytest

from agent_platform.evals import cli
from agent_platform.evals.adapters import promptfoo_agent_provider, promptfoo_judge_provider
from agent_platform.evals.store import DuckDbEvalStore
from tests.conftest import ScriptedLLM
from tests.unit.test_eval_framework import FakeAgent, FakeGrader, FakePlugin


@pytest.fixture
def plugin(monkeypatch: pytest.MonkeyPatch) -> FakePlugin:
    fake = FakePlugin(grader=FakeGrader(fail={"c1"}))
    monkeypatch.setattr(cli, "resolve_eval_plugin", lambda: fake)
    return fake


class TestCli:
    def test_generate_then_oracle_then_report(self, plugin: FakePlugin) -> None:
        assert cli.main(["generate", "--limit", "2"]) == 0
        assert plugin.agent.calls == ["c0", "c1"]
        # A broken hard invariant makes the oracle stage exit non-zero.
        assert cli.main(["oracle"]) == 1
        assert cli.main(["report"]) == 0
        assert cli.main(["runs"]) == 0
        assert cli.main(["sql", "select count(*) as n from eval_runs"]) == 0

    def test_selection_by_case_and_tag(self, plugin: FakePlugin) -> None:
        assert cli.main(["generate", "--case", "c2"]) == 0
        assert plugin.agent.calls == ["c2"]

    def test_a_stage_with_no_run_is_a_message(self, plugin: FakePlugin) -> None:
        DuckDbEvalStore().execute("DELETE FROM eval_runs")
        assert cli.main(["judge"]) == 1

    def test_run_stops_after_a_broken_invariant(
        self, plugin: FakePlugin, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called: list[str] = []
        monkeypatch.setattr(cli, "_security", lambda args: called.append("security") or 0)
        assert cli.main(["run", "--fresh"]) == 1
        assert called == []

    def test_diff_between_two_runs(self, plugin: FakePlugin) -> None:
        cli.main(["generate"])
        base = DuckDbEvalStore().latest_run("golden")
        plugin.fingerprint_inputs = lambda: {"model": "other"}  # type: ignore[method-assign]
        cli.main(["generate"])
        head = DuckDbEvalStore().latest_run("golden")
        assert base and head and base.run_id != head.run_id
        assert cli.main(["diff", base.run_id, head.run_id]) == 0


class TestPromptfooProviders:
    def test_the_agent_provider_returns_reply_and_transcript(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakePlugin()
        monkeypatch.setattr(promptfoo_agent_provider, "_plugin", fake)
        result = promptfoo_agent_provider.call_api(
            "attack", {}, {"vars": {"case_id": "c0", "category": "x"}}
        )
        assert result["output"] == "answer c0"
        assert result["metadata"]["transcript"]["case_id"] == "c0"

    def test_an_agent_failure_is_a_provider_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakePlugin(agent=FakeAgent(fail={"c0"}))
        monkeypatch.setattr(promptfoo_agent_provider, "_plugin", fake)
        result = promptfoo_agent_provider.call_api(
            "x", {}, {"vars": {"case_id": "c0", "category": "x"}}
        )
        assert "error" in result

    def test_the_judge_provider_uses_the_judge_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = ScriptedLLM([{"content": '{"pass": true, "score": 1, "reason": "ok"}'}])
        monkeypatch.setattr(promptfoo_judge_provider, "create_pinned_llm_client", lambda *_: llm)
        result: dict[str, Any] = promptfoo_judge_provider.call_api("grade this", {}, {})
        assert '"pass": true' in result["output"]
        assert llm.requests[0].messages[0].content == "grade this"
