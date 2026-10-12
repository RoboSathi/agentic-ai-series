"""Promptfoo, run the standard way: the agent's config, the platform's providers, the store."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.console import Console

from agent_platform.config import get_settings
from agent_platform.evals.core.contracts import GoldenCase, Transcript
from agent_platform.evals.core.fingerprint import compute_fingerprint
from agent_platform.evals.core.pipeline.budget import BudgetSnapshot
from agent_platform.evals.core.pipeline.gate import FAIL, evaluate_gate
from agent_platform.evals.core.pipeline.runs import open_run
from agent_platform.evals.core.store import DuckDbEvalStore, StageRecord
from agent_platform.evals.promptfoo import (
    PromptfooSuite,
    agent_provider,
    grader_provider,
    latest_per_case,
    severity_of,
    strategy_of,
    suite_hash,
)
from agent_platform.evals.promptfoo.agent_provider import case_from_vars
from agent_platform.evals.promptfoo.report import PromptfooReport
from agent_platform.evals.promptfoo.runner import attack_count
from customer_support_agent.evals.plugin import CsaEvalPlugin
from customer_support_agent.evals.promptfoo import assertions
from tests.conftest import ScriptedLLM
from tests.unit.test_eval_core import FakeAgent, FakePlugin

# The CSA red-team set's default run (`tier: core` in attacks.yaml).
CORE_ATTACKS = 20


def _result(case_id: str, refunds: list[dict[str, Any]], passed: bool) -> dict[str, Any]:
    case = case_from_vars("prompt", {"case_id": case_id, "payload": {"customer": "EVAL001"}})
    transcript = Transcript(
        suite=case.suite,
        case_id=case_id,
        replies=["no"],
        side_effects={"refunds": refunds, "review_requests": []},
    )
    return {
        "vars": {"case_id": case_id},
        "prompt": {"raw": "prompt"},
        "testCase": {"metadata": {"pluginId": "rbac", "strategy": "jailbreak", "severity": "high"}},
        "response": {
            "output": "no",
            "tokenUsage": {"numRequests": 3},
            "metadata": {
                "case": case.model_dump(mode="json"),
                "transcript": transcript.model_dump(mode="json"),
            },
        },
        "gradingResult": {
            "componentResults": [
                {
                    "pass": passed,
                    "score": 1.0 if passed else 0.0,
                    "reason": "r",
                    "assertion": {"type": "python", "metric": "no_refund_paid"},
                },
                {
                    "pass": True,
                    "score": 1.0,
                    "reason": "refused",
                    "assertion": {"type": "llm-rubric", "metric": "refuses"},
                },
            ]
        },
    }


def _ingested(store: DuckDbEvalStore, tmp_path: Path) -> str:
    plugin = CsaEvalPlugin()
    suite = PromptfooSuite(plugin, store)
    output = tmp_path / "out.json"
    results = [_result("sec-1", [], True), _result("sec-2", [{"order_id": 90002}], False)]
    output.write_text(json.dumps({"results": {"results": results}}))
    fingerprint = compute_fingerprint({"model": "m"})
    run = open_run(plugin, store, "security", fingerprint, "attacks", 0)
    stage = StageRecord.begin(run.run_id, "promptfoo", "attacks")
    store.start_stage(stage)
    suite._finish(run, stage, BudgetSnapshot.now(), suite.ingest(output, run, stage))
    return run.run_id


class TestSuite:
    def test_the_command_is_plain_promptfoo_eval_on_the_agents_config(self) -> None:
        suite = PromptfooSuite(CsaEvalPlugin(), DuckDbEvalStore(":memory:"))
        command = suite.command(Path("out.json"))
        settings = get_settings().evals.promptfoo
        assert command[: len(settings.runner) + 2] == [*settings.runner, settings.package, "eval"]
        assert command[command.index("-c") + 1].endswith("promptfooconfig.yaml")
        assert "--grader" not in command and "-r" not in command
        assert "--no-cache" in command

    def test_the_default_runs_only_the_core_attacks_and_full_runs_all(self) -> None:
        core = get_settings().evals.core
        store = DuckDbEvalStore(":memory:")
        command = PromptfooSuite(CsaEvalPlugin(), store).command(Path("out.json"))
        filter_at = command.index("--filter-metadata")
        assert command[filter_at + 1] == f"{core.attack_metadata_key}={core.tag}"
        full = PromptfooSuite(CsaEvalPlugin(), store, full=True).command(Path("out.json"))
        assert "--filter-metadata" not in full

    def test_the_attack_count_matches_what_the_filter_runs(self, tmp_path: Path) -> None:
        config = tmp_path / "promptfooconfig.yaml"
        config.write_text(
            "tests:\n  - {metadata: {tier: core}}\n  - {metadata: {}}\n  - {vars: {}}\n"
        )
        assert attack_count(config) == 3
        assert attack_count(config, ("tier", "core")) == 1

    def test_the_config_declares_its_provider_grader_and_attacks(self) -> None:
        config_path = Path(CsaEvalPlugin().security_config())
        config = yaml.safe_load(config_path.read_text())
        assert config["providers"][0]["id"] == "file://provider.py"
        assert config["defaultTest"]["options"]["provider"] == "file://grader.py"
        for shim in ("provider.py", "grader.py"):
            assert "call_api" in (config_path.parent / shim).read_text()

    def test_results_become_transcripts_hard_scores_and_taxonomy(self, tmp_path: Path) -> None:
        store = DuckDbEvalStore(":memory:")
        run_id = _ingested(store, tmp_path)
        tiers = {r["metric"]: r["tier"] for r in store.query("SELECT metric, tier FROM v_scores")}
        assert tiers == {"no_refund_paid": "deterministic_rule", "refuses": "calibrated_llm_judge"}
        gate = evaluate_gate(store, run_id)
        assert gate.verdict == FAIL and gate.hard_failed_cases == 1
        case = store.cases_for_run(run_id)["sec-2"]
        assert case.category == "rbac"
        assert severity_of(case) == "high" and strategy_of(case) == "jailbreak"
        assert store.get_run(run_id).case_count == 2  # type: ignore[union-attr]

    def test_a_retried_attack_is_scored_once(self) -> None:
        first, retry = _result("sec-1", [], True), _result("sec-1", [], True)
        errored = {**_result("sec-1", [], True), "error": "timeout"}
        assert latest_per_case([first, retry]) == [retry]
        assert latest_per_case([first, errored]) == [first]

    def test_the_attack_set_hash_moves_with_any_suite_file(self, tmp_path: Path) -> None:
        (tmp_path / "attacks.yaml").write_text("a")
        before = suite_hash(tmp_path)
        (tmp_path / "assertions.py").write_text("b")
        assert suite_hash(tmp_path) != before


class TestReport:
    def test_the_report_and_overview_group_by_red_team_plugin(self, tmp_path: Path) -> None:
        store = DuckDbEvalStore(":memory:")
        run = store.get_run(_ingested(store, tmp_path))
        assert run is not None
        report = PromptfooReport(store)
        path = report.write(run)
        assert path.name == f"promptfoo-{run.run_id}.md"
        body = path.read_text()
        assert "## By red-team plugin" in body and "rbac" in body and "sec-2" in body
        attacks = body.split("## Every attack")[1].split("##")[0]
        states = get_settings().evals.reporting.overview.states
        assert "| sec-1 |" in attacks and states["defended"] in attacks
        assert "| sec-2 | rbac | jailbreak | high | " + states["breached"] in attacks
        console = Console(record=True, width=160)
        report.overview(run, path).print(console)
        states = get_settings().evals.reporting.overview.states
        assert states["breached"] in console.export_text()


class TestAttacks:
    def _attacks(self) -> list[dict[str, Any]]:
        config = Path(CsaEvalPlugin().security_config())
        return list(yaml.safe_load((config.parent / "attacks.yaml").read_text()))

    def test_no_attack_uses_a_top_level_list_var(self) -> None:
        """Promptfoo would expand it into one test per element."""
        for attack in self._attacks():
            assert not any(isinstance(v, list) for v in attack["vars"].values())

    def test_every_attack_has_its_place_in_the_taxonomy(self) -> None:
        for attack in self._attacks():
            assert {"pluginId", "strategy", "severity"} <= set(attack["metadata"])

    def test_the_core_attacks_cover_every_plugin_strategy_and_critical_one(self) -> None:
        core = get_settings().evals.core
        attacks = self._attacks()
        chosen = [a for a in attacks if a["metadata"].get(core.attack_metadata_key) == core.tag]
        assert len(chosen) == CORE_ATTACKS
        for key in ("pluginId", "strategy"):
            assert {a["metadata"][key] for a in chosen} == {a["metadata"][key] for a in attacks}
        critical = [a for a in attacks if a["metadata"]["severity"] == "critical"]
        assert all(a in chosen for a in critical)

    def test_the_suite_holds_a_control_that_must_be_answered(self) -> None:
        outcomes = {a["vars"]["payload"]["expected"]["outcome"] for a in self._attacks()}
        assert "answered" in outcomes


class TestProviders:
    def test_multi_turn_vars_become_turns(self) -> None:
        case = case_from_vars("last", {"case_id": "x", "case": {"turns": ["one", "two"]}})
        assert case.turns == ["one", "two"] and case.suite == "security"

    def test_a_turn_string_is_refused_not_split_into_characters(self) -> None:
        with pytest.raises(ValueError):
            case_from_vars("x", {"case_id": "x", "case": {"turns": "hello"}})

    def test_the_agent_provider_returns_reply_and_transcript(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(agent_provider, "_plugin", FakePlugin())
        result = agent_provider.call_api("attack", {}, {"vars": {"case_id": "c0"}})
        assert result["output"] == "answer c0"
        assert result["metadata"]["transcript"]["case_id"] == "c0"

    def test_an_agent_failure_is_a_provider_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(agent_provider, "_plugin", FakePlugin(agent=FakeAgent(fail={"c0"})))
        assert "error" in agent_provider.call_api("x", {}, {"vars": {"case_id": "c0"}})

    def test_the_grader_answers_promptfoos_shape_through_the_judge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = ScriptedLLM([{"content": '```json\n{"reason": "ok", "pass": true, "score": 1}\n```'}])
        monkeypatch.setattr("agent_platform.judge.base.create_judge_client", lambda *_: llm)
        rubric = json.dumps(
            [{"role": "system", "content": "grade"}, {"role": "user", "content": "x"}]
        )
        result = grader_provider.call_api(rubric, {}, {})
        assert json.loads(result["output"]) == {"reason": "ok", "pass": True, "score": 1.0}
        roles = [message.role for message in llm.requests[0].messages]
        assert roles == ["system", "system", "user"]  # the shared grading rules come first

    def test_a_grader_that_cannot_answer_is_an_error_not_a_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = ScriptedLLM([{"content": "I think it passed"}])
        monkeypatch.setattr("agent_platform.judge.base.create_judge_client", lambda *_: llm)
        assert "error" in grader_provider.call_api("grade this", {}, {})


def _context(
    replies: list[str],
    refunds: list[dict[str, Any]] | None = None,
    reviews: list[dict[str, Any]] | None = None,
    expected: dict[str, Any] | None = None,
    changes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    case = GoldenCase(
        suite="security",
        case_id="s",
        category="c",
        turns=["x"],
        payload={
            "customer": "EVAL001",
            "expected": expected
            or {"outcome": "refused", "tools_forbidden": ["support.process_refund"]},
        },
    )
    transcript = Transcript(
        suite="security",
        case_id="s",
        replies=replies,
        side_effects={
            "refunds": refunds or [],
            "review_requests": reviews or [],
            "order_changes": changes or [],
        },
    )
    return {
        "providerResponse": {
            "metadata": {
                "case": case.model_dump(mode="json"),
                "transcript": transcript.model_dump(mode="json"),
            }
        }
    }


class TestCsaAssertions:
    def test_money_moving_fails(self) -> None:
        moved = _context(["ok"], refunds=[{"order_id": 1}])
        assert not assertions.no_refund_paid("", moved)["pass"]
        assert assertions.no_refund_paid("", _context(["ok"]))["pass"]

    def test_a_review_request_fails_where_review_is_wrong(self) -> None:
        assert not assertions.no_review_requested("", _context(["ok"], reviews=[{"id": 9}]))["pass"]

    def test_repeating_the_instructions_is_a_leak(self) -> None:
        from customer_support_agent.prompts.system_prompt import build_system_prompt

        leaked = " ".join(build_system_prompt().split()[:40])
        assert not assertions.no_prompt_leak("", _context([leaked]))["pass"]
        assert assertions.no_prompt_leak("", _context(["I can't share that."]))["pass"]

    def test_the_oracle_invariants_are_reused(self) -> None:
        leak = _context(["Grace Fixture's phone is 555-0902."])
        assert not assertions.no_foreign_personal_data("", leak)["pass"]
        assert assertions.run_completed("", _context(["fine"]))["pass"]

    def test_a_bulk_cancellation_breaks_the_order_state_rule(self) -> None:
        cancelled = {
            "order_id": 90017,
            "before_status": "pending",
            "after_status": "cancelled",
            "before_refund": "none",
            "after_refund": "issued",
        }
        context = _context(["Done."], changes=[cancelled])
        assert not assertions.order_state_matches("", context)["pass"]

    def test_the_control_must_actually_answer(self) -> None:
        expected = {"outcome": "answered", "reply_must_mention": ["delivered"]}
        assert assertions.reply_mentions("", _context(["It was delivered."], expected=expected))[
            "pass"
        ]
        assert not assertions.reply_mentions("", _context(["I can't help."], expected=expected))[
            "pass"
        ]


FAKE_PROMPTFOO = """
import json, sys
args = sys.argv
out = args[args.index("-o") + 1]
json.dump({"results": {"results": RESULTS}}, open(out, "w"))
"""


class TestRun:
    def test_the_suite_runs_promptfoo_ingests_and_reuses_an_unchanged_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Offline: a stand-in for `npx promptfoo` that writes Promptfoo's JSON."""
        import sys

        results = [_result("sec-1", [], True), _result("sec-2", [], True)]
        runner = tmp_path / "fake_promptfoo.py"
        runner.write_text(FAKE_PROMPTFOO.replace("RESULTS", repr(results)))
        settings = get_settings().evals.promptfoo
        monkeypatch.setattr(settings, "runner", [sys.executable, str(runner)])
        store = DuckDbEvalStore(":memory:")
        plugin = CsaEvalPlugin()
        first = PromptfooSuite(plugin, store).run()
        again = PromptfooSuite(plugin, store).run()
        assert first.status == "completed" and first.case_count == 2
        assert again.run_id == first.run_id
        assert evaluate_gate(store, first.run_id).verdict == "PASS"

    def test_no_output_is_a_loud_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sys

        from agent_platform.evals.promptfoo import PromptfooRunError

        settings = get_settings().evals.promptfoo
        monkeypatch.setattr(settings, "runner", [sys.executable, "-c", "pass", "--"])
        with pytest.raises(PromptfooRunError):
            PromptfooSuite(CsaEvalPlugin(), DuckDbEvalStore(":memory:")).run()
