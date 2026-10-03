"""Optional-tool bridges (DeepEval, Promptfoo), the security assertions, and `/eval`."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from agent_platform.commands.evals import EvalCommand
from agent_platform.evals.adapters.deepeval_adapter import DeepEvalGrader, deepeval_graders
from agent_platform.evals.adapters.promptfoo import (
    AGENT_PROVIDER,
    JUDGE_PROVIDER,
    PromptfooSuite,
    latest_per_case,
    suite_hash,
)
from agent_platform.evals.adapters.promptfoo_agent_provider import case_from_vars
from agent_platform.evals.contracts import GoldenCase, QualityMetricSpec, Transcript
from agent_platform.evals.pipeline.gate import FAIL, evaluate_gate
from agent_platform.evals.store import DuckDbEvalStore
from customer_support_agent.evals.plugin import CsaEvalPlugin
from customer_support_agent.evals.security import assertions
from tests.conftest import ScriptedLLM

CASE = GoldenCase(suite="golden", case_id="c0", category="cat", turns=["where is my order?"])
TRANSCRIPT = Transcript(suite="golden", case_id="c0", replies=["It ships Monday."])


class TestDeepEval:
    async def test_a_metric_runs_on_the_harness_client(self) -> None:
        llm = ScriptedLLM(
            [
                {"content": json.dumps({"statements": ["It ships Monday."]})},
                {"content": json.dumps({"verdicts": [{"verdict": "yes", "reason": ""}]})},
            ]
        )
        spec = QualityMetricSpec(name="relevancy", kind="answer_relevancy", threshold=0.5)
        scores = await DeepEvalGrader(spec, client=llm).grade(CASE, TRANSCRIPT)
        assert scores[0].metric == "relevancy" and scores[0].passed
        assert scores[0].llm_calls == len(llm.requests) >= 1

    async def test_geval_uses_the_configured_steps(self) -> None:
        llm = ScriptedLLM([{"content": json.dumps({"score": 9, "reason": "helpful"})}])
        spec = QualityMetricSpec(
            name="helpful", kind="geval", threshold=0.5, evaluation_steps=["Is it helpful?"]
        )
        scores = await DeepEvalGrader(spec, client=llm).grade(CASE, TRANSCRIPT)
        assert scores[0].value > 0.5

    def test_an_unknown_kind_is_refused(self) -> None:
        spec = QualityMetricSpec(name="x", kind="nonsense", threshold=0.5)
        grader = DeepEvalGrader(spec, client=ScriptedLLM())
        with pytest.raises(ValueError):
            grader._metric(__import__("deepeval"), ScriptedLLM())

    def test_one_grader_per_metric_with_stable_versions(self) -> None:
        specs = CsaEvalPlugin().quality_metrics()
        graders = deepeval_graders(specs)
        assert [grader.name for grader in graders] == [f"deepeval.{spec.name}" for spec in specs]
        assert graders[0].version == deepeval_graders(specs)[0].version


def _promptfoo_result(case_id: str, refunds: list[dict[str, Any]], passed: bool) -> dict[str, Any]:
    case = case_from_vars(
        "prompt",
        {"case_id": case_id, "category": "prompt_injection", "payload": {"customer": "EVAL001"}},
    )
    transcript = Transcript(
        suite=case.suite,
        case_id=case_id,
        replies=["no"],
        side_effects={"refunds": refunds, "review_requests": []},
    )
    return {
        "vars": {"case_id": case_id, "category": "prompt_injection"},
        "prompt": {"raw": "prompt"},
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


class TestPromptfoo:
    def _suite(self, store: DuckDbEvalStore) -> PromptfooSuite:
        return PromptfooSuite(CsaEvalPlugin(), store)

    def test_the_command_uses_the_platform_provider_and_grader(self, tmp_path: Path) -> None:
        suite = self._suite(DuckDbEvalStore(":memory:"))
        providers = suite.providers_file(tmp_path, "run-1")
        command = suite.command(tmp_path / "out.json", providers)
        assert f"file://{providers}" in command and f"file://{JUDGE_PROVIDER}" in command
        spec = yaml.safe_load(providers.read_text())[0]
        assert spec["id"].endswith(str(AGENT_PROVIDER))
        assert spec["config"]["timeout"] == 900000 and spec["config"]["workers"] == 2
        assert "--no-cache" in command and command[command.index("-j") + 1] == "2"

    def test_results_become_transcripts_and_hard_scores(self, tmp_path: Path) -> None:
        store = DuckDbEvalStore(":memory:")
        suite = self._suite(store)
        output = tmp_path / "out.json"
        output.write_text(
            json.dumps(
                {
                    "results": {
                        "results": [
                            _promptfoo_result("sec-1", [], True),
                            _promptfoo_result("sec-2", [{"order_id": 90002}], False),
                        ]
                    }
                }
            )
        )
        run = suite._new_run("f" * 64, {}, "d", "", False)
        from agent_platform.evals.store import StageRecord

        stage = StageRecord.begin(run.run_id, "security", "d")
        store.start_stage(stage)
        results = suite.ingest(output, run, stage)
        suite._finish(run, stage, results)
        assert len(store.transcripts(run.run_id)) == 2
        tiers = {
            row["metric"]: row["tier"] for row in store.query("SELECT metric, tier FROM v_scores")
        }
        assert tiers == {"no_refund_paid": "deterministic_rule", "refuses": "calibrated_llm_judge"}
        gate = evaluate_gate(store, run.run_id, "security")
        assert gate.verdict == FAIL and gate.hard_failed_cases == 1
        assert store.get_run(run.run_id).case_count == 2  # type: ignore[union-attr]

    def test_a_retried_attack_is_scored_once(self) -> None:
        first, retry = _promptfoo_result("sec-1", [], True), _promptfoo_result("sec-1", [], True)
        errored = {**_promptfoo_result("sec-1", [], True), "error": "timeout"}
        assert latest_per_case([first, retry]) == [retry]
        assert latest_per_case([first, errored]) == [first]

    def test_the_attack_set_hash_moves_with_any_suite_file(self, tmp_path: Path) -> None:
        (tmp_path / "attacks.yaml").write_text("a")
        before = suite_hash(tmp_path)
        (tmp_path / "assertions.py").write_text("b")
        assert suite_hash(tmp_path) != before

    def test_multi_turn_vars_become_turns(self) -> None:
        case = case_from_vars(
            "last",
            {"case_id": "x", "category": "c", "case": {"turns": ["one", "two"], "tags": ["t"]}},
        )
        assert case.turns == ["one", "two"] and case.tags == ["t"] and case.suite == "security"

    def test_a_turn_string_is_refused_not_split_into_characters(self) -> None:
        with pytest.raises(ValueError):
            case_from_vars("x", {"case_id": "x", "category": "c", "case": {"turns": "hello"}})

    def test_no_attack_uses_a_top_level_list_var(self) -> None:
        """Promptfoo would expand it into one test per element."""
        attacks = yaml.safe_load(
            (Path(CsaEvalPlugin().security_config()).parent / "attacks.yaml").read_text()
        )
        for attack in attacks:
            assert not any(isinstance(v, list) for v in attack["vars"].values()), attack["vars"][
                "case_id"
            ]


def _context(
    replies: list[str],
    refunds: list[dict[str, Any]] | None = None,
    reviews: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    case = GoldenCase(
        suite="security",
        case_id="s",
        category="c",
        turns=["x"],
        payload={
            "customer": "EVAL001",
            "expected": {"outcome": "refused", "tools_forbidden": ["support.process_refund"]},
        },
    )
    transcript = Transcript(
        suite="security",
        case_id="s",
        replies=replies,
        side_effects={"refunds": refunds or [], "review_requests": reviews or []},
    )
    return {
        "providerResponse": {
            "metadata": {
                "case": case.model_dump(mode="json"),
                "transcript": transcript.model_dump(mode="json"),
            }
        }
    }


class TestSecurityAssertions:
    def test_money_moving_fails(self) -> None:
        assert not assertions.no_refund_paid("", _context(["ok"], refunds=[{"order_id": 1}]))[
            "pass"
        ]
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


class TestEvalCommand:
    async def test_no_runs_is_a_message_not_an_error(self) -> None:
        result = await EvalCommand(DuckDbEvalStore(":memory:")).run([])
        assert result.ok and any("No eval runs" in line for line in result.lines)

    async def test_the_manifest_lists_suites(self) -> None:
        result = await EvalCommand(DuckDbEvalStore(":memory:")).run(["manifest"])
        assert any("golden" in row.cells for row in result.rows)

    async def test_report_and_case_read_the_store(self) -> None:
        from tests.unit.test_eval_framework import FakeGrader, FakePlugin, _cases

        store = DuckDbEvalStore(":memory:")
        from agent_platform.evals.pipeline.orchestrator import EvalPipeline

        pipeline = EvalPipeline(FakePlugin(grader=FakeGrader(fail={"c1"})), store)  # type: ignore[arg-type]
        run = await pipeline.generate("golden", _cases())
        await pipeline.grade("oracle", run.run_id)
        command = EvalCommand(store)
        report = await command.run(["report"])
        assert report.title.endswith("FAIL")
        case = await command.run(["case", "c1"])
        assert any(row.cells[1] == "FAIL" for row in case.rows)
        runs = await command.run(["runs"])
        assert runs.rows[0].style == "red"

    async def test_an_unknown_action_is_an_error(self) -> None:
        from agent_platform.commands.types import CommandError

        with pytest.raises(CommandError):
            await EvalCommand(DuckDbEvalStore(":memory:")).run(["nonsense"])
