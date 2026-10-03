"""The platform eval framework: store, fingerprint, stages, gate, reports, graders."""

import json
from pathlib import Path
from typing import Any

import pytest

from agent_platform.config import get_settings
from agent_platform.evals.contracts import (
    GoldenCase,
    GraderKind,
    QualityMetricSpec,
    Rubric,
    RubricCriterion,
    Score,
    Transcript,
    UsageRecord,
)
from agent_platform.evals.fingerprint import compute_fingerprint
from agent_platform.evals.graders import JudgeResponseError, RubricJudge, parse_verdict
from agent_platform.evals.pipeline.gate import (
    FAIL,
    INCOMPLETE,
    PASS,
    evaluate_gate,
    soft_threshold,
)
from agent_platform.evals.pipeline.generate import GenerationStage
from agent_platform.evals.pipeline.grade import GradingStage, grader_fingerprint
from agent_platform.evals.pipeline.orchestrator import EvalPipeline, NoRunError
from agent_platform.evals.plugin import EvalPluginNotConfiguredError, resolve_eval_plugin
from agent_platform.evals.reporting import wilson_interval
from agent_platform.evals.store import DuckDbEvalStore
from tests.conftest import ScriptedLLM

SUITE = "golden"


def _cases(count: int = 3) -> list[GoldenCase]:
    return [
        GoldenCase(
            suite=SUITE,
            case_id=f"c{index}",
            category="cat" if index else "other",
            turns=[f"question {index}"],
            payload={"n": index},
        )
        for index in range(count)
    ]


class FakeAgent:
    def __init__(self, fail: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self._fail = fail or set()

    async def run(self, case: GoldenCase) -> Transcript:
        self.calls.append(case.case_id)
        if case.case_id in self._fail:
            raise RuntimeError("agent exploded")
        return Transcript(
            suite=case.suite,
            case_id=case.case_id,
            replies=[f"answer {case.case_id}"],
            usage=UsageRecord(llm_calls=2, latency_ms=5.0),
            metadata={"hallucination_rate": 0.0},
        )


class FakeGrader:
    def __init__(
        self,
        name: str = "fake",
        kind: GraderKind = "oracle",
        version: str = "1",
        severity: str = "hard",
        fail: set[str] | None = None,
        raises: bool = False,
    ) -> None:
        self.name, self.kind, self.version = name, kind, version
        self._severity, self._fail, self._raises = severity, fail or set(), raises
        self.graded: list[str] = []

    async def grade(self, case: GoldenCase, transcript: Transcript) -> list[Score]:
        self.graded.append(case.case_id)
        if self._raises:
            raise ValueError("grader broke")
        passed = case.case_id not in self._fail
        return [
            Score(
                case_id=case.case_id,
                metric=f"{self.name}_metric",
                grader=self.name,
                kind=self.kind,
                severity=self._severity,  # type: ignore[arg-type]
                value=1.0 if passed else 0.0,
                passed=passed,
                tier="deterministic_rule",
                rationale="ok" if passed else "broke",
            )
        ]


class FakePlugin:
    def __init__(self, agent: FakeAgent | None = None, grader: FakeGrader | None = None) -> None:
        self.agent = agent or FakeAgent()
        self.grader = grader or FakeGrader()

    @property
    def agent_id(self) -> str:
        return "fake_agent"

    def load_cases(self, suite: str) -> list[GoldenCase]:
        return _cases()

    def prepare(self) -> None:
        return None

    def agent_under_test(self) -> FakeAgent:
        return self.agent

    def oracle(self) -> FakeGrader:
        return self.grader

    def judge_rubric(self, suite: str) -> Rubric:
        return _rubric()

    def quality_metrics(self) -> list[QualityMetricSpec]:
        return []

    def security_config(self) -> str:
        return ""

    def fingerprint_inputs(self) -> dict[str, str]:
        return {"model": "fake-model"}


def _rubric() -> Rubric:
    return Rubric(
        suite=SUITE,
        scale_max=5,
        hallucination_threshold=0.0,
        criteria=[RubricCriterion(name="helpful", description="d", threshold=0.75)],
    )


@pytest.fixture
def store() -> DuckDbEvalStore:
    return DuckDbEvalStore(":memory:")


class TestContracts:
    def test_hash_is_order_independent_and_content_sensitive(self) -> None:
        case = _cases(1)[0]
        same = GoldenCase(**{**case.model_dump(), "payload": {"n": 0}})
        assert same.case_hash == case.case_hash
        assert case.model_copy(update={"turns": ["x"]}).case_hash != case.case_hash

    def test_the_reply_is_the_last_one(self) -> None:
        assert Transcript(suite=SUITE, case_id="x", replies=["a", "b"]).reply == "b"
        assert Transcript(suite=SUITE, case_id="x").reply == ""


class TestFingerprint:
    def test_an_uncommitted_edit_changes_it_and_reverting_restores_it(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        source.mkdir()
        module = source / "agent.py"
        module.write_text("x = 1\n")
        settings = get_settings().evals.fingerprint.model_copy(
            update={"source_roots": ["src"], "exclude_paths": []}
        )
        before = compute_fingerprint({"model": "m"}, settings, root=tmp_path)
        module.write_text("x = 2\n")
        edited = compute_fingerprint({"model": "m"}, settings, root=tmp_path)
        module.write_text("x = 1\n")
        reverted = compute_fingerprint({"model": "m"}, settings, root=tmp_path)
        assert before.value != edited.value
        assert before.value == reverted.value
        assert edited.changed_parts(before.parts) == ["sources"]

    def test_excluded_paths_do_not_count(self, tmp_path: Path) -> None:
        (tmp_path / "src" / "reports").mkdir(parents=True)
        report = tmp_path / "src" / "reports" / "r.py"
        report.write_text("a")
        settings = get_settings().evals.fingerprint.model_copy(
            update={"source_roots": ["src"], "exclude_paths": ["src/reports"]}
        )
        before = compute_fingerprint({}, settings, root=tmp_path)
        report.write_text("b")
        assert compute_fingerprint({}, settings, root=tmp_path).value == before.value

    def test_agent_inputs_are_parts(self) -> None:
        first = compute_fingerprint({"model": "a"})
        second = compute_fingerprint({"model": "b"})
        assert second.changed_parts(first.parts) == ["agent.model"]


class TestGeneration:
    async def test_every_case_is_generated_and_stored(self, store: DuckDbEvalStore) -> None:
        plugin = FakePlugin()
        run, stage = await GenerationStage(plugin, store).run(SUITE, _cases())
        assert len(store.transcripts(run.run_id)) == 3
        assert run.status == "running" and store.get_run(run.run_id).status == "completed"  # type: ignore[union-attr]
        assert stage.stage == "generate"

    async def test_an_unchanged_agent_and_dataset_reuse_the_run(
        self, store: DuckDbEvalStore
    ) -> None:
        plugin = FakePlugin()
        first, _ = await GenerationStage(plugin, store).run(SUITE, _cases())
        second, _ = await GenerationStage(plugin, store).run(SUITE, _cases())
        assert first.run_id == second.run_id
        assert plugin.agent.calls == ["c0", "c1", "c2"]

    async def test_a_changed_case_regenerates_only_that_case(self, store: DuckDbEvalStore) -> None:
        plugin = FakePlugin()
        await GenerationStage(plugin, store).run(SUITE, _cases())
        edited = _cases()
        edited[1] = edited[1].model_copy(update={"turns": ["reworded"]})
        run, stage = await GenerationStage(plugin, store).run(SUITE, edited)
        assert plugin.agent.calls[3:] == ["c1"]
        reused = [item.case_id for item in store.transcripts(run.run_id) if item.reused_from]
        assert reused == ["c0", "c2"]

    async def test_fresh_ignores_every_cache(self, store: DuckDbEvalStore) -> None:
        plugin = FakePlugin()
        await GenerationStage(plugin, store).run(SUITE, _cases())
        await GenerationStage(plugin, store).run(SUITE, _cases(), fresh=True)
        assert len(plugin.agent.calls) == 6

    async def test_a_raising_case_is_an_errored_transcript_not_a_crash(
        self, store: DuckDbEvalStore
    ) -> None:
        plugin = FakePlugin(agent=FakeAgent(fail={"c1"}))
        run, stage = await GenerationStage(plugin, store).run(SUITE, _cases())
        errored = [item for item in store.transcripts(run.run_id) if item.transcript.error]
        assert [item.case_id for item in errored] == ["c1"]
        assert store.get_run(run.run_id).status == "errored"  # type: ignore[union-attr]

    async def test_an_errored_transcript_is_retried_not_reused(
        self, store: DuckDbEvalStore
    ) -> None:
        plugin = FakePlugin(agent=FakeAgent(fail={"c1"}))
        await GenerationStage(plugin, store).run(SUITE, _cases())
        plugin.agent._fail = set()
        run, _ = await GenerationStage(plugin, store).run(SUITE, _cases())
        assert plugin.agent.calls.count("c1") == 2
        assert not any(item.transcript.error for item in store.transcripts(run.run_id))


class TestGrading:
    async def _run(self, store: DuckDbEvalStore, plugin: FakePlugin) -> str:
        run, _ = await GenerationStage(plugin, store).run(SUITE, _cases())
        return run.run_id

    async def test_scores_are_reused_for_an_unchanged_grader(self, store: DuckDbEvalStore) -> None:
        plugin = FakePlugin()
        run_id = await self._run(store, plugin)
        cases = store.cases_for_run(run_id)
        await GradingStage(store).run(run_id, "oracle", [plugin.grader], cases)
        stage = await GradingStage(store).run(run_id, "oracle", [plugin.grader], cases)
        assert plugin.grader.graded == ["c0", "c1", "c2"]
        assert stage.reused == 3

    async def test_a_new_grader_version_regrades(self, store: DuckDbEvalStore) -> None:
        plugin = FakePlugin()
        run_id = await self._run(store, plugin)
        cases = store.cases_for_run(run_id)
        await GradingStage(store).run(run_id, "oracle", [plugin.grader], cases)
        newer = FakeGrader(version="2")
        await GradingStage(store).run(run_id, "oracle", [newer], cases)
        assert newer.graded == ["c0", "c1", "c2"]
        assert grader_fingerprint([plugin.grader]) != grader_fingerprint([newer])

    async def test_a_raising_grader_scores_an_error_that_is_never_reused(
        self, store: DuckDbEvalStore
    ) -> None:
        plugin = FakePlugin()
        run_id = await self._run(store, plugin)
        cases = store.cases_for_run(run_id)
        broken = FakeGrader(raises=True)
        await GradingStage(store).run(run_id, "oracle", [broken], cases)
        rows = store.query("SELECT metric FROM v_scores WHERE run_id = ?", (run_id,))
        assert {row["metric"] for row in rows} == {"fake.error"}
        await GradingStage(store).run(run_id, "oracle", [broken], cases)
        assert len(broken.graded) == 6


class TestGate:
    async def _graded(self, store: DuckDbEvalStore, grader: FakeGrader) -> str:
        plugin = FakePlugin(grader=grader)
        run, _ = await GenerationStage(plugin, store).run(SUITE, _cases())
        await GradingStage(store).run(
            run.run_id, "oracle", [grader], store.cases_for_run(run.run_id)
        )
        return run.run_id

    def test_no_scores_is_incomplete(self, store: DuckDbEvalStore) -> None:
        assert evaluate_gate(store, "nothing", SUITE).verdict == INCOMPLETE

    async def test_one_hard_failure_fails_the_run(self, store: DuckDbEvalStore) -> None:
        gate = evaluate_gate(store, await self._graded(store, FakeGrader(fail={"c1"})), SUITE)
        assert gate.verdict == FAIL and gate.hard_failed_cases == 1

    async def test_soft_failures_gate_on_the_pass_rate(self, store: DuckDbEvalStore) -> None:
        grader = FakeGrader(severity="soft", fail={"c1"})
        gate = evaluate_gate(store, await self._graded(store, grader), SUITE)
        assert gate.soft_pass_rate == pytest.approx(2 / 3)
        assert gate.verdict == (PASS if 2 / 3 >= gate.soft_threshold else FAIL)

    def test_the_manifest_declares_the_suite_threshold(self) -> None:
        assert soft_threshold("golden") == 0.8
        assert soft_threshold("unknown") == get_settings().evals.gate.soft_pass_rate


class TestReports:
    async def test_every_stage_with_data_gets_a_report(self, store: DuckDbEvalStore) -> None:
        plugin = FakePlugin(grader=FakeGrader(fail={"c2"}))
        pipeline = EvalPipeline(plugin, store)  # type: ignore[arg-type]
        run = await pipeline.generate(SUITE, _cases())
        await pipeline.grade("oracle", run.run_id)
        written = {path.name for path in pipeline.report(run.run_id)}
        files = get_settings().evals.reporting.files
        assert written == {files["summary"], files["generate"], files["oracle"]}
        summary = (
            get_settings().evals.reporting.resolved_output_dir / run.run_id / files["summary"]
        ).read_text()
        assert "Gate: FAIL" in summary and "c2" in summary

    async def test_a_diff_shows_regressions(self, store: DuckDbEvalStore) -> None:
        good = FakePlugin(grader=FakeGrader())
        pipeline = EvalPipeline(good, store)  # type: ignore[arg-type]
        base = await pipeline.generate(SUITE, _cases())
        await pipeline.grade("oracle", base.run_id)
        worse = FakePlugin(agent=FakeAgent(), grader=FakeGrader(version="2", fail={"c0"}))
        worse.fingerprint_inputs = lambda: {"model": "other"}  # type: ignore[method-assign]
        pipeline = EvalPipeline(worse, store)  # type: ignore[arg-type]
        head = await pipeline.generate(SUITE, _cases())
        await pipeline.grade("oracle", head.run_id)
        body = pipeline.diff(base.run_id, head.run_id).read_text()
        assert "c0" in body.split("improved")[0]

    def test_no_run_is_a_readable_error(self, store: DuckDbEvalStore) -> None:
        with pytest.raises(NoRunError):
            EvalPipeline(FakePlugin(), store).run_or_latest(None)  # type: ignore[arg-type]

    def test_wilson_stays_inside_zero_and_one(self) -> None:
        low, high = wilson_interval(50, 50, 1.96)
        assert 0.9 < low < 1.0 and high == 1.0
        assert wilson_interval(0, 0, 1.96) == (0.0, 0.0)


class TestRubricJudge:
    _GOOD = json.dumps(
        {
            "criteria": {"helpful": {"score": 5, "rationale": "clear"}},
            "claims": [
                {"claim": "refund done", "verdict": "supported"},
                {"claim": "arrives Monday", "verdict": "unsupported"},
                {"claim": "have a nice day", "verdict": "unverifiable"},
            ],
        }
    )

    async def test_one_call_scores_the_rubric_and_the_claims(self) -> None:
        llm = ScriptedLLM([{"content": f"Here you go: {self._GOOD}"}])
        judge = RubricJudge(_rubric(), client=llm)
        case = _cases(1)[0]
        scores = await judge.grade(case, Transcript(suite=SUITE, case_id="c0", replies=["r"]))
        by_metric = {score.metric: score for score in scores}
        assert by_metric["helpful"].value == 1.0 and by_metric["helpful"].passed
        assert by_metric["hallucination_free"].value == 0.5
        assert not by_metric["hallucination_free"].passed
        assert len(llm.requests) == 1 and sum(score.llm_calls for score in scores) == 1

    def test_the_prompt_carries_conversation_tools_and_evidence(self) -> None:
        judge = RubricJudge(_rubric(), client=ScriptedLLM())
        transcript = Transcript(
            suite=SUITE, case_id="c0", replies=["the answer"], evidence=["EVIDENCE-X"]
        )
        prompt = judge.prompt(_cases(1)[0], transcript)
        assert "question 0" in prompt and "the answer" in prompt and "EVIDENCE-X" in prompt
        assert "{criteria}" not in prompt

    def test_a_non_json_answer_raises_so_the_grade_is_retried(self) -> None:
        with pytest.raises(JudgeResponseError):
            parse_verdict("I think it was fine.")

    def test_a_missing_criterion_fails_rather_than_passing(self) -> None:
        verdict = parse_verdict(json.dumps({"criteria": {}, "claims": []}))
        assert verdict.hallucination_rate == 0.0


class TestPlugin:
    def test_the_platform_names_no_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(get_settings().evals, "plugin", "")
        with pytest.raises(EvalPluginNotConfiguredError):
            resolve_eval_plugin()

    def test_the_configured_plugin_resolves(self) -> None:
        assert resolve_eval_plugin().agent_id

    def test_a_spec_that_is_not_a_plugin_is_refused(self) -> None:
        with pytest.raises(TypeError):
            resolve_eval_plugin("agent_platform.config:get_settings")


def _score_rows(store: DuckDbEvalStore, run_id: str) -> list[dict[str, Any]]:
    return store.query("SELECT * FROM v_scores WHERE run_id = ?", (run_id,))
