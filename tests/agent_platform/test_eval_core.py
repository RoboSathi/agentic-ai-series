"""The platform eval core: store, fingerprint, stages, trials, gate, reports, overview, plugin."""

import re
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from agent_platform.config import get_settings
from agent_platform.evals.core.contracts import (
    ConversationFrame,
    GoldenCase,
    GraderKind,
    QualityMetricSpec,
    Score,
    Transcript,
    UsageRecord,
)
from agent_platform.evals.core.fingerprint import compute_fingerprint
from agent_platform.evals.core.pipeline.gate import FAIL, INCOMPLETE, PASS, evaluate_gate
from agent_platform.evals.core.pipeline.generate import GenerationStage
from agent_platform.evals.core.pipeline.grade import GradingStage, grader_fingerprint
from agent_platform.evals.core.pipeline.orchestrator import EvalPipeline, NoRunError
from agent_platform.evals.core.pipeline.trials import base_case_id, expand_trials
from agent_platform.evals.core.plugin import EvalPluginNotConfiguredError, resolve_eval_plugin
from agent_platform.evals.core.reporting import Overview, report_path, wilson_interval
from agent_platform.evals.core.store import DuckDbEvalStore
from agent_platform.evals.core.store.records import new_run_id

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

    def quality_metrics(self) -> list[QualityMetricSpec]:
        return []

    def conversation_frame(self, case: GoldenCase) -> ConversationFrame:
        return ConversationFrame(chatbot_role="a fake")

    def security_config(self) -> str:
        return ""

    def fingerprint_inputs(self) -> dict[str, str]:
        return {"model": "fake-model"}


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


class TestRunId:
    def test_it_names_the_agent_suite_and_build_and_ends_with_the_date_and_time(self) -> None:
        run_id = new_run_id("csa", "golden", "0123456789abcdef")
        assert re.fullmatch(r"csa-golden-01234567-\d{8}-\d{6}", run_id)


class TestFingerprint:
    def test_an_uncommitted_edit_changes_it_and_reverting_restores_it(self, tmp_path: Path) -> None:
        source = tmp_path / "src"
        source.mkdir()
        module = source / "agent.py"
        module.write_text("x = 1\n")
        settings = get_settings().evals.fingerprint.model_copy(
            update={"source_roots": ["src"], "exclude_paths": [], "agent_exclude_paths": []}
        )
        before = compute_fingerprint({"model": "m"}, settings, root=tmp_path)
        module.write_text("x = 2\n")
        edited = compute_fingerprint({"model": "m"}, settings, root=tmp_path)
        module.write_text("x = 1\n")
        reverted = compute_fingerprint({"model": "m"}, settings, root=tmp_path)
        assert before.value != edited.value
        assert before.value == reverted.value

    def test_platform_and_agent_exclusions_both_count(self, tmp_path: Path) -> None:
        for folder in ("reports", "evals"):
            (tmp_path / "src" / folder).mkdir(parents=True)
            (tmp_path / "src" / folder / "r.py").write_text("a")
        settings = get_settings().evals.fingerprint.model_copy(
            update={
                "source_roots": ["src"],
                "exclude_paths": ["src/reports"],
                "agent_exclude_paths": ["src/evals"],
            }
        )
        before = compute_fingerprint({}, settings, root=tmp_path)
        (tmp_path / "src" / "reports" / "r.py").write_text("b")
        (tmp_path / "src" / "evals" / "r.py").write_text("b")
        assert compute_fingerprint({}, settings, root=tmp_path).value == before.value

    def test_agent_inputs_are_parts(self) -> None:
        first = compute_fingerprint({"model": "a"})
        second = compute_fingerprint({"model": "b"})
        assert second.value != first.value


class TestGeneration:
    async def test_every_case_is_generated_and_stored(self, store: DuckDbEvalStore) -> None:
        run = await GenerationStage(FakePlugin(), store).run(SUITE, _cases())
        assert len(store.transcripts(run.run_id)) == 3
        assert run.status == "completed"

    async def test_an_unchanged_agent_and_dataset_reuse_the_run(
        self, store: DuckDbEvalStore
    ) -> None:
        plugin = FakePlugin()
        first = await GenerationStage(plugin, store).run(SUITE, _cases())
        second = await GenerationStage(plugin, store).run(SUITE, _cases())
        assert first.run_id == second.run_id
        assert plugin.agent.calls == ["c0", "c1", "c2"]

    async def test_a_changed_case_regenerates_only_that_case(self, store: DuckDbEvalStore) -> None:
        plugin = FakePlugin()
        await GenerationStage(plugin, store).run(SUITE, _cases())
        edited = _cases()
        edited[1] = edited[1].model_copy(update={"turns": ["reworded"]})
        run = await GenerationStage(plugin, store).run(SUITE, edited)
        assert plugin.agent.calls[3:] == ["c1"]
        reused = [item.case_id for item in store.transcripts(run.run_id) if item.reused_from]
        assert reused == ["c0", "c2"]

    async def test_a_raising_case_is_an_errored_transcript_not_a_crash(
        self, store: DuckDbEvalStore
    ) -> None:
        plugin = FakePlugin(agent=FakeAgent(fail={"c1"}))
        run = await GenerationStage(plugin, store).run(SUITE, _cases())
        errored = [item for item in store.transcripts(run.run_id) if item.transcript.error]
        assert [item.case_id for item in errored] == ["c1"]
        assert run.status == "errored"

    async def test_an_errored_transcript_is_retried_not_reused(
        self, store: DuckDbEvalStore
    ) -> None:
        plugin = FakePlugin(agent=FakeAgent(fail={"c1"}))
        await GenerationStage(plugin, store).run(SUITE, _cases())
        plugin.agent._fail = set()
        run = await GenerationStage(plugin, store).run(SUITE, _cases())
        assert plugin.agent.calls.count("c1") == 2
        assert not any(item.transcript.error for item in store.transcripts(run.run_id))


class TestTrials:
    def test_one_trial_leaves_the_cases_alone(self) -> None:
        assert expand_trials(_cases(), 1) == _cases()

    def test_k_trials_are_k_distinct_cases_that_fold_back(self) -> None:
        trials = expand_trials(_cases(1), 3)
        assert len({case.case_hash for case in trials}) == 3
        assert {base_case_id(case.case_id) for case in trials} == {"c0"}


class TestGrading:
    async def _run(self, store: DuckDbEvalStore, plugin: FakePlugin) -> str:
        return (await GenerationStage(plugin, store).run(SUITE, _cases())).run_id

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
        run = await GenerationStage(plugin, store).run(SUITE, _cases())
        await GradingStage(store).run(
            run.run_id, "oracle", [grader], store.cases_for_run(run.run_id)
        )
        return run.run_id

    def test_no_scores_is_incomplete(self, store: DuckDbEvalStore) -> None:
        assert evaluate_gate(store, "nothing").verdict == INCOMPLETE

    async def test_one_hard_failure_fails_the_run(self, store: DuckDbEvalStore) -> None:
        gate = evaluate_gate(store, await self._graded(store, FakeGrader(fail={"c1"})))
        assert gate.verdict == FAIL and gate.hard_failed_cases == 1

    async def test_each_stage_has_its_own_verdict(self, store: DuckDbEvalStore) -> None:
        run_id = await self._graded(store, FakeGrader(fail={"c1"}))
        assert evaluate_gate(store, run_id, "oracle").verdict == FAIL
        assert evaluate_gate(store, run_id, "deepeval").verdict == INCOMPLETE

    async def test_soft_failures_gate_on_the_configured_pass_rate(
        self, store: DuckDbEvalStore
    ) -> None:
        grader = FakeGrader(severity="soft", fail={"c1"})
        gate = evaluate_gate(store, await self._graded(store, grader))
        assert gate.soft_pass_rate == pytest.approx(2 / 3)
        assert gate.soft_threshold == get_settings().evals.gate.soft_pass_rate
        assert gate.verdict == (PASS if 2 / 3 >= gate.soft_threshold else FAIL)


def _with_judge(
    plugin: FakePlugin, monkeypatch: pytest.MonkeyPatch, judge: FakeGrader | None = None
) -> FakeGrader:
    """Give the plugin one core and one non-core metric, judged by a scripted grader."""
    plugin.quality_metrics = lambda: [  # type: ignore[method-assign]
        QualityMetricSpec(name="core_m", kind="turn_relevancy", threshold=0.5, core=True),
        QualityMetricSpec(name="extra_m", kind="turn_relevancy", threshold=0.5),
    ]
    chosen = judge or FakeGrader(name="quality", kind="deepeval", severity="soft")
    seen: list[list[str]] = []

    def build(self: EvalPipeline, metrics: list[QualityMetricSpec]) -> FakeGrader:
        seen.append([metric.name for metric in metrics])
        return chosen

    monkeypatch.setattr(EvalPipeline, "_deepeval", build)
    chosen.metrics_seen = seen  # type: ignore[attr-defined]
    return chosen


class TestGoldenPipeline:
    async def test_each_section_writes_its_own_report_named_for_the_run(
        self, store: DuckDbEvalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plugin = FakePlugin(grader=FakeGrader(fail={"c2"}))
        _with_judge(plugin, monkeypatch)
        oracle, deepeval = await EvalPipeline(plugin, store).golden()  # type: ignore[arg-type]
        assert (oracle.section, deepeval.section) == ("oracle", "deepeval")
        assert oracle.run_id == deepeval.run_id
        assert oracle.report == report_path("oracle", oracle.run_id)
        assert deepeval.report == report_path("deepeval", deepeval.run_id)
        body = oracle.report.read_text()
        assert "Gate: FAIL" in body and "### c2" in body
        assert oracle.failed and not deepeval.failed

    async def test_the_oracle_report_lists_every_case_with_why_it_failed(
        self, store: DuckDbEvalStore
    ) -> None:
        (oracle,) = await EvalPipeline(FakePlugin(grader=FakeGrader(fail={"c1"})), store).golden()  # type: ignore[arg-type]
        cases = oracle.report.read_text().split("## Every case")[1].split("##")[0]
        states = get_settings().evals.reporting.overview.states
        assert "| c0 | other | " + states["pass"] + " | 1/1 |" in cases
        assert "| c1 | cat | " + states["fail"] + " | 0/1 | **fake_metric**: broke |" in cases
        assert "| c2 |" in cases

    async def test_one_section_runs_alone(
        self, store: DuckDbEvalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plugin = FakePlugin()
        judge = _with_judge(plugin, monkeypatch)
        (outcome,) = await EvalPipeline(plugin, store).golden(["oracle"])  # type: ignore[arg-type]
        stages = store.query(
            "SELECT DISTINCT stage FROM eval_stages WHERE run_id = ?", (outcome.run_id,)
        )
        assert {row["stage"] for row in stages} == {"generate", "oracle"}
        assert judge.graded == []
        (judged,) = await EvalPipeline(plugin, store).golden(["deepeval"])  # type: ignore[arg-type]
        assert judged.run_id == outcome.run_id and plugin.agent.calls == ["c0", "c1", "c2"]

    async def test_a_broken_invariant_never_skips_the_judge(
        self, store: DuckDbEvalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plugin = FakePlugin(grader=FakeGrader(fail={"c0"}))
        judge = _with_judge(plugin, monkeypatch)
        pipeline = EvalPipeline(plugin, store)  # type: ignore[arg-type]
        await pipeline.golden()
        assert judge.graded == ["c0", "c1", "c2"]
        assert list(pipeline.clock.elapsed) == ["agent", "oracle", "deepeval"]

    async def test_a_section_that_raises_is_reported_and_the_next_still_runs(
        self, store: DuckDbEvalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plugin = FakePlugin(grader=FakeGrader(raises=True))
        _with_judge(plugin, monkeypatch)
        seen: list[tuple[str, str]] = []

        def sink(section: str, result: object) -> None:
            seen.append((section, type(result).__name__))

        await EvalPipeline(plugin, store).golden(sink=sink)  # type: ignore[arg-type]
        assert [section for section, _ in seen] == ["oracle", "deepeval"]

    async def test_core_is_the_default_and_full_adds_the_rest(
        self, store: DuckDbEvalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tag = get_settings().evals.core.tag
        cases = [c.model_copy(update={"tags": [tag]}) if c.case_id != "c2" else c for c in _cases()]
        plugin = FakePlugin()
        plugin.load_cases = lambda suite: cases  # type: ignore[method-assign]
        judge = _with_judge(plugin, monkeypatch)
        await EvalPipeline(plugin, store).golden()  # type: ignore[arg-type]
        assert plugin.agent.calls == ["c0", "c1"]
        await EvalPipeline(plugin, store).golden(full=True)  # type: ignore[arg-type]
        assert plugin.agent.calls == ["c0", "c1", "c2"]  # the core transcripts are reused
        assert judge.metrics_seen == [["core_m"], ["core_m", "extra_m"]]  # type: ignore[attr-defined]

    async def test_an_agent_with_no_core_runs_everything(self, store: DuckDbEvalStore) -> None:
        plugin = FakePlugin()
        await EvalPipeline(plugin, store).golden()  # type: ignore[arg-type]
        assert plugin.agent.calls == ["c0", "c1", "c2"]

    async def test_the_overview_is_a_coloured_table_with_the_gate(
        self, store: DuckDbEvalStore
    ) -> None:
        (outcome,) = await EvalPipeline(FakePlugin(grader=FakeGrader(fail={"c1"})), store).golden()  # type: ignore[arg-type]
        console = Console(record=True, width=160)
        outcome.overview.print(console)
        text = console.export_text()
        states = get_settings().evals.reporting.overview.states
        assert "fake_metric" in text and states["fail"] in text and "c1" in text
        assert str(outcome.report) in text

    async def test_reports_are_rebuilt_from_the_store(
        self, store: DuckDbEvalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plugin = FakePlugin()
        _with_judge(plugin, monkeypatch)
        pipeline = EvalPipeline(plugin, store)  # type: ignore[arg-type]
        first = await pipeline.golden()
        for outcome in first:
            outcome.report.unlink()
        rebuilt = pipeline.report(first[0].run_id)
        assert [outcome.section for outcome in rebuilt] == ["oracle", "deepeval"]
        assert all(outcome.report.exists() for outcome in rebuilt)

    def test_no_run_is_a_readable_error(self, store: DuckDbEvalStore) -> None:
        with pytest.raises(NoRunError):
            EvalPipeline(FakePlugin(), store).report("missing")  # type: ignore[arg-type]


class BakeoffPlugin(FakePlugin):
    async def intent_bakeoff(self) -> object:
        from agent_platform.evals.intent import BakeoffResult, LabelledMessage, predict, score

        async def classify(text: str) -> tuple[str, float]:
            return "REFUND", 0.9

        predictions = await predict([LabelledMessage(text="refund", intent="REFUND")], classify)
        return BakeoffResult(
            title="bake-off", scores={"m · model": score(predictions, 0.5, 10)}, misses={}
        )


class TestOtherStages:
    async def test_the_intent_stage_writes_its_report_and_overview(
        self, store: DuckDbEvalStore
    ) -> None:
        outcome = await EvalPipeline(BakeoffPlugin(), store).intent()  # type: ignore[arg-type]
        assert outcome.report.name == f"intent-{outcome.run_id}.md"
        assert "m · model" in outcome.report.read_text()
        assert outcome.overview.result.tables[0].rows[0][1] == "1/1"

    async def test_trials_report_pass_at_k_and_pass_hat_k(
        self, store: DuckDbEvalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(get_settings().evals, "trials", 2)
        separator = get_settings().evals.trial_separator
        flaky = FakeGrader(fail={f"c1{separator}2"})
        (outcome,) = await EvalPipeline(FakePlugin(grader=flaky), store).golden()  # type: ignore[arg-type]
        body = outcome.report.read_text()
        assert "| 2 | 3 | 3/3 | 2/3 |" in body  # c1 passed once: pass@2 yes, pass^2 no

    async def test_the_judge_is_measured_against_the_runtime_guard(
        self, store: DuckDbEvalStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        guard = get_settings().evals.reporting.guard_metric
        grader = FakeGrader(name="g", kind="deepeval", severity="soft")

        async def grade(case: GoldenCase, transcript: Transcript) -> list[Score]:
            scores = await FakeGrader.grade(grader, case, transcript)
            return [score.model_copy(update={"metric": guard}) for score in scores]

        grader.grade = grade  # type: ignore[method-assign]
        plugin = FakePlugin()
        _with_judge(plugin, monkeypatch, grader)
        (outcome,) = await EvalPipeline(plugin, store).golden(["deepeval"])  # type: ignore[arg-type]
        assert "Cohen's κ" in outcome.report.read_text()


class TestOverview:
    def test_metric_states_follow_kind_and_rate(self) -> None:
        overview = Overview("t")
        rows: list[dict[str, Any]] = [
            {"metric": "hard_ok", "severity": "hard", "passed": 3, "cases": 3, "threshold": 1.0},
            {"metric": "hard_bad", "severity": "hard", "passed": 2, "cases": 3, "threshold": 1.0},
            {"metric": "soft_low", "severity": "soft", "passed": 1, "cases": 3, "threshold": 0.7},
        ]
        table = overview.metrics("m", rows).result.tables[0]
        states = get_settings().evals.reporting.overview.states
        assert [row[-1] for row in table.rows] == [states["pass"], states["fail"], states["warn"]]

    def test_only_the_first_failures_are_listed(self, tmp_path: Path) -> None:
        cap = get_settings().evals.reporting.overview.max_failures
        rows = [
            {"case_id": f"c{i}", "category": "x", "metric": "m", "rationale": "why"}
            for i in range(cap + 2)
        ]
        result = Overview("t").failures(rows, tmp_path / "r.md").result
        assert len(result.tables[0].rows) == cap and "+2 more" in result.lines[0]

    def test_wilson_stays_inside_zero_and_one(self) -> None:
        low, high = wilson_interval(50, 50, 1.96)
        assert 0.9 < low < 1.0 and high == 1.0
        assert wilson_interval(0, 0, 1.96) == (0.0, 0.0)


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
