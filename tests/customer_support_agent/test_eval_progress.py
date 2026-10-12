"""What `make eval` shows while it runs: progress per section, batches, Ctrl+C, 429s by model."""

import json
from pathlib import Path

import httpx
import pytest
from rich.console import Console

from agent_platform.evals.core import cli
from agent_platform.evals.core.contracts import GoldenCase, Score, Transcript
from agent_platform.evals.core.pipeline.grade import GradingStage
from agent_platform.evals.core.pipeline.orchestrator import EvalPipeline, StageOutcome
from agent_platform.evals.core.reporting.progress_bar import ProgressBar
from agent_platform.evals.core.store import DuckDbEvalStore
from agent_platform.evals.promptfoo import agent_provider
from agent_platform.evals.promptfoo.runner import attack_count
from agent_platform.resilience.rate_limiter.transport import model_of
from customer_support_agent.evals.plugin import CsaEvalPlugin
from tests.agent_platform.test_eval_core import FakeGrader, FakePlugin


class Recorder:
    """An `EvalProgress` that remembers what it was told."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, int | None]] = []

    def start(self, section: str, total: int | None) -> None:
        self.events.append(("start", section, total))

    def advance(self, section: str, completed: int) -> None:
        self.events.append(("advance", section, completed))

    def finish(self, section: str) -> None:
        self.events.append(("finish", section, None))


class BatchFake(FakeGrader):
    """A grader that takes transcripts a batch at a time."""

    batch_size = 2

    def __init__(self) -> None:
        super().__init__(name="batch", kind="deepeval", severity="soft")
        self.batches: list[int] = []

    async def grade_batch(self, items: list[tuple[GoldenCase, Transcript]]) -> list[list[Score]]:
        self.batches.append(len(items))
        return [await self.grade(case, transcript) for case, transcript in items]


class TestSectionsReportProgress:
    async def test_every_section_starts_advances_to_its_total_and_finishes(self) -> None:
        recorder = Recorder()
        await EvalPipeline(FakePlugin(), DuckDbEvalStore(":memory:"), progress=recorder).golden()  # type: ignore[arg-type]
        for section in ("agent", "oracle"):
            ours = [event for event in recorder.events if event[1] == section]
            assert ours[0] == ("start", section, 3)
            assert ("advance", section, 3) in ours and ours[-1] == ("finish", section, None)

    async def test_a_batch_grader_is_handed_batches_and_reports_each(self) -> None:
        store = DuckDbEvalStore(":memory:")
        run = (await EvalPipeline(FakePlugin(), store).golden())[0].run_id
        recorder, grader = Recorder(), BatchFake()
        await GradingStage(store, recorder).run(run, "deepeval", [grader], store.cases_for_run(run))
        assert grader.batches == [2, 1]
        assert [e[2] for e in recorder.events if e[0] == "advance"] == [0, 2, 3]

    def test_the_bar_draws_each_section_with_its_count(self) -> None:
        console = Console(record=True, width=120)
        bar = ProgressBar(console)
        with bar:
            bar.start("agent", 4)
            bar.advance("agent", 2)
            bar.start("intent", None)
            bar.finish("intent")
        text = console.export_text()
        assert "2/4" in text and "agent runs" in text


class TestPromptfooProgress:
    def test_the_attack_count_is_read_from_the_config(self) -> None:
        config = Path(CsaEvalPlugin().security_config())
        listed = (config.parent / "attacks.yaml").read_text().splitlines()
        assert attack_count(config) == sum(line.startswith("- description:") for line in listed)

    def test_each_finished_attack_ticks_the_progress_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agent_platform.config import get_settings

        ticks = tmp_path / "run.progress"
        monkeypatch.setenv(get_settings().evals.promptfoo.progress_env, str(ticks))
        monkeypatch.setattr(agent_provider, "_plugin", FakePlugin())
        agent_provider.call_api("x", {}, {"vars": {"case_id": "c0"}})
        agent_provider.call_api("x", {}, {"vars": {"case_id": "c1"}})
        assert ticks.read_text().splitlines() == ["c0", "c1"]


class TestCtrlC:
    def test_ctrl_c_marks_the_run_interrupted_and_exits_130(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        plugin = FakePlugin()
        monkeypatch.setattr(cli, "resolve_eval_plugin", lambda: plugin)
        store = DuckDbEvalStore()
        store.execute("DELETE FROM eval_runs")

        async def stopped(self: EvalPipeline, *args: object) -> list[StageOutcome]:
            from agent_platform.evals.core.pipeline.generate import GenerationStage

            with self.clock.measure("agent"):
                await GenerationStage(plugin, self.store).run("golden", plugin.load_cases("golden"))  # type: ignore[arg-type]
                self.store.execute("UPDATE eval_runs SET status = 'running'")
                raise KeyboardInterrupt

        monkeypatch.setattr(EvalPipeline, "golden", stopped)
        assert cli.main(["run", "--section", "deepeval", "--invocation", "make eval csa"]) == 130
        statuses = {row["status"] for row in store.query("SELECT status FROM eval_runs")}
        assert statuses == {"interrupted"}
        out = capsys.readouterr().out
        assert "`make eval csa` to resume" in out
        assert "stopped" in out

    async def test_another_processes_run_is_left_running(self) -> None:
        """Two `make eval`s side by side: stopping one must not close the other's run."""
        from agent_platform.evals.core.pipeline.generate import GenerationStage

        path = DuckDbEvalStore().db_path
        theirs, ours = DuckDbEvalStore(path), DuckDbEvalStore(path)
        plugin = FakePlugin()
        other = await GenerationStage(plugin, theirs).run("golden", plugin.load_cases("golden"))  # type: ignore[arg-type]
        theirs.execute("UPDATE eval_runs SET status = 'running' WHERE run_id = ?", (other.run_id,))
        assert ours.mark_interrupted() == []
        assert theirs.get_run(other.run_id).status == "running"  # type: ignore[union-attr]


class TestRateLimitNamesTheModel:
    def test_a_json_request_names_its_model(self) -> None:
        body = json.dumps({"model": "nvidia/nemotron-3-ultra-550b-a55b"}).encode()
        request = httpx.Request("POST", "https://x.invalid/v1/chat/completions", content=body)
        assert model_of(request) == "nvidia/nemotron-3-ultra-550b-a55b"

    def test_a_request_without_one_is_named_by_its_path(self) -> None:
        request = httpx.Request("POST", "https://x.invalid/v1/security/jailbreak", content=b"{}")
        assert model_of(request) == "/v1/security/jailbreak"
