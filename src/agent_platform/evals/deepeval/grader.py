"""DeepEval as the golden suite's quality grader — run the standard way, `evaluate()`.

One grader for every metric an agent declares: each case becomes a
`ConversationalTestCase` and goes through `deepeval.evaluate()` with the
agent's metrics, all judged by the platform judge (`judge:`). DeepEval runs
its own event loop, so the call is made in a worker thread — one case at a
time per process, because DeepEval keeps its test run in a process-wide
manager; the metrics of that case still run concurrently. Its console
output, cache and telemetry are switched off — the eval store and the
Markdown report are the record.

DeepEval is never a runtime dependency: it is the optional `evals` extra,
imported here lazily and only when this grader runs. A metric that errors
(the judge unreachable, or its answer unusable after repair) is recorded
as `<metric>.error`, which is never reused, rather than as a low score.
"""

import asyncio
import os
import threading
from typing import Any

from agent_platform.config import get_settings
from agent_platform.evals.core.contracts import (
    ConversationFrame,
    GoldenCase,
    GraderKind,
    QualityMetricSpec,
    Score,
    Transcript,
    canonical_hash,
)
from agent_platform.evals.core.pipeline.grade import ERROR_METRIC_SUFFIX
from agent_platform.evals.deepeval.metrics import build_metrics
from agent_platform.evals.deepeval.model import judge_model
from agent_platform.evals.deepeval.test_cases import conversation
from agent_platform.llm.client import LLMClient

_TELEMETRY_ENV = "DEEPEVAL_TELEMETRY_OPT_OUT"
_OPT_OUT = "YES"
_INSTALL_HINT = "DeepEval is not installed. Install the eval extra: pip install -e '.[evals]'"
_VERSION_CHARS = 16
# DeepEval's test-run manager is process-wide: one `evaluate()` at a time.
_EVALUATE_LOCK = threading.Lock()


class DeepEvalUnavailableError(RuntimeError):
    """The optional `deepeval` package is missing."""


class DeepEvalGrader:
    """Every quality metric of one agent, over one transcript at a time."""

    kind: GraderKind = "deepeval"
    name = "deepeval"

    def __init__(
        self,
        specs: list[QualityMetricSpec],
        frame: Any,
        client: LLMClient | None = None,
    ) -> None:
        """`frame(case)` gives the role, scenario and expected outcome a case is judged against."""
        self._deepeval = import_deepeval()
        self._specs = specs
        self._frame = frame
        self._client = client
        judge, settings = get_settings().judge, get_settings().evals.deepeval
        self.version = canonical_hash(
            [[spec.model_dump() for spec in specs], judge.model_dump(), settings.model_dump()]
        )[:_VERSION_CHARS]

    async def grade(self, case: GoldenCase, transcript: Transcript) -> list[Score]:
        frame: ConversationFrame = self._frame(case)
        return await asyncio.to_thread(self._evaluate, case, transcript, frame)

    def _evaluate(
        self, case: GoldenCase, transcript: Transcript, frame: ConversationFrame
    ) -> list[Score]:
        """One `evaluate()` call — in this worker thread, on DeepEval's own loop."""
        deepeval = self._deepeval
        model = judge_model(deepeval, self._client)
        metrics = build_metrics(deepeval, self._specs, model)
        cap = get_settings().judge.max_evidence_chars
        test_case = conversation(deepeval, case, transcript, frame, cap)
        settings = get_settings().evals.deepeval
        from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig

        with _EVALUATE_LOCK:
            result = deepeval.evaluate(
                test_cases=[test_case],
                metrics=metrics,
                async_config=AsyncConfig(run_async=True, max_concurrent=settings.max_concurrent),
                display_config=DisplayConfig(show_indicator=False, print_results=False),
                cache_config=CacheConfig(write_cache=False, use_cache=False),
                error_config=ErrorConfig(ignore_errors=True),
            )
        names = {
            metric.__name__: spec.name for metric, spec in zip(metrics, self._specs, strict=True)
        }
        data = [item for test in result.test_results for item in (test.metrics_data or [])]
        share = len(data) or 1
        return [
            self._score(case, item, names.get(item.name, item.name), model, share) for item in data
        ]

    def _score(self, case: GoldenCase, item: Any, metric: str, model: Any, share: int) -> Score:
        errored = bool(item.error)
        value = 0.0 if errored or item.score is None else float(item.score)
        return Score(
            case_id=case.case_id,
            metric=f"{metric}{ERROR_METRIC_SUFFIX}" if errored else metric,
            grader=self.name,
            kind=self.kind,
            severity="soft",
            value=round(value, 4),
            passed=bool(item.success) and not errored,
            threshold=float(item.threshold) if item.threshold is not None else None,
            tier="calibrated_llm_judge",
            rationale=str(item.error or item.reason or ""),
            # The judge's spend for the case, shared evenly over its metrics.
            cost_usd=model.cost_usd / share,
            llm_calls=round(model.calls / share),
        )


def import_deepeval() -> Any:
    """Import DeepEval lazily, with its telemetry off when configured."""
    if get_settings().evals.deepeval.telemetry_opt_out:
        os.environ.setdefault(_TELEMETRY_ENV, _OPT_OUT)
    try:
        import deepeval
        import deepeval.evaluate.configs
        import deepeval.metrics
        import deepeval.metrics.g_eval.utils
        import deepeval.models.base_model
        import deepeval.test_case
    except ImportError as exc:
        raise DeepEvalUnavailableError(_INSTALL_HINT) from exc
    if get_settings().evals.deepeval.quiet:
        import deepeval.test_run.test_run as test_run

        # Its end-of-run banner is printed on this module's Rich console.
        test_run.console.quiet = True
    return deepeval
