"""DeepEval as a grading stage — optional, offline, and on the harness's own client.

DeepEval is never a runtime dependency: it is an optional extra
(`pip install -e .[evals]`), imported here lazily and only when the
`deepeval` stage runs. Its metrics call a model; that model is the eval
judge, reached through the harness's `LLMClient` — so every DeepEval call
is logged, costed and drawn from the same rate-limit budget as the rest.

`deepeval.evaluate()` is deliberately not used: it brings its own
concurrency, caching, console output and telemetry. Each metric's
`a_measure` is called directly, under the pipeline's concurrency.
"""

import asyncio
import json
import os
from typing import Any

from agent_platform.config import get_settings
from agent_platform.evals.contracts import (
    GoldenCase,
    Grader,
    GraderKind,
    QualityMetricSpec,
    Score,
    Transcript,
    canonical_hash,
)
from agent_platform.evals.judge import extract_json
from agent_platform.evals.judge_client import create_judge_client
from agent_platform.llm.client import LLMClient
from agent_platform.llm.recording import RecordingLLMClient
from agent_platform.llm.types import CompletionRequest, Message

_TELEMETRY_ENV = "DEEPEVAL_TELEMETRY_OPT_OUT"
_OPT_OUT = "YES"
_INSTALL_HINT = "DeepEval is not installed. Install the eval extra: pip install -e '.[evals]'"
_TURN_SEPARATOR = "\n"


class DeepEvalUnavailableError(RuntimeError):
    """The optional `deepeval` package is missing."""


def deepeval_graders(specs: list[QualityMetricSpec]) -> list[Grader]:
    """One grader per configured metric."""
    _import_deepeval()
    return [DeepEvalGrader(spec) for spec in specs]


class DeepEvalGrader:
    """Runs one DeepEval metric over a transcript."""

    kind: GraderKind = "deepeval"

    def __init__(self, spec: QualityMetricSpec, client: LLMClient | None = None) -> None:
        judge = get_settings().evals.judge
        self._settings = get_settings().evals.deepeval
        self._spec = spec
        self._client = client or create_judge_client(judge)
        self._evidence_cap = judge.max_evidence_chars
        self.name = f"deepeval.{spec.name}"
        self.version = canonical_hash(
            [spec.model_dump(), judge.model_dump(), self._settings.model_dump()]
        )[:16]

    async def grade(self, case: GoldenCase, transcript: Transcript) -> list[Score]:
        deepeval = _import_deepeval()
        recording = RecordingLLMClient(self._client)
        metric = self._metric(deepeval, recording)
        test_case = deepeval.test_case.LLMTestCase(
            input=_TURN_SEPARATOR.join(case.turns),
            actual_output=transcript.reply,
            retrieval_context=self._evidence(transcript) if self._spec.uses_evidence else None,
        )
        await metric.a_measure(test_case, _show_indicator=False)
        value = float(metric.score or 0.0)
        return [
            Score(
                case_id=case.case_id,
                metric=self._spec.name,
                grader=self.name,
                kind=self.kind,
                severity="soft",
                value=round(value, 4),
                passed=value >= self._spec.threshold,
                threshold=self._spec.threshold,
                tier="calibrated_llm_judge",
                rationale=str(metric.reason or ""),
                cost_usd=recording.cost_usd,
                llm_calls=recording.calls,
            )
        ]

    def _metric(self, deepeval: Any, client: LLMClient) -> Any:
        model = harness_model(deepeval, client)
        metrics = deepeval.metrics
        common = {"threshold": self._spec.threshold, "model": model, "async_mode": True}
        kind = self._spec.kind
        if kind == "answer_relevancy":
            return metrics.AnswerRelevancyMetric(
                include_reason=self._settings.include_reason, **common
            )
        if kind == "faithfulness":
            return metrics.FaithfulnessMetric(
                include_reason=self._settings.include_reason, **common
            )
        if kind == "geval":
            params = deepeval.test_case.SingleTurnParams
            return metrics.GEval(
                name=self._spec.name,
                criteria=self._spec.criteria or None,
                evaluation_steps=self._spec.evaluation_steps or None,
                evaluation_params=[params.INPUT, params.ACTUAL_OUTPUT],
                **common,
            )
        raise ValueError(f"Unknown DeepEval metric kind '{kind}'")

    def _evidence(self, transcript: Transcript) -> list[str]:
        if self._evidence_cap <= 0:
            return transcript.evidence
        kept, used = [], 0
        for item in transcript.evidence:
            if used + len(item) > self._evidence_cap:
                break
            kept.append(item)
            used += len(item)
        return kept


def harness_model(deepeval: Any, client: LLMClient) -> Any:
    """A `DeepEvalBaseLLM` backed by a harness `LLMClient` (Adapter)."""
    base = deepeval.models.base_model.DeepEvalBaseLLM

    class HarnessModel(base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            self._client = client
            super().__init__(model=client.model)

        def load_model(self, *args: Any, **kwargs: Any) -> Any:
            return self._client

        def get_model_name(self, *args: Any, **kwargs: Any) -> str:
            return str(self._client.model)

        def generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
            return asyncio.run(self.a_generate(prompt, schema=schema))

        async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
            judge = get_settings().evals.judge
            result = await self._client.complete(
                CompletionRequest(
                    messages=[Message(role="user", content=prompt)],
                    temperature=judge.temperature,
                    max_tokens=judge.max_tokens,
                )
            )
            if schema is None:
                return result.content
            return schema.model_validate(json.loads(extract_json(result.content)))

    return HarnessModel()


def _import_deepeval() -> Any:
    """Import DeepEval lazily, with its telemetry off when configured."""
    if get_settings().evals.deepeval.telemetry_opt_out:
        os.environ.setdefault(_TELEMETRY_ENV, _OPT_OUT)
    try:
        import deepeval
        import deepeval.metrics
        import deepeval.models.base_model
        import deepeval.test_case
    except ImportError as exc:
        raise DeepEvalUnavailableError(_INSTALL_HINT) from exc
    return deepeval
