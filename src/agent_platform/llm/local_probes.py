"""In-process model probes for `make model-check` — loaded and run, not just found on disk.

A model directory that exists can still fail to load (a partial download, a library upgrade
that cannot read it) and one that loads can still be slow. Each probe loads its model the way
the harness does, runs it once on a tiny configured input, and reports both timings: the cost
paid once per process, and the cost paid per call.

Built from configuration like the endpoint probes: only the models the running agent's
configuration names and enables are probed. A model not on disk yet is downloaded first, by the
same loader the agent uses, so the same command readies a fresh checkout.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Literal

from pydantic import BaseModel

from agent_platform.config import AppConfig, get_settings, resolve_path
from agent_platform.logging import describe_error

LocalStatus = Literal["ok", "downloaded", "failed"]


class LocalProbeResult(BaseModel):
    """One in-process model: where it is, whether it ran, and how long that took."""

    name: str
    model: str
    path: str
    status: LocalStatus
    load_ms: float = 0.0
    infer_ms: float = 0.0
    detail: str = ""


# A probe's work: load (blocking, in a thread), then one inference returning what it said.
Load = Callable[[], None]
Infer = Callable[[], Awaitable[str]]


class LocalModelProbe:
    """Load a model, run it once, time both."""

    def __init__(self, name: str, model: str, path: str, load: Load, infer: Infer) -> None:
        self.name = name
        self._model = model
        self._path = path
        self._load = load
        self._infer = infer

    async def probe(self) -> LocalProbeResult:
        base = LocalProbeResult(name=self.name, model=self._model, path=self._path, status="ok")
        # A model not on disk yet is downloaded by its own loader here, like on first use.
        downloaded = not _has_files(self._path)
        try:
            started = time.perf_counter()
            await asyncio.to_thread(self._load)
            loaded = time.perf_counter()
            detail = await self._infer()
            done = time.perf_counter()
        except Exception as exc:
            return base.model_copy(update={"status": "failed", "detail": describe_error(exc)})
        return base.model_copy(
            update={
                "status": "downloaded" if downloaded else "ok",
                "load_ms": round((loaded - started) * 1000, 1),
                "infer_ms": round((done - loaded) * 1000, 1),
                "detail": detail,
            }
        )


LocalProbe = LocalModelProbe


def build_local_probes(settings: AppConfig | None = None) -> list[LocalProbe]:
    """Every in-process model the configuration names, in a stable order."""
    config = settings or get_settings()
    return [
        probe
        for probe in (
            _nli_probe(config),
            _reranker_probe(config),
            _intent_probe(config),
            _embedder_probe(config),
            _summarizer_probe(config),
            _presidio_probe(config),
        )
        if probe is not None
    ]


def _nli_probe(config: AppConfig) -> LocalProbe | None:
    encoders = config.llm.encoders
    path = encoders.nli_local_model_dir
    if not path:
        return None
    if not encoders.enabled:
        return None
    from agent_platform.llm.local_nli import shared_local_nli_model

    model = shared_local_nli_model()
    check = config.llm.check.local

    async def infer() -> str:
        result = await model.entails(check.nli_premise, check.nli_hypothesis)
        return f"{result.label} {result.score:.2f}"

    # A model that needs no loading (the lexical fallback a test substitutes) loads instantly.
    load = getattr(model, "ensure_loaded_sync", _nothing_to_load)
    return LocalModelProbe("nli", encoders.nli_model, path, load, infer)


def _reranker_probe(config: AppConfig) -> LocalProbe | None:
    encoders = config.llm.encoders
    if not encoders.enabled or not encoders.reranker_local_model_dir:
        return None
    from agent_platform.llm.local_reranker import LocalCrossEncoderRanker

    ranker = LocalCrossEncoderRanker()
    check = config.llm.check.local

    async def infer() -> str:
        scores = await ranker.rank(check.reranker_query, list(check.reranker_passages))
        return " · ".join(f"{score:.2f}" for score in scores)

    return LocalModelProbe(
        "reranker",
        encoders.reranker_model,
        encoders.reranker_local_model_dir,
        ranker.ensure_loaded_sync,
        infer,
    )


def _intent_probe(config: AppConfig) -> LocalProbe | None:
    """The intent classifier the configuration selects, not merely the one it names first."""
    settings = config.llm.intent_classifier
    if not settings.enabled:
        return None
    if settings.provider == "laya_local":
        return _laya_intent_probe(config)
    path = settings.local_model_dir
    if not path:
        return None
    from agent_platform.intent.classifier import HuggingFaceIntentClassifier

    classifier = HuggingFaceIntentClassifier()
    text = config.llm.check.local.intent_text

    async def infer() -> str:
        candidates = await classifier.classify(text)
        top = max(candidates, key=lambda c: c.score, default=None)
        return f"{top.intent} {top.score:.2f}" if top is not None else "no label"

    return LocalModelProbe(
        "intent_classifier", settings.model, path, classifier.ensure_loaded_sync, infer
    )


def _laya_intent_probe(config: AppConfig) -> LocalProbe | None:
    settings = config.llm.intent_classifier
    if not settings.laya.local_model_dir:
        return None
    from agent_platform.intent.laya import LayaIntentClassifier

    check = config.llm.check.local
    classifier = LayaIntentClassifier(check.intent_choices, check.intent_question, settings)

    async def infer() -> str:
        top = (await classifier.classify(check.intent_text))[0]
        return f"{top.intent} {top.score:.2f}"

    return LocalModelProbe(
        "intent_classifier",
        classifier.model_name,
        settings.laya.local_model_dir,
        classifier.ensure_loaded_sync,
        infer,
    )


def _embedder_probe(config: AppConfig) -> LocalProbe | None:
    settings = config.llm.local_embedder
    if not settings.enabled or not settings.local_model_dir:
        return None
    from agent_platform.llm.local_embeddings import LocalSentenceEmbedder

    embedder = LocalSentenceEmbedder(settings)
    text = config.llm.check.local.intent_text

    async def infer() -> str:
        return f"{len((await embedder.embed([text]))[0])} dimensions"

    return LocalModelProbe(
        "sentence_embedder",
        settings.model,
        settings.local_model_dir,
        embedder.ensure_loaded_sync,
        infer,
    )


def _summarizer_probe(config: AppConfig) -> LocalProbe | None:
    settings = config.llm.dialogue_summarizer
    path = settings.local_model_dir
    if not path:
        return None
    if not settings.enabled:
        return None
    from agent_platform.llm.local_summarizer import LocalDialogueSummarizer

    summarizer = LocalDialogueSummarizer(settings)
    dialogue = config.llm.check.local.dialogue

    async def infer() -> str:
        return (await summarizer.summarize(dialogue)) or "no summary"

    return LocalModelProbe(
        "dialogue_summarizer", settings.model, path, summarizer.ensure_loaded_sync, infer
    )


def _presidio_probe(config: AppConfig) -> LocalProbe | None:
    pii = config.security.pii
    if pii.engine != "presidio":
        return None
    from agent_platform.security.pii.detector import PatternPIIDetector
    from agent_platform.security.pii.presidio import (
        PresidioPIIDetector,
        shared_presidio_analyzer,
    )

    detector = PresidioPIIDetector(fallback=PatternPIIDetector())
    text = config.llm.check.local.intent_text

    def load() -> None:
        if shared_presidio_analyzer() is None:
            raise RuntimeError("Presidio did not load (see the pii log)")

    async def infer() -> str:
        found = detector.detect(text)
        return ", ".join(span.category for span in found) or "no PII"

    settings = pii.presidio
    return LocalModelProbe(
        "pii_presidio",
        f"{settings.model}-{settings.version}",
        settings.local_model_dir,
        load,
        infer,
    )


def _nothing_to_load() -> None:
    return None


def _has_files(configured: str) -> bool:
    path = resolve_path(configured)
    return path.is_dir() and any(path.iterdir())


__all__ = ["LocalProbeResult", "build_local_probes"]


def download_all(settings: AppConfig | None = None) -> list[LocalProbeResult]:
    """Download (if missing), load and run every local model this configuration uses."""
    return [asyncio.run(probe.probe()) for probe in build_local_probes(settings)]


def main() -> int:
    """`make models [agent]`: ready this agent's local models; non-zero if one fails."""
    failed = 0
    for result in download_all():
        failed += result.status == "failed"
        print(f"{result.status:10} {result.name:20} {result.model}  {result.detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
