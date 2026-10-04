"""PII detection with Presidio over a local spaCy pipeline.

Regex finds what has a shape — an email, a card number. Names and places have none; Presidio's
spaCy NER finds them, and its recognizers check what a pattern cannot (a card number's Luhn
digit, a phone number's country plan). The deployment's own patterns (`security.pii.patterns`)
are registered beside them, so a jurisdiction-specific identifier stays a config edit.

The pipeline is unpacked into a folder inside the repository (`make models`, or the first load)
and never fetched again at run time. One analyzer is resident per process: a `/harness` rebuild
composes a new filter, not a second pipeline.

Loading takes seconds (Presidio's import and recognizer registry, then spaCy), so a console
starts it on a background thread before its event loop opens and does not wait: the window
comes up at once, and a message that arrives first waits for the analyzer. An analyzer that
cannot load leaves detection to the configured patterns, and says so once.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from agent_platform.config import get_settings
from agent_platform.config.settings import PresidioSettings
from agent_platform.logging import describe_error, get_logger
from agent_platform.security.pii.types import PIISpan

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine

    from agent_platform.security.pii.detector import PIIDetector

_lock = threading.Lock()
_analyzer: AnalyzerEngine | None = None
_failure: str = ""


class PresidioPIIDetector:
    """A `PIIDetector` over the shared analyzer, limited to the configured entities.

    `fallback` answers when the analyzer cannot load (the configured patterns).
    """

    def __init__(self, fallback: PIIDetector, settings: PresidioSettings | None = None) -> None:
        self._settings = settings or get_settings().security.pii.presidio
        self._fallback = fallback

    @property
    def model(self) -> str:
        """The NER pipeline behind it, as a card names it ("" once it fell back to patterns)."""
        return "" if _failure else f"{self._settings.model}-{self._settings.version}"

    @property
    def unavailable(self) -> str:
        """Why Presidio could not load ("" while it is fine)."""
        return _failure

    def detect(self, text: str) -> list[PIISpan]:
        if not text.strip():
            return []
        analyzer = shared_presidio_analyzer()
        if analyzer is None:
            return self._fallback.detect(text)
        results = analyzer.analyze(
            text=text,
            language=self._settings.language,
            entities=self._settings.entities,
            score_threshold=self._settings.score_threshold,
        )
        return [_span(text, result) for result in results]


def preload_presidio_in_background() -> threading.Thread:
    """Start loading the analyzer now, on its own thread; the caller does not wait for it."""
    thread = threading.Thread(target=shared_presidio_analyzer, name="presidio-load", daemon=True)
    thread.start()
    return thread


def shared_presidio_analyzer() -> AnalyzerEngine | None:
    """The process's analyzer, loaded once however many threads ask; None if it cannot load."""
    global _analyzer, _failure
    with _lock:
        if _analyzer is None and not _failure:
            try:
                _analyzer = _build_analyzer()
            except Exception as exc:
                _failure = describe_error(exc)
                get_logger("pii").warning(
                    "Presidio unavailable; PII detection uses the configured patterns only",
                    error=_failure,
                )
        return _analyzer


def _build_analyzer() -> AnalyzerEngine:
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    pii = get_settings().security.pii
    settings = pii.presidio
    folder = ensure_pipeline(settings)
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": settings.language, "model_name": str(folder)}],
        }
    )
    analyzer = AnalyzerEngine(
        nlp_engine=provider.create_engine(), supported_languages=[settings.language]
    )
    for entity, regex in pii.patterns.items():
        analyzer.registry.add_recognizer(
            PatternRecognizer(
                supported_entity=entity,
                patterns=[Pattern(name=entity.lower(), regex=regex, score=settings.pattern_score)],
                supported_language=settings.language,
            )
        )
    get_logger("pii").info(
        "Presidio analyzer loaded", model=settings.model, version=settings.version,
        path=str(folder), entities=settings.entities,
    )  # fmt: skip
    return analyzer


def ensure_pipeline(settings: PresidioSettings) -> Path:
    """The pipeline's folder — unpacked from its release wheel first if it is not there yet.

    Unpacked into a sibling temp folder and renamed into place, so a download that dies
    halfway never leaves a half-populated folder the next start would try to load.
    """
    folder = settings.resolved_local_model_dir
    if (folder / "config.cfg").exists():
        return folder
    if not settings.download_if_missing:
        raise RuntimeError(f"spaCy pipeline missing at {folder}; run `make models`")
    url = settings.download_url.format(model=settings.model, version=settings.version)
    get_logger("pii").warning("spaCy pipeline missing; downloading", url=url)
    response = httpx.get(url, follow_redirects=True, timeout=settings.download_timeout_seconds)
    response.raise_for_status()
    folder.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{folder.name}.", dir=folder.parent))
    try:
        _unpack(response.content, f"{settings.model}/{settings.model}-{settings.version}/", staging)
        if folder.exists():
            shutil.rmtree(folder)
        staging.rename(folder)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return folder


def _unpack(wheel: bytes, prefix: str, into: Path) -> None:
    """The pipeline folder inside the wheel (`<model>/<model>-<version>/…`), and nothing else."""
    with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
        members = [name for name in archive.namelist() if name.startswith(prefix)]
        if not members:
            raise RuntimeError(f"the wheel holds no pipeline under {prefix}")
        for name in members:
            target = into / name[len(prefix) :]
            if name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))


def _span(text: str, result: Any) -> PIISpan:
    explanation = result.recognition_metadata or {}
    return PIISpan(
        category=result.entity_type,
        start=result.start,
        end=result.end,
        text=text[result.start : result.end],
        detector=str(explanation.get("recognizer_name", "presidio")),
        score=float(result.score),
    )


__all__ = [
    "PresidioPIIDetector",
    "ensure_pipeline",
    "preload_presidio_in_background",
    "shared_presidio_analyzer",
]
