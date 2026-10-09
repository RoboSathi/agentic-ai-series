"""The check recorder: one `fact_check` row per guard, classifier or model dependency call.

Schema §6.3. Three places call it, and none of them builds a row by hand:

- `MessageScreener` hands over the guard entries a screen produced
  (`record_screen`), which this maps onto the check vocabulary — the rule
  scans, the hosted jailbreak and content-safety models, PII;
- the grounding protocol reports the claims it judged (`record_grounding`);
- the model-dependency factories are wrapped by `recorded_dependency`, so
  every NLI, embedding, rerank and intent-model call is timed, and one that
  raises is recorded as an `error`.

A screen's guard reports its status in the guards' words (`PASS`, `FAIL`,
`SKIPPED`, `ADVISORY`); the verdict a fact carries is the analytics vocabulary
(UL "Check"), including `fail_open` — a guard that was unavailable while the
turn went on unprotected.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Final, TypeVar

from agent_platform.config import get_settings
from agent_platform.observability.facts import CheckFact, fact_store

# What a check looked at (UL "Check").
PII: Final = "pii"
INJECTION: Final = "injection"
JAILBREAK: Final = "jailbreak"
MODERATION: Final = "moderation"
CONTENT_SAFETY: Final = "content_safety"
GROUNDING: Final = "grounding"
# Sentence confidence from the model's own token log-probabilities (step 1 of grounding).
LOGPROB: Final = "logprob"
NLI: Final = "nli"
EMBEDDING: Final = "embedding"
RERANK: Final = "rerank"
INTENT_MODEL: Final = "intent_model"
# Why it ran.
SCREEN: Final = "screen"
GROUNDING_PURPOSE: Final = "grounding"
DEPENDENCY: Final = "dependency"
# What answered.
RULE: Final = "rule"
LOCAL_MODEL: Final = "local_model"
HOSTED_MODEL: Final = "hosted_model"
LLM: Final = "llm"
# What it decided.
PASS: Final = "pass"
FLAG: Final = "flag"
BLOCK: Final = "block"
SKIPPED: Final = "skipped"
ERROR: Final = "error"
FAIL_OPEN: Final = "fail_open"

# A guard entry's status, in the guards' own words (`security/suite.py`).
_GUARD_FAIL, _GUARD_SKIPPED, _GUARD_ADVISORY = "FAIL", "SKIPPED", "ADVISORY"
_UNAVAILABLE_SUFFIX = "_UNAVAILABLE"

T = TypeVar("T")


def record_check(
    check: str,
    purpose: str,
    backend: str,
    model: str,
    verdict: str,
    latency_ms: float,
    score: float | None = None,
    items: int | None = None,
    flagged: int | None = None,
) -> None:
    """Append one check invocation (never raises: the store swallows its own failures)."""
    fact_store().append(
        CheckFact(
            check_name=check,
            purpose=purpose,
            backend=backend,
            model=model,
            verdict=verdict,
            score=score,
            latency_ms=latency_ms,
            items=items,
            flagged=flagged,
        )
    )


def record_screen(
    entries: Sequence[Mapping[str, Any]], *, rule: str, rule_hit: bool, elapsed_ms: float
) -> None:
    """One screen's rows: each model guard it consulted, and its rule scan.

    The rule scan always runs, but its guard entry is written only on some
    paths, so the screen says what the rule found (`rule_hit`); its latency is
    what the screen spent outside the model guards.
    """
    names = _guard_names()
    model_ms = 0.0
    for entry in entries:
        check = names.get(str(entry.get("name", "")))
        if check is None or check == rule:
            continue
        latency = float(entry.get("latency_ms") or 0.0)
        model_ms += latency
        record_check(
            check,
            SCREEN,
            HOSTED_MODEL,
            str(entry.get("model", "")),
            _verdict(entry),
            latency,
            score=_score(entry),
        )
    rule_ms = max(elapsed_ms - model_ms, 0.0)
    record_check(rule, SCREEN, RULE, "", BLOCK if rule_hit else PASS, rule_ms)


def record_pii(model: str, unavailable: str, matches: int, latency_ms: float) -> None:
    """The PII screen: it never blocks, so a match is a flag."""
    backend = LOCAL_MODEL if model and not unavailable else RULE
    verdict = FLAG if matches else PASS
    record_check(PII, SCREEN, backend, model, verdict, latency_ms, items=None, flagged=matches)


def record_grounding(
    backend: str, model: str, claims: int, unsupported: int, latency_ms: float
) -> None:
    """The grounding check over one reply: claims judged, and those it could not support."""
    verdict = FLAG if unsupported else PASS
    record_check(
        GROUNDING,
        GROUNDING_PURPOSE,
        backend,
        model,
        verdict,
        latency_ms,
        items=claims,
        flagged=unsupported,
    )


def recorded_dependency(
    check: str, method: str, backend: Callable[[Any], str]
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorate a model-dependency factory: every `method` call on what it builds is recorded.

    The factory's product is instrumented in place (once — a cached, shared model
    is not wrapped twice), so callers keep the protocol type they asked for.
    """

    def decorate(factory: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(factory)
        def build(*args: Any, **kwargs: Any) -> T:
            product = factory(*args, **kwargs)
            if product is not None:
                _instrument(product, check, method, backend)
            return product

        return build

    return decorate


def _instrument(product: Any, check: str, method: str, backend: Callable[[Any], str]) -> None:
    marker = f"_recorded_{method}"
    if getattr(product, marker, False):
        return
    call: Callable[..., Awaitable[Any]] = getattr(product, method)

    @functools.wraps(call)
    async def recorded(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        verdict = PASS
        try:
            return await call(*args, **kwargs)
        except Exception:
            verdict = ERROR
            raise
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            record_check(check, DEPENDENCY, backend(product), _model_of(product), verdict, elapsed)

    setattr(product, method, recorded)
    setattr(product, marker, True)


def _model_of(product: Any) -> str:
    """The model a dependency answered with, under whichever name its protocol uses."""
    for attribute in ("model_name", "model_id", "model"):
        value = getattr(product, attribute, None)
        if isinstance(value, str):
            return value
    return ""


def _guard_names() -> dict[str, str]:
    """The guards' configured check names, mapped onto the check vocabulary."""
    guards = get_settings().security.guards
    return {
        guards.rule_checks.injection: INJECTION,
        guards.rule_checks.moderation: MODERATION,
        guards.jailbreak.check_name: JAILBREAK,
        guards.content_safety.check_name: CONTENT_SAFETY,
    }


def _verdict(entry: Mapping[str, Any]) -> str:
    """A guard entry's status as a check verdict."""
    status = str(entry.get("status", ""))
    if status == _GUARD_FAIL:
        return BLOCK
    if status == _GUARD_ADVISORY:
        return FLAG
    if status == _GUARD_SKIPPED and str(entry.get("reason_code", "")).endswith(_UNAVAILABLE_SUFFIX):
        return FAIL_OPEN if get_settings().security.guards.fail_open_on_unavailable else ERROR
    if status == _GUARD_SKIPPED:
        return SKIPPED
    return PASS


def _score(entry: Mapping[str, Any]) -> float | None:
    score = entry.get("score")
    return float(score) if isinstance(score, int | float) else None


__all__ = [
    "BLOCK",
    "CONTENT_SAFETY",
    "DEPENDENCY",
    "EMBEDDING",
    "ERROR",
    "FAIL_OPEN",
    "FLAG",
    "GROUNDING",
    "HOSTED_MODEL",
    "INJECTION",
    "INTENT_MODEL",
    "JAILBREAK",
    "LLM",
    "LOCAL_MODEL",
    "LOGPROB",
    "MODERATION",
    "NLI",
    "PASS",
    "PII",
    "RERANK",
    "RULE",
    "SCREEN",
    "SKIPPED",
    "record_check",
    "record_grounding",
    "record_pii",
    "record_screen",
    "recorded_dependency",
]
