"""Fetched-content defence, layers 1 and 2 (layer 3 is the quarantined extractor).

Layer 1 — structural (`injection_structural`). Nothing here detects an
attack; it removes the attacker's options. Workers hold only `search` and
`fetch`. Every outbound query and URL is checked before it leaves: it may
not contain a string the user marked private, it must match an allow rule
and no deny rule, and a fetch may not target a domain the Brief blocks.
Search hits are filtered and ordered by the Brief's source policy.
Fetched text is ranked at the bottom of the authority hierarchy, so
nothing a page says can raise its own standing.

Layer 2 — classifier (`injection_classifier`). The platform's
prompt-injection detector reads every fetched page; a flagged page is
dropped, logged as a security event, and counted as a rejected source.
"""

import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Literal, Protocol
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from agentic_commons.logging import instrument
from agentic_commons.security.guard_types import GuardVerdict
from agentic_commons.security.prompt_injection.detector import InstructionDataSeparator
from agentic_commons.web import SearchHit
from research_agent.config.settings import ResearchSettings
from research_agent.domain.scope import ScopeBrief


class OutboundVerdict(BaseModel):
    allowed: bool
    reason_code: str = ""
    detail: str = ""


ALLOWED = OutboundVerdict(allowed=True)


@instrument("research_agent")
class OutboundGuard:
    """Checks a query or URL before it leaves the process (exfiltration control)."""

    def __init__(self, brief: ScopeBrief, settings: ResearchSettings) -> None:
        self._private = [term.casefold() for term in brief.private_terms if term.strip()]
        self._blocked = [domain.casefold() for domain in brief.source_policy.blocked_domains]
        self._allow = [re.compile(pattern) for pattern in settings.outbound_allow_patterns]
        self._deny = [re.compile(pattern) for pattern in settings.outbound_deny_patterns]

    def check_query(self, query: str) -> OutboundVerdict:
        return self._common(query) or ALLOWED

    def check_url(self, url: str) -> OutboundVerdict:
        common = self._common(url)
        if common is not None:
            return common
        host = (urlparse(url).hostname or "").casefold()
        if any(host == domain or host.endswith(f".{domain}") for domain in self._blocked):
            return OutboundVerdict(allowed=False, reason_code="BLOCKED_DOMAIN", detail=host)
        return ALLOWED

    def _common(self, value: str) -> OutboundVerdict | None:
        folded = value.casefold()
        leaked = [term for term in self._private if term in folded]
        if leaked:
            return OutboundVerdict(
                allowed=False, reason_code="PRIVATE_TERM", detail="brief-private string"
            )
        if not any(pattern.search(value) for pattern in self._allow):
            return OutboundVerdict(allowed=False, reason_code="NOT_ALLOWED", detail=value[:80])
        denied = next((pattern.pattern for pattern in self._deny if pattern.search(value)), None)
        if denied is not None:
            return OutboundVerdict(allowed=False, reason_code="DENY_RULE", detail=denied)
        return None


def apply_source_policy(hits: Sequence[SearchHit], brief: ScopeBrief) -> list[SearchHit]:
    """Drop blocked domains; put preferred domains first (stable otherwise)."""
    policy = brief.source_policy
    kept = [hit for hit in hits if not _on(hit.url, policy.blocked_domains)]
    return sorted(kept, key=lambda hit: 0 if _on(hit.url, policy.preferred_domains) else 1)


WindowVerdict = Literal["in", "out", "undated"]


def parse_published(source_date: str) -> date | None:
    """A provider's publication date as a calendar date, or None when it cannot be read.

    Providers disagree on format: DuckDuckGo news returns ISO 8601, Tavily's news
    vertical returns RFC 2822 (`Tue, 29 Sep 2026 13:53:01 GMT`). Both are read here;
    anything else is "unknown", never silently "old".
    """
    text = (source_date or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text).date()
    except (TypeError, ValueError, IndexError):
        return None


def window_verdict(source_date: str, as_of: date, days: int) -> WindowVerdict:
    """Whether a source's date is inside the window, outside it, or unreadable."""
    published = parse_published(source_date)
    if published is None:
        return "undated"
    return "in" if published >= as_of - timedelta(days=days) else "out"


def within_window(source_date: str, as_of: date, days: int) -> bool:
    """Whether a source's date falls inside a recency window; an undated source does not."""
    return window_verdict(source_date, as_of, days) == "in"


def _on(url: str, domains: Sequence[str]) -> bool:
    host = (urlparse(url).hostname or "").casefold()
    return any(host == d.casefold() or host.endswith(f".{d.casefold()}") for d in domains)


class InjectionGuardModel(Protocol):
    """A hosted injection/jailbreak model (the platform's `NvidiaHostedJailbreakGuard`)."""

    @property
    def model(self) -> str: ...

    async def classify(self, text: str) -> GuardVerdict: ...


class InjectionVerdict(BaseModel):
    """What the screen decided about one page, and on which evidence."""

    decision: Literal["clean", "kept", "dropped"] = "clean"
    phrases: list[str] = Field(default_factory=list)
    score: float | None = None  # the highest model score seen
    model: str = ""
    why: str = ""

    @property
    def drop(self) -> bool:
        return self.decision == "dropped"


@instrument("research_agent")
class InjectionScreen:
    """Layer 2: a phrase check over fetched text, with a hosted model as the second opinion.

    A phrase hit alone is a weak signal ("you are now" turns up in ordinary prose), so with a
    model configured the phrase is only a feature: the model reads the text around it and the
    page is dropped only if the model agrees. With `scan_all` the model also reads pages the
    phrase check passed, which is how an attack in no listed wording is caught.

    Fail-safe either way: a phrase hit whose model check is unavailable still drops the page,
    exactly as it did before a model was added.
    """

    def __init__(
        self,
        separator: InstructionDataSeparator,
        guard: InjectionGuardModel | None = None,
        *,
        threshold: float = 0.5,
        scan_all: bool = False,
        window_chars: int = 2000,
        max_windows: int = 4,
    ) -> None:
        self._separator = separator
        self._guard = guard
        self._threshold = threshold
        self._scan_all = scan_all
        self._window = window_chars
        self._max_windows = max_windows

    def flagged(self, text: str) -> list[str]:
        """The injection phrasings found (empty when the page looks clean)."""
        return self._separator.find_phrases(text)

    async def assess(self, text: str) -> InjectionVerdict:
        phrases = self.flagged(text)
        if self._guard is None:
            if not phrases:
                return InjectionVerdict()
            return InjectionVerdict(decision="dropped", phrases=phrases, why="phrase check")
        if not phrases and not self._scan_all:
            return InjectionVerdict(model=self._guard.model)
        best: float | None = None
        unavailable = False
        for window in self._windows(text, phrases):
            verdict = await self._guard.classify(window)
            if not verdict.available:
                unavailable = True
                continue
            if verdict.score is not None:
                best = verdict.score if best is None else max(best, verdict.score)
            if verdict.flagged or (verdict.score is not None and verdict.score >= self._threshold):
                return self._result(
                    "dropped",
                    phrases,
                    verdict.score if verdict.score is not None else best,
                    "model flagged it",
                )
        model = self._guard.model
        if phrases and unavailable:
            return InjectionVerdict(
                decision="dropped", phrases=phrases, model=model,
                why="phrase hit and the injection model was unavailable",
            )  # fmt: skip
        if phrases:
            return self._result("kept", phrases, best, "model scored the context as benign")
        return InjectionVerdict(score=best, model=model)

    def _result(
        self,
        decision: Literal["kept", "dropped"],
        phrases: list[str],
        score: float | None,
        why: str,
    ) -> InjectionVerdict:
        assert self._guard is not None
        shown = f" (score {score:.2f})" if score is not None else ""
        return InjectionVerdict(
            decision=decision,
            phrases=phrases,
            score=score,
            model=self._guard.model,
            why=why + shown,
        )

    def _windows(self, text: str, phrases: list[str]) -> list[str]:
        """Text around each flagged phrase; with none, evenly spaced windows over the page."""
        half = self._window // 2
        found: list[str] = []
        for phrase in phrases:
            hit = re.search(re.escape(phrase), text, re.IGNORECASE)
            if hit is not None:
                found.append(text[max(0, hit.start() - half) : hit.start() + half])
        if phrases:
            return found[: self._max_windows] or [text[: self._window]]
        count = min(self._max_windows, max(1, -(-len(text) // self._window)))
        step = max(len(text) - self._window, 0) / max(count - 1, 1)
        return [text[int(i * step) : int(i * step) + self._window] for i in range(count)]


def rank_by_kind(hits: Sequence[SearchHit], labels: Mapping[str, Sequence[str]]) -> list[SearchHit]:
    """Hits from labelled kinds of source (press releases, say) after the rest; none dropped."""
    suffixes = [s.casefold() for group in labels.values() for s in group]

    def labelled(hit: SearchHit) -> int:
        host = (urlparse(hit.url).hostname or "").casefold()
        return 1 if any(host == s or host.endswith(f".{s}") for s in suffixes) else 0

    return sorted(hits, key=labelled)


def delimit(separator: InstructionDataSeparator, instructions: str, text: str) -> str:
    """Untrusted page text in its data envelope, closing tag neutralised."""
    return separator.separate(instructions, text).delimited_data


def is_primary(url: str, markers: Sequence[str]) -> bool:
    """Whether a source counts as primary under the configured host markers."""
    host = (urlparse(url).hostname or "").casefold()
    return any(marker.casefold() in host for marker in markers)
