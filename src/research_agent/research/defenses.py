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
from urllib.parse import urlparse

from pydantic import BaseModel

from agent_platform.knowledge.web import SearchHit
from agent_platform.logging import instrument
from agent_platform.security.prompt_injection.detector import InstructionDataSeparator
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


def _on(url: str, domains: Sequence[str]) -> bool:
    host = (urlparse(url).hostname or "").casefold()
    return any(host == d.casefold() or host.endswith(f".{d.casefold()}") for d in domains)


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
