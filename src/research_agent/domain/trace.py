"""What a run did, kept as it goes, read by the synthesis and the audit.

Plain bookkeeping with no behaviour beyond recording, so it sits in the
domain: the research engine writes it, and Coverage & Limits and the audit
read it, without either importing the other.
"""

from pydantic import BaseModel, Field


class SeenHit(BaseModel):
    """A search result as the provider gave it: what a best-effort report can still cite."""

    node_id: str
    url: str
    title: str = ""
    snippet: str = ""
    published: str = ""


class RunTrace(BaseModel):
    """Mutable run bookkeeping that ends up in Coverage & Limits and the audit."""

    queries_by_node: dict[str, list[str]] = Field(default_factory=dict)
    sources_examined: list[str] = Field(default_factory=list)
    rejected_sources: dict[str, str] = Field(default_factory=dict)
    # A short machine code per rejected URL (recency, undated, dead_link, injection),
    # so a report can group rejections instead of listing every one.
    rejected_codes: dict[str, str] = Field(default_factory=dict)
    blocked_outbound: list[str] = Field(default_factory=list)
    attempted_outbound: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    span_rejections: int = 0
    provider_summaries_seen: list[str] = Field(default_factory=list)
    budget_stops: list[str] = Field(default_factory=list)
    # Every search result the run saw, first sighting only: a run whose ledger stays empty
    # writes its best-effort report from these (`research/snippets.py`).
    seen_hits: list[SeenHit] = Field(default_factory=list)
    # The one rescue round of a run that found nothing: whether it ran (recency relaxed from
    # then on) and the searches and fetches it was granted on top of the size's budget.
    rescued: bool = False
    rescue_grant: dict[str, float] = Field(default_factory=dict)
    # Why a source admitted under the rescue is weaker evidence ("outside the window"), by URL.
    caveats: dict[str, str] = Field(default_factory=dict)

    def tried(self, node_id: str, query: str) -> None:
        self.queries_by_node.setdefault(node_id, []).append(query)

    def examined(self, url: str) -> None:
        if url not in self.sources_examined:
            self.sources_examined.append(url)

    def saw(self, hit: SeenHit) -> None:
        if all(seen.url != hit.url for seen in self.seen_hits):
            self.seen_hits.append(hit)

    def reject(self, url: str, reason: str, code: str = "other") -> None:
        self.rejected_sources.setdefault(url, reason)
        self.rejected_codes.setdefault(url, code)
