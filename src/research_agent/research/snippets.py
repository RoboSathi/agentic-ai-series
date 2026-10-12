"""A search snippet as evidence: weaker than a page, better than nothing, and labelled.

When a page cannot be read (blocked, paywalled, unreadable) the search engine's snippet is still
the page's own words, and a run that admits nothing reports nothing. So a snippet's sentences
are admitted in the page's place — under the same verbatim rule, checked against the snippet
itself — each with a caveat the writer and the reader both see.

    SnippetSources   the texts the verbatim rule reads: fetched pages, and the snippets
                     admitted in their place (a snippet stands for its URL only when the page
                     itself was never admitted)
    SnippetAdmission one hit's snippet → ledger entries, with a caveat
"""

from agent_platform.core.primitives.hashing import text_hash
from agent_platform.hallucination_check import split_sentences
from agent_platform.hallucination_check.evidence import (
    EvidenceCandidate,
    EvidenceEntry,
    EvidenceLedger,
)
from agent_platform.hallucination_check.evidence.validator import SourceTextLookup
from research_agent.domain.trace import SeenHit


class SnippetSources:
    """Fetched pages first; a URL whose snippet was admitted reads as that snippet."""

    def __init__(self, pages: SourceTextLookup) -> None:
        self._pages = pages
        self._snippets: dict[str, str] = {}

    def remember(self, url: str, snippet: str) -> str:
        """Hold a snippet as the text behind `url`; its hash, for the entries that cite it."""
        self._snippets[url] = snippet
        return text_hash(snippet)

    def text_for(self, source_url: str) -> str | None:
        if source_url in self._snippets:
            return self._snippets[source_url]
        return self._pages.text_for(source_url)


class SnippetAdmission:
    """One search hit's snippet, sentence by sentence, into the ledger with a caveat."""

    def __init__(self, ledger: EvidenceLedger, sources: SnippetSources, min_words: int) -> None:
        self._ledger = ledger
        self._sources = sources
        self._min_words = min_words

    def admit(self, hit: SeenHit, caveat: str) -> list[EvidenceEntry]:
        """The entries admitted from the hit's snippet (none for a snippet with no claim)."""
        sentences = [s.strip() for s in split_sentences(hit.snippet) if self._claim_sized(s)]
        if not sentences:
            return []
        page_hash = self._sources.remember(hit.url, hit.snippet)
        outcomes = [
            self._ledger.append(
                EvidenceCandidate(
                    claim=sentence,
                    span=sentence,
                    source_url=hit.url,
                    source_hash=page_hash,
                    source_date=hit.published,
                    subject=hit.node_id,
                    caveat=caveat,
                )
            )
            for sentence in sentences
        ]
        return [outcome for outcome in outcomes if isinstance(outcome, EvidenceEntry)]

    def _claim_sized(self, sentence: str) -> bool:
        return len(sentence.split()) >= self._min_words


__all__ = ["SnippetAdmission", "SnippetSources"]
