"""Evidence ledger: append-only, and the verbatim-span rule."""

from agentic_commons.evidence import (
    EvidenceCandidate,
    EvidenceEntry,
    EvidenceLedger,
    SpanRejection,
    VerbatimSpanValidator,
)
from agentic_commons.hashing import text_hash

PAGE = "The Veltrane cell stores 410 Wh/kg. It entered pilot production in 2025."


class Pages:
    def __init__(self, pages: dict[str, str]) -> None:
        self._pages = pages

    def text_for(self, source_url: str) -> str | None:
        return self._pages.get(source_url)


def _candidate(span: str, url: str = "https://a.example/p", page: str = PAGE) -> EvidenceCandidate:
    return EvidenceCandidate(
        claim="claim", span=span, source_url=url, source_hash=text_hash(page), subject="q1"
    )


def _ledger(rejected: list[SpanRejection] | None = None) -> EvidenceLedger:
    sink = rejected if rejected is not None else []
    return EvidenceLedger(
        VerbatimSpanValidator(Pages({"https://a.example/p": PAGE})), "E", on_rejected=sink.append
    )


class TestVerbatimRule:
    def test_an_exact_substring_is_admitted(self) -> None:
        entry = _ledger().append(_candidate("stores 410 Wh/kg"))
        assert isinstance(entry, EvidenceEntry)
        assert entry.entry_id == "E1"

    def test_a_paraphrase_is_rejected(self) -> None:
        rejected: list[SpanRejection] = []
        result = _ledger(rejected).append(_candidate("stores about 410 Wh per kg"))
        assert isinstance(result, SpanRejection)
        assert result.reason_code == "SPAN_NOT_VERBATIM"
        assert rejected == [result]

    def test_an_unfetched_source_is_rejected(self) -> None:
        result = _ledger().append(_candidate("stores", url="https://never.example"))
        assert isinstance(result, SpanRejection) and result.reason_code == "SOURCE_UNKNOWN"

    def test_a_span_citing_another_version_of_the_page_is_rejected(self) -> None:
        result = _ledger().append(_candidate("stores", page="an older copy"))
        assert isinstance(result, SpanRejection) and result.reason_code == "SOURCE_HASH_MISMATCH"

    def test_an_empty_span_is_rejected(self) -> None:
        result = _ledger().append(_candidate("   "))
        assert isinstance(result, SpanRejection) and result.reason_code == "EMPTY_SPAN"


class TestLedger:
    def test_it_only_grows_and_snapshots_are_immutable(self) -> None:
        ledger = _ledger()
        ledger.append(_candidate("stores 410 Wh/kg"))
        snapshot = ledger.snapshot()
        ledger.append(_candidate("pilot production in 2025"))
        assert len(snapshot.entries) == 1
        assert len(ledger.entries) == 2
        assert snapshot.entry("E1") is not None
        assert snapshot.for_subject("q1")[0].entry_id == "E1"
        assert snapshot.sources == ["https://a.example/p"]

    def test_a_duplicate_returns_the_existing_entry(self) -> None:
        ledger = _ledger()
        first = ledger.append(_candidate("stores 410 Wh/kg"))
        second = ledger.append(_candidate("stores 410 Wh/kg"))
        assert first == second and len(ledger.entries) == 1

    def test_the_same_span_found_by_another_subject_is_not_new_evidence(self) -> None:
        ledger = _ledger()
        first = ledger.append(_candidate("stores 410 Wh/kg"))
        again = ledger.append(_candidate("stores 410 Wh/kg").model_copy(update={"subject": "q2"}))
        assert again == first and len(ledger.entries) == 1

    def test_restore_and_discard(self) -> None:
        ledger = _ledger()
        ledger.append(_candidate("stores 410 Wh/kg"))
        ledger.append(_candidate("nope"))
        snapshot = ledger.snapshot()
        other = _ledger()
        other.restore(snapshot)
        assert len(other.entries) == 1 and len(other.rejections) == 1
        other.discard()
        assert other.entries == [] and other.rejections == []
