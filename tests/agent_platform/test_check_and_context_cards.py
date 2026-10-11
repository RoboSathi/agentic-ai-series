"""Cards that say one thing once: the evidence line a verdict rests on, a page's chrome kept
out of evidence, a batch of page reads as one CONTEXT card, and a budget stop named by its
own mechanism."""

from typing import Any

# The guardrails package first: importing hallucination_check first hits its import cycle.
import agent_platform.guardrails  # noqa: F401
from agent_platform.hallucination_check.readings import readings
from agent_platform.hallucination_check.types import DetectionVerdict, EvidenceCheck
from agent_platform.llm.types import CompletionRequest, Message
from agent_platform.loop_control import BudgetGuard
from agent_platform.observability.model_observer import StageReporterModelObserver
from agent_platform.web.fetcher import extract_text


def _line(label: str, score: float, *, chosen: bool = False) -> EvidenceCheck:
    return EvidenceCheck(
        rank=1, similarity=0.9, tier="tool_result", excerpt=f"{label} line", nli_label=label,
        nli_score=score, decisive=True, source="https://a.example/p", model="nli-model",
        chosen=chosen,
    )  # fmt: skip


def test_a_supported_claim_quotes_the_support_not_a_stronger_contradiction() -> None:
    # The resolver chose the entailment (any support beats a contradiction); a 0.986
    # contradiction from a menu line must not be drawn as what decided it.
    verdict = DetectionVerdict(
        claim="c", supported=True, confidence=0.945, tier="tool_result",
        evidence_checks=[_line("entailment", 0.945, chosen=True), _line("contradiction", 0.986)],
    )  # fmt: skip
    [step] = readings(verdict)
    assert step.evidence[0].excerpt == "entailment line"
    assert step.result.startswith("entailment")


def test_a_pages_menu_and_footer_are_not_text() -> None:
    html = (
        "<nav><a>Topics</a><a>Trending</a><a>Collections</a></nav>"
        "<article><h1>Reflection</h1><p>The agent drafts, critiques, revises.</p>"
        "<ul><li>first</li><li>second</li></ul></article><footer>© 2026</footer>"
    )
    _, text = extract_text(html, chrome=["nav", "footer"], blocks=["ul", "li"])
    assert "Topics" not in text and "©" not in text
    assert "first" in text.splitlines() and "second" in text.splitlines()


class _Reporter:
    def __init__(self) -> None:
        self.cards: list[dict[str, Any]] = []

    def emit(self, stage: str, status: str, **kwargs: Any) -> None:
        self.cards.append({"stage": stage, **kwargs})


def _request() -> CompletionRequest:
    return CompletionRequest(messages=[Message(role="user", content="read this page")])


def test_page_reads_are_one_context_card_per_batch() -> None:
    reporter = _Reporter()
    observer = StageReporterModelObserver(lambda: reporter, aggregate=["extractor"])  # type: ignore[arg-type, return-value]
    for _ in range(3):
        observer.request_starting("extractor", "m-1", _request())
    observer.request_starting("planner", "m-1", _request())
    assert len(reporter.cards) == 1  # the planner's own card; page reads wait for the batch
    observer.flush()
    assert len(reporter.cards) == 2
    assert "3 requests" in reporter.cards[1]["summary"]
    assert "of 131k window" in reporter.cards[0]["summary"]
    observer.flush()
    assert len(reporter.cards) == 2  # nothing pending, nothing drawn


def test_a_spent_counter_reports_as_its_own_mechanism() -> None:
    guard = BudgetGuard({"rounds": 1, "searches": 1}, layers={"rounds": "step_count"})
    guard.charge("rounds", 1)
    assert guard.verdict(1, 1, stopping=True).intervened is False
    assert guard.verdict(1, 1, stopping=True, cause="rounds").layer == "step_count"
    guard.charge("searches", 1)
    # Not named: the guard's own layer, as before.
    assert guard.verdict(1, 1, stopping=True, cause="searches").layer == "token_budget"


def test_a_menu_or_heading_may_not_refute_but_a_statement_may() -> None:
    from agent_platform.config.settings import RefutingPremiseSettings
    from agent_platform.hallucination_check.resolver import GroundingResolver

    resolver = GroundingResolver.__new__(GroundingResolver)
    resolver._refuting = RefutingPremiseSettings(min_words=6, max_capitalised_share=0.6)
    assert not resolver._is_statement("Topics Trending Collections Enterprise")
    assert not resolver._is_statement("Process Optimization Quality Assurance Error Detection Bias")
    assert resolver._is_statement("Reflection lets an agent critique its own draft and revise it.")
