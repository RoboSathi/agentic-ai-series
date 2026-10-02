"""Spans tolerate typography, never content; the result is always the page's own text."""

import pytest

from research_agent.research.spans import canonical_span, nearest_match

PAGE = (
    "Pope Leo XIV said \u201cAI going rogue\u201d is not \u2018fake news\u2019.\n"
    "The model\u00a0calculated the amplitude \u2014 to nine-loop order\u2014surpassing humans.\n\n"
    "Omega cut handling time by up to 75%."
)


@pytest.mark.parametrize(
    "span",
    [
        "said \"AI going rogue\" is not 'fake news'.",  # straight quotes for curly ones
        "The model calculated the amplitude - to nine-loop order-surpassing humans.",  # dash, nbsp
        "is not 'fake news'. The model calculated",  # newline in the page, space in the span
        "Omega   cut handling\ttime by up to 75%.",  # stray whitespace in the span
    ],
)
def test_typographic_differences_are_forgiven_and_the_pages_own_text_is_returned(span: str) -> None:
    found = canonical_span(PAGE, span)
    assert found is not None and found in PAGE


def test_an_exact_span_is_returned_unchanged() -> None:
    assert canonical_span(PAGE, "Omega cut handling time by up to 75%.") == (
        "Omega cut handling time by up to 75%."
    )


@pytest.mark.parametrize(
    "span",
    [
        "Omega cut handling time by up to 90%.",  # a changed number
        "omega cut handling time by up to 75%.",  # a changed case
        "Omega reduced handling time by up to 75%.",  # a changed word
        "handling time Omega cut by up to 75%.",  # a changed order
        "",
        "   ",
    ],
)
def test_content_differences_are_still_rejected(span: str) -> None:
    assert canonical_span(PAGE, span) is None


def test_the_canonical_span_ends_where_the_pages_text_ends() -> None:
    found = canonical_span(PAGE, "to nine-loop order-surpassing humans.")
    assert found == "to nine-loop order\u2014surpassing humans."


def test_nearest_match_reports_how_close_a_rejected_span_came() -> None:
    near = nearest_match(PAGE, "Omega cut handling time by up to 90%.")
    assert near.coverage > 0.8 and "Omega" in near.context
    far = nearest_match(PAGE, "Something entirely unrelated to this page")
    assert far.coverage < 0.5
