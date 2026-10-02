"""A direct answer written as prose becomes points; anything else is left as written."""

from research_agent.graph.points import as_points

SOURCES = "Sources:"


def _points(text: str) -> str:
    return as_points(text, SOURCES, "-", 3)


def test_a_paragraph_becomes_one_point_per_sentence() -> None:
    text = "Meta reported a test. Google reorganised DeepMind, e.g. a new chair. Gartner saw 96.5 % growth."
    assert _points(text).splitlines() == [
        "- Meta reported a test.",
        "- Google reorganised DeepMind, e.g. a new chair.",
        "- Gartner saw 96.5 % growth.",
    ]


def test_the_sources_block_is_kept_as_written() -> None:
    text = "A one. B two. C three.\n\nSources:\n- https://x.example/a.b"
    assert _points(text).endswith("Sources:\n- https://x.example/a.b")


def test_a_short_answer_and_an_existing_list_are_unchanged() -> None:
    assert _points("It is 26 °C in Pune.") == "It is 26 °C in Pune."
    listed = "- one\n- two\n- three"
    assert _points(listed) == listed
