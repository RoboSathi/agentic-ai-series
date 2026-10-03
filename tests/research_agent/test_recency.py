"""The recency filter must read the date formats providers actually return.

Regression: Tavily's news vertical returns RFC 2822 dates ("Tue, 29 Sep 2026 13:53:01 GMT").
`within_window` once parsed only ISO dates, so every such hit was rejected as "outside the
window" and a Q&A run for "AI news in the last 30 days" found nothing.
"""

from datetime import date

import pytest

from research_agent.domain.sources import parse_published, window_verdict, within_window

AS_OF = date(2026, 10, 1)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Tue, 29 Sep 2026 13:53:01 GMT", date(2026, 9, 29)),  # Tavily news (from a real log)
        ("Wed, 02 Sep 2026 02:12:43 GMT", date(2026, 9, 2)),
        ("2026-09-29T13:53:01Z", date(2026, 9, 29)),  # ISO with time and Z
        ("2026-09-29T13:53:01+00:00", date(2026, 9, 29)),
        ("2026-09-29", date(2026, 9, 29)),
        ("", None),
        ("yesterday", None),
        ("29/09/2026", None),
    ],
)
def test_parse_published(raw: str, expected: date | None) -> None:
    assert parse_published(raw) == expected


def test_every_rfc_date_from_the_failing_run_is_inside_thirty_days() -> None:
    dates = [
        "Tue, 29 Sep 2026 13:53:01 GMT",
        "Tue, 29 Sep 2026 05:13:43 GMT",
        "Wed, 30 Sep 2026 17:24:45 GMT",
        "Sat, 12 Sep 2026 11:00:00 GMT",
        "Tue, 15 Sep 2026 02:52:23 GMT",
        "Wed, 02 Sep 2026 02:12:43 GMT",
        "Fri, 25 Sep 2026 12:04:41 GMT",
    ]
    assert all(window_verdict(d, AS_OF, 30) == "in" for d in dates)


def test_old_date_is_out_and_unreadable_date_is_undated() -> None:
    assert window_verdict("Mon, 01 Jan 2024 00:00:00 GMT", AS_OF, 30) == "out"
    assert window_verdict("not a date", AS_OF, 30) == "undated"
    assert window_verdict("", AS_OF, 30) == "undated"


def test_window_edge_is_inclusive() -> None:
    assert window_verdict("2026-09-01", AS_OF, 30) == "in"
    assert window_verdict("2026-08-31", AS_OF, 30) == "out"


def test_within_window_stays_strict_for_the_audit() -> None:
    assert within_window("Tue, 29 Sep 2026 13:53:01 GMT", AS_OF, 30)
    assert not within_window("", AS_OF, 30)  # an undated claim cannot be shown fresh
