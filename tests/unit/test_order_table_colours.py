"""The order table's headings read as headings, and its grid recedes.

The whole heading row used to be `bold` — the same weight as a data row's
order id — so the headings read as one more row. A highlight rule may now
be scoped to the lines it applies to (`line`), so the headings take their
own colour cell by cell and a heading word in a data row is left alone.
"""

from typing import Any

from agent_platform.config import get_settings
from agent_platform.tui.widgets import render_response
from customer_support_agent.presentation.orders import render_order_list


def _styles(text: str) -> dict[str, set[str]]:
    rendered = render_response(text, get_settings().tui.response_styles)
    found: dict[str, set[str]] = {}
    for span in rendered.spans:
        found.setdefault(rendered.plain[span.start : span.end].strip(), set()).add(str(span.style))
    return found


def _table(order: Any) -> str:
    return render_order_list([order])


class TestTheHeadingRow:
    def test_each_heading_takes_the_heading_colour(self, seeded_pool: Any) -> None:
        from customer_support_agent.persistence.repositories import OrdersRepository

        [order, *_] = OrdersRepository(seeded_pool).for_user(1)
        styles = _styles(_table(order))
        assert "bold cyan" in styles["Order"] and "bold cyan" in styles["Total"]
        assert styles["│"] == {"bright_black"}

    def test_a_heading_word_in_a_sentence_is_left_alone(self) -> None:
        assert "Order" not in _styles("Order 10001 is on its way.")
