"""The order list is a box table, coloured by state in its cells and nowhere else."""

from agent_platform.config import get_settings
from agent_platform.persistence import SqliteConnectionPool
from agent_platform.presentation import box_table
from agent_platform.tui.widgets import render_response
from customer_support_agent.persistence.repositories import OrdersRepository
from customer_support_agent.presentation.orders import customer_orders, render_order_list


class TestTheBoxTable:
    def test_every_row_is_as_wide_as_the_header(self) -> None:
        lines = box_table(["#", "Total"], [["1", "$1.00"], ["10", "$145.50"]], ["right", "right"])
        assert len({len(line) for line in lines}) == 1
        assert lines[3].endswith("   $1.00 │") and lines[4].startswith("│ 10 │")


class TestTheOrderList:
    def test_it_lists_each_order_in_its_own_row(self, seeded_pool: SqliteConnectionPool) -> None:
        orders = customer_orders(OrdersRepository(seeded_pool), 1)
        rendered = render_order_list(orders)
        assert all(f"│ {order.order_id} │" in rendered for order in orders)

    def test_a_state_is_coloured_in_its_cell_but_not_in_a_sentence(self) -> None:
        text = (
            "│ 1 │ 10001 │ 2026-09-22 │ Keyboard │ delivered  │ none │ $1.00 │\n"
            "Your order was delivered yesterday."
        )
        out = render_response(text, get_settings().tui.response_styles)
        styled = {out.plain[span.start : span.end]: str(span.style) for span in out.spans}
        assert styled["delivered"] == "green"
        assert styled["Your order was delivered yesterday."] == "default"
