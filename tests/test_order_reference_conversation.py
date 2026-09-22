from __future__ import annotations

from customer_support_agent.core.presentation import OrderMenu, resolve_selection


def test_order_reference_is_not_interpreted_as_menu_selection() -> None:
    menu = OrderMenu(order_ids=[10560, 10428])

    assert resolve_selection("order 10560", menu) is None
    assert resolve_selection("what about 10560", menu) is None


def test_bare_number_remains_an_explicit_menu_selection() -> None:
    menu = OrderMenu(order_ids=[10560, 10428])

    assert resolve_selection("1", menu) == 10560
    assert resolve_selection("10560", menu) == 10560
