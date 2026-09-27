"""Rendering orders for a person to read."""

from customer_support_agent.presentation.orders import (
    OrderMenu,
    build_menu,
    customer_orders,
    describe_customer,
    describe_orders,
    render_order_detail,
    render_order_list,
    resolve_selection,
)

__all__ = [
    "OrderMenu",
    "build_menu",
    "customer_orders",
    "describe_customer",
    "describe_orders",
    "render_order_detail",
    "render_order_list",
    "resolve_selection",
]
