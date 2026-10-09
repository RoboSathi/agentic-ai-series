---
name: list-orders
description: List the customer's orders, or only their orders in one state (pending, shipped, delivered, cancelled).
triggers: [intent:LIST_ORDERS]
keywords: [my orders, order history, what did i order, purchases, pending orders, delivered orders]
---

## Preconditions
- The customer wants to know which orders they have.

## Procedure
1. Call `list_orders`, narrowed to the state they named if they named one.
2. List what it returned, one order per line, with the values given.

## Boundaries
- List only what the tool returned; never add an order from memory.

## Completion
Stop once the list has been given.
