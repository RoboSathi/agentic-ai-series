"""Generates seed_data.sql and (re)builds agentic.db from schema.sql + seed_data.sql.

A deterministic (fixed random seed), reproducible synthetic dataset — 100
users, 1000 orders. Not a source of real data, purely a demo fixture.
Re-running regenerates both seed_data.sql and agentic.db from scratch.

The data obeys the domain's own rules. Generating each column
independently is easy and wrong: it produced pending orders with
completed refunds, which is a state the business cannot reach, and an
agent asked to reason over impossible data will produce impossible
answers and look like a model problem.
"""

import random
from datetime import datetime, timedelta, timezone

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger
from agentic_commons.persistence import SqliteConnectionPool
from customer_support_agent.config import get_agent_settings
from customer_support_agent.dummy_data import DUMMY_DATA_DIR

# The DDL lives with the persistence layer that owns it, not with the
# fixture data that happens to be loaded into it.
_SCHEMA_FILE = DUMMY_DATA_DIR.parent / "persistence" / "schema.sql"
_SEED_DATA_FILE = DUMMY_DATA_DIR / "seed_data.sql"
_AGENT_POLICY_FILE = DUMMY_DATA_DIR / "agent_policy_seed.sql"

_NUM_USERS = 100
_NUM_ORDERS = 1000
_RANDOM_SEED = 42
_FIRST_ORDER_NUMBER = 10001
_FIRST_USER_NUMBER = 1

# user_code is the credential a customer types to authenticate. Seven
# characters from an alphabet with no 0/O or 1/I/L: a code that is read
# aloud or copied from an email should not hinge on telling those apart.
_USER_CODE_LENGTH = 7
_USER_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"

_FIRST_NAMES = [
    "Ada",
    "Grace",
    "Alan",
    "Katherine",
    "Linus",
    "Barbara",
    "Dennis",
    "Radia",
    "Ken",
    "Margaret",
    "Edsger",
    "Frances",
    "Tony",
    "Jean",
    "Donald",
    "Adele",
]
_LAST_NAMES = [
    "Lovelace",
    "Hopper",
    "Turing",
    "Johnson",
    "Torvalds",
    "Liskov",
    "Ritchie",
    "Perlman",
    "Thompson",
    "Hamilton",
    "Dijkstra",
    "Allen",
    "Hoare",
    "Bartik",
]


# Every code carries at least one digit. Without that rule a seven-letter
# English word is a well-formed code, and "what's the weather" reads as a
# login attempt — the customer is then told their code was not found,
# which is baffling and wrong.
_USER_CODE_DIGITS = "23456789"


def _user_code(rng: random.Random, taken: set[str]) -> str:
    """A unique 7-character code, unambiguous and never a plain word."""
    while True:
        body = rng.choices(_USER_CODE_ALPHABET, k=_USER_CODE_LENGTH - 1)
        body.insert(rng.randrange(_USER_CODE_LENGTH), rng.choice(_USER_CODE_DIGITS))
        code = "".join(body)
        if code not in taken and any(char.isdigit() for char in code):
            taken.add(code)
            return code


_ORDER_STATUS_WEIGHTS: list[tuple[str, float]] = [
    ("pending", 0.10),
    ("shipped", 0.20),
    ("delivered", 0.60),
    ("cancelled", 0.10),
]
# An order becomes refundable only once it is settled: delivered (the
# customer has it and wants their money back) or cancelled before
# delivery (they never received it). A pending or shipped order is still
# in flight, so there is nothing to refund yet — generating one with a
# completed refund produces data that contradicts the policy the agent is
# asked to apply, and the agent then learns the contradiction.
_REFUNDABLE_ORDER_STATUSES = frozenset({"delivered", "cancelled"})

# Refund outcomes for an order that has reached a refundable state.
_REFUND_STATUS_WEIGHTS: list[tuple[str, float]] = [
    ("none", 0.70),
    ("pending_review", 0.08),
    ("approved", 0.06),
    ("denied", 0.06),
    ("completed", 0.10),
]

# The only refund status an in-flight order can carry.
_NO_REFUND = "none"

# One shop, one currency.
_CURRENCY = "USD"

# The catalogue this shop sells. A name and a price band, so an order's
# total is consistent with what it claims to be: a $7 laptop is the kind
# of fixture that makes a grounding check look broken when it is right.
_CATALOGUE: list[tuple[str, str, float, float]] = [
    ("Wireless Noise-Cancelling Headphones", "AUD-WH-100", 120.00, 320.00),
    ("Mechanical Keyboard, 87-key", "PER-KB-087", 60.00, 180.00),
    ("USB-C Charging Cable, 2m", "ACC-CBL-2M", 8.00, 22.00),
    ("Cotton Crew T-Shirt", "APP-TS-CRW", 12.00, 35.00),
    ("Stainless Water Bottle, 750ml", "HOM-WB-750", 15.00, 40.00),
    ("Espresso Grinder", "KIT-GR-ESP", 90.00, 260.00),
    ("Running Shoes", "APP-SH-RUN", 55.00, 160.00),
    ("Desk Lamp, Adjustable", "HOM-LMP-ADJ", 25.00, 75.00),
    ("Bluetooth Speaker", "AUD-SPK-BT", 35.00, 130.00),
    ("Laptop Stand, Aluminium", "ACC-STD-ALU", 30.00, 85.00),
]

_CARRIERS = ["Parcelforce", "DHL", "UPS", "Royal Mail", "FedEx"]

# Where a parcel in transit has got to. Any of these is consistent with
# "shipped"; none of them is consistent with "delivered" or "cancelled",
# which is the rule `_whereabouts` below enforces.
_TRANSIT_LOCATIONS = [
    "Birmingham depot",
    "Manchester sorting centre",
    "London Heathrow hub",
    "Leeds distribution centre",
    "Bristol depot",
    "out for delivery",
]


def refund_status_for(rng: random.Random, order_status: str) -> str:
    """The refund status an order in this state is allowed to have."""
    if order_status not in _REFUNDABLE_ORDER_STATUSES:
        return _NO_REFUND
    return _weighted_choice(rng, _REFUND_STATUS_WEIGHTS)


def _whereabouts(
    rng: random.Random, order_status: str, timestamp: str, now: datetime
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Carrier, tracking, ETA, delivered_at and location for this status.

    The rule the agent applies is the rule the fixtures obey, exactly as
    `refund_status_for` already does for refunds. A cancelled order with
    a delivery date, or a pending one already in a depot, is data that
    contradicts the policy — and the agent then learns the contradiction.
    """
    placed = datetime.fromisoformat(timestamp)
    if order_status == "pending":
        # Not handed to a carrier yet: an estimate, and nothing else.
        eta = (placed + timedelta(days=rng.randint(3, 10))).isoformat()
        return None, None, eta, None, None
    if order_status == "cancelled":
        # Never shipped, so there is nothing on its way and no date to give.
        return None, None, None, None, None
    carrier = rng.choice(_CARRIERS)
    tracking = f"{carrier[:2].upper()}{rng.randint(10**9, 10**10 - 1)}GB"
    if order_status == "delivered":
        delivered = placed + timedelta(days=rng.randint(1, 9))
        delivered = min(delivered, now)
        return carrier, tracking, delivered.isoformat(), delivered.isoformat(), None
    eta = (placed + timedelta(days=rng.randint(2, 8))).isoformat()
    return carrier, tracking, eta, None, rng.choice(_TRANSIT_LOCATIONS)


def _sql_null(value: str | None) -> str:
    """Inline SQL for an optional text column."""
    return _sql_str(value) if value is not None else "NULL"


def _sql_str(value: str) -> str:
    """Quote and escape a string for inline SQL."""
    return "'" + value.replace("'", "''") + "'"


def _weighted_choice(rng: random.Random, weights: list[tuple[str, float]]) -> str:
    values = [value for value, _ in weights]
    probs = [prob for _, prob in weights]
    return rng.choices(values, weights=probs, k=1)[0]


def _generate_seed_sql() -> str:
    """Build seed_data.sql's content: INSERTs for every user and order."""
    rng = random.Random(_RANDOM_SEED)
    now = datetime.now(timezone.utc)
    lines = ["-- Generated by generate_dummy_data.py — do not hand-edit.", ""]

    lines.append("-- Users")
    user_ids = list(range(_FIRST_USER_NUMBER, _FIRST_USER_NUMBER + _NUM_USERS))
    taken: set[str] = set()
    for i, user_id in enumerate(user_ids, start=1):
        created_at = (now - timedelta(days=rng.randint(30, 720))).isoformat()
        name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
        email = f"user{i}@example.com"
        phone = f"555-{rng.randint(1000, 9999)}"
        lines.append(
            "INSERT INTO users (user_id, user_code, created_at, name, email, phone) VALUES "
            f"({user_id}, {_sql_str(_user_code(rng, taken))}, {_sql_str(created_at)}, "
            f"{_sql_str(name)}, {_sql_str(email)}, {_sql_str(phone)});"
        )

    lines.append("")
    lines.append("-- Orders")
    for i in range(_NUM_ORDERS):
        order_id = _FIRST_ORDER_NUMBER + i
        user_id = rng.choice(user_ids)
        timestamp = (now - timedelta(days=rng.randint(0, 90))).isoformat()
        order_status = _weighted_choice(rng, _ORDER_STATUS_WEIGHTS)
        refund_status = refund_status_for(rng, order_status)
        product_name, sku, low, high = rng.choice(_CATALOGUE)
        quantity = rng.choices([1, 2, 3], weights=[0.75, 0.18, 0.07], k=1)[0]
        # Integer cents throughout: a price drawn in cents, a total that is
        # an exact product — no float ever becomes money.
        per_major = 10 ** get_settings().money.minor_units
        unit_price_cents = rng.randint(round(low * per_major), round(high * per_major))
        # The total is what was actually bought, not an unrelated number.
        amount_cents = unit_price_cents * quantity
        carrier, tracking, eta, delivered_at, location = _whereabouts(
            rng, order_status, timestamp, now
        )
        lines.append(
            "INSERT INTO orders "
            "(order_id, user_id, timestamp, order_status, refund_status, amount_cents, "
            "product_name, product_sku, quantity, unit_price_cents, currency, "
            "carrier, tracking_number, estimated_delivery_date, delivered_at, "
            "current_location) "
            f"VALUES ({order_id}, {user_id}, "
            f"{_sql_str(timestamp)}, {_sql_str(order_status)}, "
            f"{_sql_str(refund_status)}, {amount_cents}, "
            f"{_sql_str(product_name)}, {_sql_str(sku)}, {quantity}, {unit_price_cents}, "
            f"{_sql_str(_CURRENCY)}, "
            f"{_sql_null(carrier)}, {_sql_null(tracking)}, {_sql_null(eta)}, "
            f"{_sql_null(delivered_at)}, {_sql_null(location)});"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    """Write seed_data.sql and (re)build agentic.db from schema.sql + seed_data.sql."""
    logger = get_logger("persistence")

    seed_sql = _generate_seed_sql()
    _SEED_DATA_FILE.write_text(seed_sql)
    logger.info(f"Wrote {_SEED_DATA_FILE}")

    agent_settings = get_agent_settings()
    from agentic_commons.config import get_settings
    platform_settings = get_settings()
    db_path = agent_settings.resolved_db_path
    if db_path.exists():
        db_path.unlink()

    pool = SqliteConnectionPool(db_path)
    pool.execute_script(_SCHEMA_FILE.read_text())
    pool.execute_script(seed_sql)

    agent_policy_sql = _AGENT_POLICY_FILE.read_text().replace(
        "__AGENT_ID__", platform_settings.agent.id
    ).replace("__AGENT_VERSION__", platform_settings.agent.version)
    logger.info(f"Applying agent runtime policy from {_AGENT_POLICY_FILE}")
    pool.execute_script(agent_policy_sql)
    logger.info(
        f"Applied agent policy: agent={platform_settings.agent.id} "
        "role=support_agent authority_rank=4 max_refund=$50.00 "
        "refund_budget=$4,000.00/24h "
        "permissions=process_refund, queue_for_approval, read_orders"
    )
    pool.close()

    logger.info(
        f"Built {db_path} ({_NUM_USERS} users, {_NUM_ORDERS} orders, "
        f"order ids {_FIRST_ORDER_NUMBER}-{_FIRST_ORDER_NUMBER + _NUM_ORDERS - 1})"
    )


if __name__ == "__main__":
    main()
