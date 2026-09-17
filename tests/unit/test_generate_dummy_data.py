"""Unit tests for the dummy data generator."""

from pathlib import Path

import pytest

from customer_support_agent.dummy_data import generate_dummy_data as gen


def test_sql_str_escapes_embedded_quotes() -> None:
    assert gen._sql_str("O'Brien") == "'O''Brien'"


def test_weighted_choice_only_returns_given_values() -> None:
    import random

    rng = random.Random(1)
    weights = [("a", 0.9), ("b", 0.1)]

    results = {gen._weighted_choice(rng, weights) for _ in range(50)}

    assert results <= {"a", "b"}


def test_generate_seed_sql_produces_expected_row_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gen, "_NUM_USERS", 5)
    monkeypatch.setattr(gen, "_NUM_ORDERS", 10)

    sql = gen._generate_seed_sql()

    assert sql.count("INSERT INTO users") == 5
    assert sql.count("INSERT INTO orders") == 10


def test_main_writes_seed_sql_and_builds_a_real_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gen, "_NUM_USERS", 3)
    monkeypatch.setattr(gen, "_NUM_ORDERS", 7)
    seed_file = tmp_path / "seed_data.sql"
    db_path = tmp_path / "agentic.db"
    monkeypatch.setattr(gen, "_SEED_DATA_FILE", seed_file)
    monkeypatch.setattr(gen, "DB_PATH", db_path)

    gen.main()

    assert seed_file.exists()
    assert db_path.exists()

    from agentic_commons.persistence.database import Database

    database = Database(db_path=db_path)
    assert database.execute("SELECT COUNT(*) AS c FROM users").rows[0]["c"] == 3
    assert database.execute("SELECT COUNT(*) AS c FROM orders").rows[0]["c"] == 7


def test_main_is_idempotent_rerunning_does_not_duplicate_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gen, "_NUM_USERS", 3)
    monkeypatch.setattr(gen, "_NUM_ORDERS", 4)
    monkeypatch.setattr(gen, "_SEED_DATA_FILE", tmp_path / "seed_data.sql")
    monkeypatch.setattr(gen, "DB_PATH", tmp_path / "agentic.db")

    gen.main()
    gen.main()

    from agentic_commons.persistence.database import Database

    database = Database(db_path=tmp_path / "agentic.db")
    assert database.execute("SELECT COUNT(*) AS c FROM users").rows[0]["c"] == 3
    assert database.execute("SELECT COUNT(*) AS c FROM orders").rows[0]["c"] == 4
