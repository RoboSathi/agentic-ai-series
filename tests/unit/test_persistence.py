"""Unit tests for the real SQLite-backed persistence layer."""

from pathlib import Path

from agentic_commons.persistence.database import Database


def test_get_connection_returns_real_sqlite_connection(tmp_path: Path) -> None:
    """Test that get_connection returns a working sqlite3 connection."""
    db = Database(db_path=tmp_path / "test.db")

    connection = db.get_connection()
    try:
        connection.execute("SELECT 1")
    finally:
        connection.close()


def test_execute_creates_table_and_inserts_row(tmp_path: Path) -> None:
    """Test that execute() actually persists data to the SQLite file."""
    db = Database(db_path=tmp_path / "test.db")

    db.execute("CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT)")
    result = db.execute("INSERT INTO widgets (name) VALUES (?)", ("sprocket",))

    assert result.affected == 1
    assert result.rows == []  # INSERT has no result rows


def test_execute_select_returns_rows_as_dicts(tmp_path: Path) -> None:
    """Test that SELECT results come back as one dict per row."""
    db = Database(db_path=tmp_path / "test.db")
    db.execute("CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT)")
    db.execute("INSERT INTO widgets (name) VALUES (?)", ("sprocket",))

    result = db.execute("SELECT id, name FROM widgets")

    assert result.rows == [{"id": 1, "name": "sprocket"}]


def test_execute_persists_across_separate_calls(tmp_path: Path) -> None:
    """Test that data survives across calls, each opening its own connection."""
    db_path = tmp_path / "test.db"
    db1 = Database(db_path=db_path)
    db1.execute("CREATE TABLE widgets (id INTEGER PRIMARY KEY, name TEXT)")
    db1.execute("INSERT INTO widgets (name) VALUES (?)", ("sprocket",))

    db2 = Database(db_path=db_path)
    result = db2.execute("SELECT name FROM widgets")

    assert result.rows == [{"name": "sprocket"}]


def test_default_db_path_is_package_relative_not_cwd() -> None:
    """Test that the default db path doesn't depend on the process's cwd."""
    db = Database()
    assert db._db_path.is_absolute()
    assert db._db_path.parent.name == "data"
