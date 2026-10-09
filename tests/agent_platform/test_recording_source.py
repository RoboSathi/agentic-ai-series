"""A running agent records to the live source; `SOURCE` moves only the readers.

`make agent csa SOURCE=dummy` used to write that console's facts and audit into
the demo world, because one setting chose both where agents write and where
readers read.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_platform.analytics import cli
from agent_platform.analytics.refresh import ParquetRefresher
from agent_platform.config import get_settings, reload_settings
from agent_platform.observability.store import DuckDbObservabilityStore


@pytest.fixture
def sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Readers pointed at `dummy`; both sources in tmp."""
    for name in ("live", "dummy"):
        prefix = f"ANALYTICS__SOURCES__{name.upper()}__"
        monkeypatch.setenv(prefix + "DB_PATH", str(tmp_path / name / "obs.duckdb"))
        monkeypatch.setenv(prefix + "PARQUET_DIR", str(tmp_path / name / "parquet"))
    monkeypatch.setenv("ANALYTICS__SOURCE", "dummy")
    reload_settings()
    yield tmp_path
    monkeypatch.undo()
    reload_settings()


def test_an_agents_store_opens_the_live_source(sources: Path) -> None:
    store = DuckDbObservabilityStore()
    assert Path(store._db_path) == sources / "live" / "obs.duckdb"


def test_the_export_command_reads_the_source_readers_were_pointed_at(sources: Path) -> None:
    store = cli._store()
    assert store is not None
    assert Path(store._db_path) == sources / "dummy" / "obs.duckdb"


def test_a_running_console_exports_what_it_records(sources: Path) -> None:
    ParquetRefresher()._export()
    live = get_settings().analytics.recording.resolved_parquet_dir
    assert (live / get_settings().analytics.manifest_file).is_file()
    assert not (sources / "dummy" / "parquet").exists()
