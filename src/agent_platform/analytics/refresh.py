"""The mart, kept fresh by whatever process holds the observability database.

Only the process holding the DuckDB lock can read it, and while an agent runs
that is the agent's console. So the console refreshes the mart on a clock
(`analytics.export_interval_seconds`), off the event loop, and a failed export
is logged and retried next tick: a stale dashboard is better than a dead console.
"""

import asyncio

from agent_platform.analytics.exporter import MartExporter, MartManifest
from agent_platform.config import get_settings
from agent_platform.logging import get_logger
from agent_platform.observability.store import DuckDbObservabilityStore


class MartRefresher:
    """Exports the mart on demand, one export at a time, never raising."""

    def __init__(self, exporter: MartExporter | None = None) -> None:
        self._exporter = exporter
        self._lock = asyncio.Lock()
        self._logger = get_logger("analytics")

    @staticmethod
    def interval_seconds() -> float:
        """How often to refresh; 0 when observability or the periodic export is off."""
        settings = get_settings()
        if not settings.observability.enabled:
            return 0.0
        return settings.analytics.export_interval_seconds

    async def refresh(self) -> MartManifest | None:
        """One export in a worker thread; None when it failed (and was logged)."""
        async with self._lock:
            try:
                return await asyncio.to_thread(self._export)
            except Exception as exc:  # noqa: BLE001 - logged; the next tick retries
                self._logger.warning("Analytics mart export failed", error=str(exc))
                return None

    def _export(self) -> MartManifest:
        if self._exporter is None:
            self._exporter = MartExporter(DuckDbObservabilityStore())
        return self._exporter.export()


__all__ = ["MartRefresher"]
