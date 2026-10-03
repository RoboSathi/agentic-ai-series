"""Where frozen artifacts live: one directory per run, one file per kind.

Plain files on purpose. An archived artifact is meant to be read by a
person (the rendered companion) and checked by a machine (the JSON with
its hash) long after the process that wrote it has gone; neither needs a
database to do that.

    <root>/<run_id>/<kind>.json   the frozen artifact, hash included
    <root>/<run_id>/<kind>.md     an optional human-readable rendering
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

from agent_platform.artifacts.types import ArtifactTamperedError, FrozenArtifact
from agent_platform.logging import get_logger

_JSON = ".json"
_RENDERED = ".md"


@runtime_checkable
class ArtifactStore(Protocol):
    """Writes and reads frozen artifacts, verifying each on the way out."""

    def write(self, run_id: str, artifact: FrozenArtifact, rendered: str | None = None) -> Path: ...

    def read(self, run_id: str, kind: str) -> FrozenArtifact: ...

    def runs(self) -> list[str]: ...


class DirectoryArtifactStore:
    """File-backed store under one root directory."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._logger = get_logger("artifacts")

    @property
    def root(self) -> Path:
        return self._root

    def run_dir(self, run_id: str) -> Path:
        return self._root / run_id

    def write(self, run_id: str, artifact: FrozenArtifact, rendered: str | None = None) -> Path:
        """Persist an intact artifact (and its rendering); return the JSON path."""
        artifact.require_intact()
        folder = self.run_dir(run_id)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{artifact.kind}{_JSON}"
        path.write_text(artifact.model_dump_json(indent=2))
        if rendered is not None:
            (folder / f"{artifact.kind}{_RENDERED}").write_text(rendered)
        self._logger.info(
            "Artifact written", run=run_id, kind=artifact.kind, hash=artifact.content_hash
        )
        return path

    def read(self, run_id: str, kind: str) -> FrozenArtifact:
        """Load one artifact, raising `ArtifactTamperedError` if its hash no longer holds."""
        path = self.run_dir(run_id) / f"{kind}{_JSON}"
        artifact = FrozenArtifact.model_validate_json(path.read_text())
        if not artifact.verify():
            self._logger.error("Artifact failed verification", run=run_id, kind=kind)
            raise ArtifactTamperedError(f"{path} does not match its frozen hash")
        return artifact

    def rendered(self, run_id: str, kind: str) -> str | None:
        path = self.run_dir(run_id) / f"{kind}{_RENDERED}"
        return path.read_text() if path.exists() else None

    def paths(self, run_id: str) -> list[Path]:
        folder = self.run_dir(run_id)
        return sorted(folder.iterdir()) if folder.exists() else []

    def runs(self) -> list[str]:
        """Every run with at least one artifact, oldest directory first."""
        if not self._root.exists():
            return []
        folders = [path for path in self._root.iterdir() if path.is_dir()]
        return [path.name for path in sorted(folders, key=lambda path: path.stat().st_mtime)]
