"""Frozen artifacts: hashing, verification, tamper detection on read."""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from agent_platform.artifacts import ArtifactTamperedError, DirectoryArtifactStore, FrozenArtifact
from agent_platform.primitives.hashing import canonical_hash


class Thing(BaseModel):
    name: str
    size: int


class TestFrozenArtifact:
    def test_the_hash_is_the_canonical_hash_of_the_payload(self) -> None:
        artifact = FrozenArtifact.freeze("thing", "t1", Thing(name="x", size=1))
        assert artifact.content_hash == canonical_hash({"name": "x", "size": 1})
        assert artifact.verify()

    def test_matches_detects_a_changed_value(self) -> None:
        artifact = FrozenArtifact.freeze("thing", "t1", {"a": 1})
        assert artifact.matches({"a": 1})
        assert not artifact.matches({"a": 2})

    def test_it_is_immutable(self) -> None:
        artifact = FrozenArtifact.freeze("thing", "t1", {"a": 1})
        with pytest.raises(Exception):
            artifact.kind = "other"  # type: ignore[misc]

    def test_require_intact_raises_on_a_forged_payload(self) -> None:
        forged = FrozenArtifact(kind="k", artifact_id="i", payload={"a": 2}, content_hash="0")
        with pytest.raises(ArtifactTamperedError):
            forged.require_intact()


class TestDirectoryArtifactStore:
    def test_round_trip_with_rendering(self, tmp_path: Path) -> None:
        store = DirectoryArtifactStore(tmp_path)
        artifact = FrozenArtifact.freeze("thing", "t1", {"a": 1})
        path = store.write("run-1", artifact, rendered="# Thing")
        assert path.exists()
        assert store.read("run-1", "thing") == artifact
        assert store.rendered("run-1", "thing") == "# Thing"
        assert store.runs() == ["run-1"]
        assert len(store.paths("run-1")) == 2

    def test_a_file_edited_on_disk_fails_verification(self, tmp_path: Path) -> None:
        store = DirectoryArtifactStore(tmp_path)
        path = store.write("run-1", FrozenArtifact.freeze("thing", "t1", {"a": 1}))
        data = json.loads(path.read_text())
        data["payload"]["a"] = 99
        path.write_text(json.dumps(data))
        with pytest.raises(ArtifactTamperedError):
            store.read("run-1", "thing")

    def test_an_empty_root_has_no_runs(self, tmp_path: Path) -> None:
        store = DirectoryArtifactStore(tmp_path / "missing")
        assert store.runs() == []
        assert store.paths("x") == []
        assert store.rendered("x", "thing") is None
