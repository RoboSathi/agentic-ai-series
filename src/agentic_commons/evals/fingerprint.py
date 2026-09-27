"""The agent fingerprint: "would this agent answer differently now?"

Hashed from what is on disk and in effect, never from git. A commit hash
would let an uncommitted fix reuse the transcripts of the code it fixed —
the exact case a pre-commit eval exists to catch. Reverting the edit
restores the old fingerprint, and with it the cached transcripts.

The parts are kept, not just their hash, so a report can say *what*
changed ("sources changed, config unchanged") when a rerun regenerates.
"""

import hashlib
import subprocess
from pathlib import Path

from pydantic import BaseModel

from agentic_commons.config import REPO_ROOT, get_settings
from agentic_commons.config.settings import EvalFingerprintSettings
from agentic_commons.evals.contracts import canonical_hash


class AgentFingerprint(BaseModel):
    """The combined hash and the parts it was built from."""

    value: str
    parts: dict[str, str]
    git_sha: str
    git_dirty: bool

    def changed_parts(self, previous: dict[str, str]) -> list[str]:
        """Names of the parts that differ from an earlier fingerprint."""
        names = set(self.parts) | set(previous)
        return sorted(name for name in names if self.parts.get(name) != previous.get(name))


def compute_fingerprint(
    agent_inputs: dict[str, str],
    settings: EvalFingerprintSettings | None = None,
    root: Path = REPO_ROOT,
) -> AgentFingerprint:
    """Fingerprint the working tree, the effective config and the agent's own inputs."""
    config = settings or get_settings().evals.fingerprint
    parts = {
        "sources": source_hash(config, root),
        "config": config_hash(config),
        **{f"agent.{name}": value for name, value in sorted(agent_inputs.items())},
    }
    sha, dirty = _git_state(root)
    return AgentFingerprint(value=canonical_hash(parts), parts=parts, git_sha=sha, git_dirty=dirty)


def source_hash(settings: EvalFingerprintSettings, root: Path) -> str:
    """Content hash of every behaviour-bearing source file, tracked or not."""
    entries = [
        (str(path.relative_to(root)), _file_digest(path)) for path in _source_files(settings, root)
    ]
    return canonical_hash(entries)


def config_hash(settings: EvalFingerprintSettings) -> str:
    """Hash of the effective settings — YAML layers and env overrides alike."""
    dumped = get_settings().model_dump(mode="json")
    relevant = {
        section: value
        for section, value in dumped.items()
        if section not in settings.exclude_config_sections
    }
    return canonical_hash(relevant)


def _source_files(settings: EvalFingerprintSettings, root: Path) -> list[Path]:
    suffixes = set(settings.include_suffixes)
    skipped_parts = set(settings.exclude_parts)
    skipped_paths = [root / entry for entry in settings.exclude_paths]
    files = [
        path
        for source in settings.source_roots
        for path in (root / source).rglob("*")
        if path.is_file()
        and path.suffix in suffixes
        and not skipped_parts.intersection(path.parts)
        and not any(path == skip or skip in path.parents for skip in skipped_paths)
    ]
    return sorted(files)


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_state(root: Path) -> tuple[str, bool]:
    """Commit and dirty flag — recorded for the report, never for cache hits."""
    try:
        sha = _git(root, "rev-parse", "HEAD")
        dirty = bool(_git(root, "status", "--porcelain"))
    except (OSError, subprocess.CalledProcessError):
        return "", False
    return sha, dirty


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return completed.stdout.strip()
