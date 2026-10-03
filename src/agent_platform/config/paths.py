"""Package-relative path resolution.

Every path in config is relative to the repository root, resolved from this
package's own location — never from the shell's working directory, so the
same config works regardless of where a process is started.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(value: str | Path) -> Path:
    """Resolve a configured path against the repo root if it is relative."""
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path
