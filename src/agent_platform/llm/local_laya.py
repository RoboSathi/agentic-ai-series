"""A Laya decision checkpoint: kept in its repository folder, loaded once per process.

Laya answers typed questions (a choice among described options) about a state. The platform
asks it two kinds of question — which intent a message is (`intent/laya.py`), and whether a
source supports a claim (`hallucination_check/confirmer.py`) — and both read the same
weights, so they share one resident copy here rather than loading ~800 MB each.

Like every local model here, the checkpoint lives inside the repository: downloaded once into
its folder, loaded from that folder ever after.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

from agent_platform.logging import get_logger

# Files a Laya checkpoint folder cannot load without.
REQUIRED_FILES = ("model.safetensors", "rl_agent_config.json")
_INSTALL = "make install"

# (checkpoint folder, device) → a loaded Laya agent.
Loader = Callable[[Path, str], Any]
# (repository, subfolder, destination folder) → the checkpoint written there.
Fetcher = Callable[[str, str, Path], None]


def checkpoint_complete(folder: Path) -> bool:
    return all((folder / name).exists() for name in REQUIRED_FILES)


def checkpoint_folder(
    folder: Path, model: str, subfolder: str, *, download: bool, fetcher: Fetcher
) -> Path:
    """The checkpoint folder, fetched first if it is missing and fetching is allowed."""
    if checkpoint_complete(folder):
        return folder
    if not download:
        raise RuntimeError(f"Laya checkpoint unavailable at {folder}")
    get_logger("llm").warning(
        "Laya checkpoint missing; downloading", model=f"{model}/{subfolder}", folder=str(folder)
    )
    fetcher(model, subfolder, folder)
    return folder


def load_laya(folder: Path, device: str) -> Any:
    """The checkpoint in this folder, loaded once per process however many callers ask.

    Every `/harness` rebuild builds new components; reloading the weights for each would
    cost seconds per toggle, for weights that never change.
    """
    return resident_laya(str(folder), device)


@cache
def resident_laya(folder: str, device: str) -> Any:
    try:
        import laya
    except ImportError as missing:
        raise RuntimeError(f"Laya is not installed: run `{_INSTALL}`") from missing
    return laya.load(folder, device=device)


def fetch_checkpoint(repository: str, subfolder: str, destination: Path) -> None:
    """Download one checkpoint of a bundle into a sibling folder, then rename it into place.

    A download that dies halfway leaves the staging folder, never a half-filled
    checkpoint folder that the next start would try to load.
    """
    from huggingface_hub import snapshot_download

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=destination.parent))
    try:
        snapshot_download(
            repo_id=repository, local_dir=str(staging), allow_patterns=[f"{subfolder}/*"]
        )
        if destination.exists():
            shutil.rmtree(destination)
        (staging / subfolder).rename(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "REQUIRED_FILES",
    "Fetcher",
    "Loader",
    "checkpoint_complete",
    "checkpoint_folder",
    "fetch_checkpoint",
    "load_laya",
    "resident_laya",
]
