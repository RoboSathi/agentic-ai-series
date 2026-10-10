"""The eval manifest — a checked-in test plan, not a spreadsheet (E09).

Every component that can regress declares here what it is evaluated on,
by which scorer, and how far it is allowed to fall before CI calls it a
regression. A component with no manifest entry has no regression signal,
and the manifest makes that absence visible instead of implicit.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from agent_platform.config import get_settings
from agent_platform.evals.types import EvalManifestEntry


class EvalManifest(BaseModel):
    """Every declared eval, queryable by component and pyramid level."""

    entries: list[EvalManifestEntry] = Field(default_factory=list)

    def for_component(self, component: str) -> list[EvalManifestEntry]:
        """Entries declared for one component."""
        return [entry for entry in self.entries if entry.component == component]

    def threshold_for(self, component: str, default: float = 0.0) -> float:
        """Strictest regression threshold declared for a component."""
        thresholds = [entry.regression_threshold for entry in self.for_component(component)]
        return max(thresholds) if thresholds else default

    def components(self) -> list[str]:
        """Every component with at least one declared eval."""
        return sorted({entry.component for entry in self.entries})


def load_manifest(path: Path | None = None) -> EvalManifest:
    """Load the manifest, merging every declared source in order.

    The platform declares the evals every agent's harness can regress
    on; an agent declares the ones only it can. Reading both here — and
    concatenating rather than overwriting — is what lets a golden-task
    eval mention refunds without the shared manifest mentioning them.
    """
    paths = [path] if path is not None else get_settings().evals.resolved_manifest_paths
    return EvalManifest(entries=[entry for source in paths for entry in _entries(source)])


def _entries(path: Path) -> list[EvalManifestEntry]:
    """One manifest file's entries, or none when it is not there."""
    if not path.exists():
        return []
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    return [EvalManifestEntry(**entry) for entry in raw.get("evals", [])]
