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

from agentic_commons.config import get_settings
from agentic_commons.evals.types import EvalManifestEntry, PyramidLevel


class EvalManifest(BaseModel):
    """Every declared eval, queryable by component and pyramid level."""

    entries: list[EvalManifestEntry] = Field(default_factory=list)

    def for_component(self, component: str) -> list[EvalManifestEntry]:
        """Entries declared for one component."""
        return [entry for entry in self.entries if entry.component == component]

    def for_level(self, level: PyramidLevel) -> list[EvalManifestEntry]:
        """Entries belonging to one pyramid level."""
        return [entry for entry in self.entries if entry.level == level]

    def threshold_for(self, component: str, default: float = 0.0) -> float:
        """Strictest regression threshold declared for a component."""
        thresholds = [entry.regression_threshold for entry in self.for_component(component)]
        return max(thresholds) if thresholds else default

    def components(self) -> list[str]:
        """Every component with at least one declared eval."""
        return sorted({entry.component for entry in self.entries})

    def missing_coverage(self, expected_components: list[str]) -> list[str]:
        """Components that ship without a declared eval — the gap report."""
        declared = set(self.components())
        return [name for name in expected_components if name not in declared]


def load_manifest(path: Path | None = None) -> EvalManifest:
    """Load the manifest from its configured, package-relative path."""
    manifest_path = path or get_settings().evals.resolved_manifest_path
    if not manifest_path.exists():
        return EvalManifest()
    raw: dict[str, Any] = yaml.safe_load(manifest_path.read_text()) or {}
    return EvalManifest(entries=[EvalManifestEntry(**entry) for entry in raw.get("evals", [])])
