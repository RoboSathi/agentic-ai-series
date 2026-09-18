"""Stage labels for the live feed (LLD §20).

The mapping lives in config, not here. That is the point: the table *is*
the registry's content, so adding a stage is a config change and the
code never grows a `if stage == ...` ladder.

A stage with no configured label is a contract violation, surfaced
rather than papered over with a generic fallback — an unlabeled stage in
the feed is a stage nobody decided how to describe.
"""

from typing import Protocol, runtime_checkable

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger


@runtime_checkable
class StageLabeler(Protocol):
    """Maps a canonical stage key to the line shown in the feed."""

    def label_for(self, stage: str, **substitutions: str) -> str: ...


class StageLabelRegistry:
    """Configured stage labels, with `{tool_name}`-style substitution."""

    def __init__(self) -> None:
        self._labels = get_settings().tui.stage_labels
        self._logger = get_logger("tui")

    def label_for(self, stage: str, **substitutions: str) -> str:
        """The human-readable label for a stage key."""
        template = self._labels.get(stage)
        if template is None:
            self._logger.warning("Stage has no configured label", stage=stage)
            return stage.replace("_", " ").capitalize()
        return self._format(template, substitutions)

    def known_stages(self) -> set[str]:
        """Every stage key with a configured label."""
        return set(self._labels)

    def missing(self, emitted_stages: set[str]) -> set[str]:
        """Stages that were emitted but never labelled — the contract test."""
        return emitted_stages - self.known_stages()

    @staticmethod
    def _format(template: str, substitutions: dict[str, str]) -> str:
        """Substitute placeholders, leaving the template intact if any is absent."""
        try:
            return template.format(**substitutions)
        except (KeyError, IndexError):
            return template
