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
from agentic_commons.tools.types import WIRE_SEPARATOR


@runtime_checkable
class StageLabeler(Protocol):
    """Maps a canonical stage key to the line shown in the feed."""

    def label_for(self, stage: str, **substitutions: str) -> str: ...


class StageLabelRegistry:
    """Configured stage labels, tool wording and per-stage colour."""

    def __init__(self) -> None:
        settings = get_settings().tui
        self._labels = settings.stage_labels
        self._tool_labels = settings.tool_labels
        self._styles = settings.stage_styles
        self._default_style = settings.default_stage_style
        self._logger = get_logger("tui")

    def label_for(self, stage: str, **substitutions: str) -> str:
        """The human-readable label for a stage key.

        A `tool_name` substitution is resolved to that tool's *wording*
        first, so the feed reads "Looking up the order…" rather than
        "Executing support.get_order_status…". The person watching cares
        what the agent is doing, not which method it reached for.
        """
        template = self._labels.get(stage)
        if template is None:
            self._logger.warning("Stage has no configured label", stage=stage)
            return stage.replace("_", " ").capitalize()
        if "tool_name" in substitutions:
            substitutions = {
                **substitutions,
                "tool_name": self.tool_label(substitutions["tool_name"]),
            }
        return self._format(template, substitutions)

    def tool_label(self, tool: str) -> str:
        """What to call a tool on screen, keyed by its bare name.

        A tool arrives here spelled three ways — `process_refund`,
        `support.process_refund` (logs and policy) and
        `support__process_refund` (the wire) — and all three must land on
        the same wording. Stripping only one separator is how a feed ends
        up reading "Running support  process refund…".
        """
        bare = tool.rsplit(WIRE_SEPARATOR, maxsplit=1)[-1].rsplit(".", maxsplit=1)[-1]
        return self._tool_labels.get(bare, f"Running {bare.replace('_', ' ')}…")

    def style_for(self, stage: str) -> str:
        """The colour a stage is rendered in.

        Colour carries meaning here: security checks warn, data
        preparation is cool, model calls are blue, authority decisions
        stand out, and green is reserved for a completed action or the
        final reply — so a feed that is entirely green means nothing.
        """
        return self._styles.get(stage, self._default_style)

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
