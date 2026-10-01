"""Rubric generation: archetype template + Scope Brief, expanded deterministically.

No second interview and no model call. The user edits the result
(`rubric/editor.py`); what the generator guarantees, and tests pin:

- every user-stated criterion has at least one item;
- every known unknown has an "acknowledged and bounded" item;
- a source policy becomes source checks; a recency window becomes a recency check;
- every cited link must have resolved.
"""

from pathlib import Path
from typing import Any

import yaml

from agentic_commons.logging import instrument
from research_agent.domain.rubric import ItemOrigin, Rubric, RubricCheck, RubricItem
from research_agent.domain.scope import ScopeBrief

_ID_PREFIX = "R"


@instrument("research_agent")
class RubricGenerator:
    """Expands `templates.yaml` against a Brief."""

    def __init__(self, templates_path: Path) -> None:
        self._templates: dict[str, Any] = yaml.safe_load(templates_path.read_text())

    @property
    def rules(self) -> dict[str, Any]:
        return dict(self._templates["rules"])

    def stop_rule(self, archetype: str) -> str:
        return str(self._templates[archetype]["stop_rule"])

    def generate(self, brief: ScopeBrief) -> Rubric:
        drafts = [
            *self._template_items(brief),
            *self._criterion_items(brief),
            *self._subject_items(brief),
            *self._unknown_items(brief),
            *self._source_items(brief),
            self._findings_item(),
        ]
        weight = float(self._templates["defaults"]["weight"])
        return Rubric(
            items=tuple(
                RubricItem(item_id=f"{_ID_PREFIX}{index}", **{"weight": weight, **draft})
                for index, draft in enumerate(drafts, start=1)
            )
        )

    def _template_items(self, brief: ScopeBrief) -> list[dict[str, Any]]:
        spec = self._templates[brief.archetype]
        return [_draft(raw, "template", ()) for raw in spec.get("template", [])]

    def _criterion_items(self, brief: ScopeBrief) -> list[dict[str, Any]]:
        spec = self._templates[brief.archetype]
        return [
            _draft(raw, "criterion", (criterion,), criterion=criterion)
            for criterion in brief.criteria
            for raw in spec.get("per_criterion", [])
        ]

    def _subject_items(self, brief: ScopeBrief) -> list[dict[str, Any]]:
        spec = self._templates[brief.archetype]
        return [
            _draft(raw, "criterion", (subject,), subject=subject)
            for subject in brief.subjects
            for raw in spec.get("per_subject", [])
        ]

    def _unknown_items(self, brief: ScopeBrief) -> list[dict[str, Any]]:
        rule = self.rules["known_unknown"]
        return [
            _draft(
                rule,
                "known_unknown",
                (unknown.item_id,),
                subject=unknown.question,
                item_id=unknown.item_id,
            )
            for unknown in brief.known_unknowns
        ]

    def _source_items(self, brief: ScopeBrief) -> list[dict[str, Any]]:
        policy, drafts = brief.source_policy, []
        if policy.min_distinct_domains > 1:
            drafts.append(
                self._rule(
                    "min_distinct_domains",
                    policy.min_distinct_domains,
                    min_domains=policy.min_distinct_domains,
                )
            )
        if policy.blocked_domains:
            drafts.append(
                self._rule(
                    "blocked_domains",
                    ", ".join(policy.blocked_domains),
                    domains=list(policy.blocked_domains),
                )
            )
        if policy.primary_sources_preferred:
            drafts.append(self._rule("primary_sources", ""))
        if brief.recency_days is not None:
            drafts.append(self._rule("recency", brief.recency_days, days=brief.recency_days))
        drafts.append(self._rule("links_resolved", ""))
        return drafts

    def _findings_item(self) -> dict[str, Any]:
        """Every archetype: a report that establishes nothing is never COMPLETE.

        Without it every other item can pass vacuously — no criteria, no
        pairs to cover, no sentences to cite — and an empty report would
        be delivered as a finished one.
        """
        rule = self.rules["findings_present"]
        return _draft(rule, "template", (), value=rule["check"]["params"]["min_findings"])

    def _rule(self, name: str, value: object, **params: Any) -> dict[str, Any]:
        return _draft(self.rules[name], "source_policy", (name,), value=value, **params)


def _draft(
    raw: dict[str, Any], origin: ItemOrigin, maps_to: tuple[str, ...], **params: Any
) -> dict[str, Any]:
    """One item's fields from a template entry, with its placeholders filled."""
    text = str(raw.get("text", "")).format(**{key: value for key, value in params.items()})
    check = dict(raw["check"])
    merged = {**check.get("params", {}), **{k: v for k, v in params.items() if k != "value"}}
    draft: dict[str, Any] = {
        "text": text,
        "check": RubricCheck(kind=check["kind"], name=check["name"], params=merged),
        "threshold": float(raw["threshold"]),
        "must_pass": bool(raw["must_pass"]),
        "origin": origin,
        "maps_to": maps_to,
    }
    if "weight" in raw:
        draft["weight"] = float(raw["weight"])
    return draft
