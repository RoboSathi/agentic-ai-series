"""Incident postmortems, in a fixed shape (E37, FR-46).

The template is not a formality. Every field here is one an incident
review skips when it is written freehand, and each omission has a
predictable consequence:

- **impact** — skipped, and severity becomes a matter of who is telling
  the story;
- **trigger** — skipped, and the fix addresses the symptom;
- **missing control** — skipped, and the same class of incident recurs
  behind a different trigger;
- **detection gap** — skipped, and the next one is again found by a
  customer;
- **containment** — skipped, and nobody learns how long the bleeding
  lasted;
- **root cause** — skipped, and the action items are cosmetic;
- **permanent eval** — skipped, and the whole exercise is a document.

The last one is the load-bearing field. An incident that does not end in
a regression eval has not been fixed; it has been survived. So a
postmortem without one is *incomplete* here, and says so — this module
refuses to pretend otherwise, which is the only enforcement a template
can honestly offer.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from agent_platform.logging import get_logger, instrument
from agent_platform.observability.types import utc_now


class Postmortem(BaseModel):
    """One incident, in the fixed template."""

    incident_id: str
    title: str
    occurred_at: datetime = Field(default_factory=utc_now)
    # What it did to whom. Not "the agent misbehaved" — how many
    # customers, how much money, which promise was broken.
    impact: str = ""
    # The proximate event that started it.
    trigger: str = ""
    # The control that would have stopped it and was not there. This is
    # the field that generalises: the trigger is specific, the missing
    # control is the class.
    missing_control: str = ""
    # How it was noticed, and how long that took.
    detection_gap: str = ""
    # What stopped the bleeding, and when.
    containment: str = ""
    # Why the missing control was missing.
    root_cause: str = ""
    # The eval that now fails if this recurs, by id.
    permanent_eval: str = ""
    slo_breached: str = ""

    @property
    def required_fields(self) -> tuple[str, ...]:
        return (
            "impact",
            "trigger",
            "missing_control",
            "detection_gap",
            "containment",
            "root_cause",
            "permanent_eval",
        )

    @property
    def missing(self) -> list[str]:
        """Template fields nobody filled in."""
        return [field for field in self.required_fields if not getattr(self, field).strip()]

    @property
    def complete(self) -> bool:
        """Whether this postmortem is finished.

        Completeness includes the regression eval. An incident review
        that ends without one has produced a document, not a control.
        """
        return not self.missing


@instrument("observability")
class PostmortemRegister:
    """Every postmortem, and what is still outstanding.

    In memory, deliberately: incidents are few, and the durable record is
    the eval each one produces plus the audit trail it was written from.
    A second database for prose would be a second thing to migrate.
    """

    def __init__(self) -> None:
        self._postmortems: dict[str, Postmortem] = {}
        self._logger = get_logger("observability")

    def record(self, postmortem: Postmortem) -> Postmortem:
        """File a postmortem, complete or not."""
        self._postmortems[postmortem.incident_id] = postmortem
        if postmortem.complete:
            self._logger.info(
                "Postmortem filed",
                incident=postmortem.incident_id,
                eval=postmortem.permanent_eval,
            )
        else:
            self._logger.warning(
                "Postmortem is incomplete",
                incident=postmortem.incident_id,
                missing=postmortem.missing,
            )
        return postmortem

    def get(self, incident_id: str) -> Postmortem | None:
        return self._postmortems.get(incident_id)

    def all(self) -> list[Postmortem]:
        """Every postmortem, newest first."""
        return sorted(self._postmortems.values(), key=lambda p: p.occurred_at, reverse=True)

    def outstanding(self) -> list[Postmortem]:
        """Postmortems still missing a field — the review backlog."""
        return [postmortem for postmortem in self.all() if not postmortem.complete]

    def regression_evals(self) -> list[str]:
        """Every eval an incident has produced.

        The series' actual claim about learning from failure: this list
        should only ever grow, and each entry should be runnable.
        """
        return [p.permanent_eval for p in self.all() if p.permanent_eval]
