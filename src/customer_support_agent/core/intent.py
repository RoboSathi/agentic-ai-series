"""What the customer is actually asking for (LLD §21).

Business policy, stated deterministically, and it runs *before* any
similarity score. The rule it encodes is simple and was expensive to get
wrong: **a refund is discussed when the customer raises it.** Asking
where an order is is not asking for money back, and an agent that treats
every order enquiry as a possible refund will eventually issue one.

Relying on embedding similarity for this was the earlier mistake. A
weak score is a continuum, so "is this about a refund?" became a
threshold to tune rather than a policy to state — and tuning it moves
the boundary for every phrasing at once, invisibly.

This is the same NOLLM discipline the rest of the system uses: the cheap,
deterministic, auditable answer first. Retrieval still narrows further
within whatever the intent permits; it can never widen it.
"""

import re

from pydantic import BaseModel, Field

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger, logged


class Intent(BaseModel):
    """The classified intent of one customer message."""

    name: str
    matched_phrase: str = ""
    allowed_tools: list[str] = Field(default_factory=list)

    @property
    def permits_tools(self) -> bool:
        """Whether this intent unlocks any tool at all."""
        return bool(self.allowed_tools)


class IntentClassifier:
    """Maps a customer message to the tools that message can justify."""

    def __init__(self) -> None:
        settings = get_settings().customer_support_agent
        self._intents = settings.intents
        self._default = settings.default_intent
        self._followup_phrases = [p.casefold() for p in settings.followup.phrases]
        self._reference = re.compile(settings.followup.reference_pattern)
        self._logger = get_logger("customer_support_agent")

    @logged("customer_support_agent")
    def classify(self, message: str, previous: Intent | None = None) -> Intent:
        """The intent of this message, in the context of the last one.

        Intents are ordered most specific first in config, so an explicit
        refund request wins over the order words it also contains —
        "I want a refund for order 10005" is a refund request, not an
        order enquiry that happens to mention money.

        A message that states no intent of its own continues the previous
        one. "what about 10010" is not a new topic; the customer dropped
        the subject because they had already established it, and an agent
        that forgets it between turns cannot hold a conversation.
        """
        lowered = message.casefold()
        for intent in self._intents:
            matched = next(
                (phrase for phrase in intent.phrases if phrase.casefold() in lowered), None
            )
            if matched is not None:
                return Intent(
                    name=intent.name, matched_phrase=matched, allowed_tools=list(intent.tools)
                )
        inherited = self._inherit(message, previous)
        return inherited if inherited is not None else Intent(name=self._default)

    def _inherit(self, message: str, previous: Intent | None) -> Intent | None:
        """Carry the previous intent forward when this message continues it.

        Two signals, either sufficient: a lead-in phrase ("what about"),
        or a bare reference that looks like an order id. Both mean the
        customer is still talking about the same thing.

        Inheritance never *widens* authority beyond what the earlier
        message already justified — it reuses that intent, it does not
        escalate to a stronger one.
        """
        if previous is None or not previous.permits_tools:
            return None
        lowered = message.casefold()
        has_phrase = any(phrase in lowered for phrase in self._followup_phrases)
        has_reference = bool(self._reference.search(message))
        if not (has_phrase or has_reference):
            return None
        self._logger.debug("Continuing the previous intent", intent=previous.name)
        return previous.model_copy(update={"matched_phrase": "continued"})

    def permitted_tools(self, message: str, previous: Intent | None = None) -> set[str]:
        """The bare tool names this message justifies offering."""
        return set(self.classify(message, previous).allowed_tools)

    def intent_names(self) -> list[str]:
        """Every configured intent, plus the default."""
        return [intent.name for intent in self._intents] + [self._default]
