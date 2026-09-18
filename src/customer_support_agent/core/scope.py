"""What this agent will and will not talk about (LLD §21).

A customer support agent is not a general assistant. Asked for the
weather, to write code, or who the prime minister is, it should decline —
and decline *before* calling a model, because the refusal is the same
every time and paying a model to produce it is pure waste.

The rule is default-deny. A message is answered when it classifies into a
known support topic or is ordinary conversational courtesy; anything else
is refused. An allow-list is the only shape that holds: a deny-list of
off-topic subjects is a list of the things somebody thought of, and the
next question is always the one nobody thought of.

This runs first in the pipeline, so an off-topic message costs one regex
pass rather than a completion.
"""

import re

from pydantic import BaseModel

from agentic_commons.logging import get_logger, logged
from customer_support_agent.config import get_agent_settings


class ScopeVerdict(BaseModel):
    """Whether a message is this agent's business."""

    in_scope: bool
    topic: str = ""
    refusal: str = ""


class ScopeGuard:
    """Default-deny topic gate, evaluated before any model call."""

    def __init__(self) -> None:
        settings = get_agent_settings()
        self._topics = settings.scope.in_scope_topics
        self._refusal = settings.scope.refusal_message
        self._min_words_for_refusal = settings.scope.min_words_before_refusal
        self._reference = re.compile(settings.scope.reference_pattern)
        self._patterns = {
            topic: re.compile("|".join(re.escape(p) for p in phrases), re.IGNORECASE)
            for topic, phrases in self._topics.items()
            if phrases
        }
        self._logger = get_logger("customer_support_agent")

    @logged("customer_support_agent")
    def check(self, message: str) -> ScopeVerdict:
        """Classify a message as in-scope support talk, or refuse it."""
        text = message.strip()
        if not text:
            return ScopeVerdict(in_scope=True, topic="empty")
        for topic, pattern in self._patterns.items():
            if pattern.search(text):
                return ScopeVerdict(in_scope=True, topic=topic)
        if self._reference.search(text):
            # A bare order reference continues the conversation. Refusing
            # "what about 10002" because it names no topic would break
            # every follow-up.
            return ScopeVerdict(in_scope=True, topic="reference")
        # A very short message with no topic is usually courtesy ("ok",
        # "thanks") rather than an off-topic request, and refusing those
        # reads as hostile. Anything longer is a real question about
        # something else.
        if len(text.split()) < self._min_words_for_refusal:
            return ScopeVerdict(in_scope=True, topic="courtesy")
        self._logger.info("Refusing an out-of-scope request", words=len(text.split()))
        return ScopeVerdict(in_scope=False, topic="out_of_scope", refusal=self._refusal)
