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

An encoder *does* run, but strictly second and strictly bounded. When no
configured phrase matches, the message is embedded and compared against
the phrase sets of the intents config marks `semantic: true` — so a
paraphrase like "any news on my parcel?" is recognised as an order
enquiry, while `refund_request` stays on the deterministic path where a
policy is a sentence rather than a threshold. An embedding model, never
a completion: classification is a similarity question.

This is the same NOLLM discipline the rest of the system uses: the cheap,
deterministic, auditable answer first. Retrieval still narrows further
within whatever the intent permits; it can never widen it.
"""

import re

from pydantic import BaseModel, Field

from agentic_commons.llm.embeddings import (
    EmbeddingClient,
    cosine_similarity,
    create_embedding_client,
)
from agentic_commons.logging import get_logger, logged
from agentic_commons.observability import traced
from customer_support_agent.config import get_agent_settings
from customer_support_agent.config.settings import IntentSettings


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

    def __init__(self, embeddings: EmbeddingClient | None = None) -> None:
        settings = get_agent_settings()
        self._intents = settings.intents
        self._default = settings.default_intent
        self._followup_phrases = [p.casefold() for p in settings.followup.phrases]
        self._reference = re.compile(settings.followup.reference_pattern)
        self._semantic = settings.semantic_intent
        self._embeddings = embeddings
        # Cached per phrase set, because the phrases never change within a
        # process and re-embedding a fixed vocabulary on every message is
        # a network call per turn for an answer that cannot have moved.
        self._phrase_vectors: dict[str, list[list[float]]] = {}
        self._logger = get_logger("customer_support_agent")

    @traced("intent_classification", subsystem="customer_support_agent")
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

    async def classify_async(self, message: str, previous: Intent | None = None) -> Intent:
        """Classify deterministically, then fall back to the encoder.

        The order is the whole design. A configured phrase is a stated
        policy and wins outright; only a message that matched nothing is
        put to an embedding model, and only intents marked `semantic` in
        config can be reached that way. So the encoder can *recognise* a
        paraphrase of an order enquiry, and can never quietly decide that
        a sentence was about money.

        This is an embedding call, not a completion: classification is a
        similarity question, and generation is the wrong instrument for
        it — slower, dearer, and less stable across runs.
        """
        deterministic = self.classify(message, previous)
        if deterministic.name != self._default or not self._semantic.enabled:
            return deterministic
        matched = await self._semantic_match(message)
        return matched if matched is not None else deterministic

    async def _semantic_match(self, message: str) -> Intent | None:
        """The best semantically similar intent, if one clears the bar."""
        text = message.strip()
        if not text or len(text.split()) > self._semantic.max_words:
            return None
        candidates = [intent for intent in self._intents if intent.semantic and intent.phrases]
        if not candidates:
            return None
        client = self._embedding_client()
        try:
            query = (await client.embed([text]))[0]
        except Exception as exc:
            # A classifier that cannot reach its encoder degrades to the
            # deterministic answer. It does not fail the customer's turn.
            self._logger.warning("Semantic intent matching unavailable", error=str(exc))
            return None
        best, score = await self._best_match(client, query, candidates)
        if best is None or score < self._semantic.threshold:
            self._logger.debug("No intent cleared the similarity bar", best_score=score)
            return None
        self._logger.info("Intent matched semantically", intent=best.name, score=score)
        return Intent(
            name=best.name,
            matched_phrase=f"~{score:.2f} similarity",
            allowed_tools=list(best.tools),
        )

    async def _best_match(
        self, client: EmbeddingClient, query: list[float], candidates: list[IntentSettings]
    ) -> tuple[IntentSettings | None, float]:
        """The highest-scoring intent and its score."""
        best: IntentSettings | None = None
        best_score = 0.0
        for intent in candidates:
            for vector in await self._vectors_for(client, intent):
                score = cosine_similarity(query, vector)
                if score > best_score:
                    best, best_score = intent, score
        return best, best_score

    async def _vectors_for(
        self, client: EmbeddingClient, intent: IntentSettings
    ) -> list[list[float]]:
        """This intent's phrase vectors, embedded once per process."""
        if intent.name not in self._phrase_vectors:
            self._phrase_vectors[intent.name] = await client.embed(list(intent.phrases))
        return self._phrase_vectors[intent.name]

    def _embedding_client(self) -> EmbeddingClient:
        """The encoder, built on first use.

        Lazily, because most turns never reach it: a classifier built at
        startup would open a client for a path the deterministic rules
        usually settle.
        """
        if self._embeddings is None:
            self._embeddings = create_embedding_client()
        return self._embeddings

    def permitted_tools(self, message: str, previous: Intent | None = None) -> set[str]:
        """The bare tool names this message justifies offering."""
        return set(self.classify(message, previous).allowed_tools)

    def intent_names(self) -> list[str]:
        """Every configured intent, plus the default."""
        return [intent.name for intent in self._intents] + [self._default]
