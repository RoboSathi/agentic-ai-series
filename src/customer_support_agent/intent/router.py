"""Customer Support intent routing.

The router has one job at the front door:

1. deterministically reject requests that are *provably* outside Customer
   Support;
2. otherwise classify into the agent's small intent set with a lightweight
   BERT-family model;
3. if the classifier is not confident enough, ask one deterministic
   clarification question.

The router does not authorize tools.  The selected intent only determines
which business workflow/tools may be considered later; the Gatekeeper still
controls every side effect.
"""

from __future__ import annotations

import re
from typing import Any

from agent_platform.config import get_settings
from agent_platform.intent import (
    IntentAssessment,
    IntentCandidate,
    IntentClassifierInterface,
    IntentDisposition,
    create_intent_classifier,
)
from agent_platform.logging import get_logger
from agent_platform.observability import traced
from customer_support_agent.config import get_agent_settings
from customer_support_agent.intent.phrases import PhraseReader


class Intent:
    """Backwards-compatible lightweight view used by the planner."""

    def __init__(self, assessment: IntentAssessment) -> None:
        self.name = assessment.intent
        self.allowed_tools = assessment.allowed_tools
        self.confidence = assessment.confidence
        self.disposition = assessment.disposition
        self.source = assessment.source
        self.matched_phrase = assessment.matched_rule
        self.candidates = assessment.candidates
        self.clarification = assessment.clarification
        self.reason = assessment.reason
        self.model = assessment.model
        self.score_type = assessment.score_type


class CustomerSupportIntentRouter:
    """One unified Customer Support intent + capability router."""

    def __init__(self, backend: IntentClassifierInterface | None = None) -> None:
        self._app_config = get_settings()
        settings = get_agent_settings()
        self._settings = settings
        self._intents = {item.name: item for item in settings.intents}
        self._default = settings.default_intent
        self._followup_phrases = [phrase.casefold() for phrase in settings.followup.phrases]
        self._phrases = PhraseReader(settings.intent_router.negation)
        self._reference = re.compile(settings.followup.reference_pattern)
        self._logger = get_logger("customer_support_agent")
        # The composition root supplies the agent-owned router; the router
        # immediately wires the shared classifier provider. Model weights are
        # still loaded asynchronously on first semantic classification.
        self._backend = backend or create_intent_classifier(**intent_choices())
        self._backend_error: str = ""
        self._backend_ready = bool(getattr(self._backend, "is_loaded", False))
        self._phrase_rules = settings.intent_router.phrase_rules
        # The classifier in use, and its threshold and labels, read once.
        self._provider = get_settings().llm.intent_classifier.provider
        self._routing = settings.intent_router.routing(self._provider)
        self._reject_patterns = [
            (rule.rule_id, re.compile(rule.pattern, re.IGNORECASE), rule.reason)
            for rule in settings.intent_router.deterministic_rejects
        ]

    @property
    def model_name(self) -> str:
        return self._backend.model_name

    @property
    def _source(self) -> str:
        """What a model-decided intent names as its source: the configured provider."""
        return self._provider

    @property
    def model_ready(self) -> bool:
        return self._backend_ready

    async def ensure_ready(self) -> None:
        """Load the configured classifier once; surface failures rather than hiding them."""
        if self._backend_ready:
            return
        if self._backend_error:
            raise RuntimeError(self._backend_error)
        try:
            await self._backend.ensure_loaded()
            self._backend_ready = True
            cache_path = getattr(self._backend, "cache_path", None)
            self._logger.info(
                "Customer Support intent classifier ready",
                model=self._backend.model_name,
                cache_path=str(cache_path) if cache_path is not None else "provider-managed",
            )
        except Exception as exc:
            self._backend_error = f"{type(exc).__name__}: {exc}"
            self._logger.error(
                "Customer Support intent classifier failed to load",
                model=self._backend.model_name,
                error=self._backend_error,
            )
            raise

    @traced("intent_classification", subsystem="customer_support_agent")
    async def classify_async(
        self,
        message: str,
        previous: Intent | None = None,
    ) -> Intent:
        """Assess intent and whether Customer Support can serve the request."""
        text = message.strip()
        if not text:
            return self._supported_unknown("Empty customer message")

        deterministic = self._deterministic_rejection(text)
        if deterministic is not None:
            return deterministic

        # IMPORTANT: previous intent is context, not a classification result.
        # Always classify the current turn first. The previous intent is used
        # only as a constrained fallback when the current turn is genuinely
        # underspecified. This prevents a follow-up such as "what about a
        # refund?" from being forced into the previous intent.
        # Courtesy/greeting messages are intentionally NOT terminal deterministic
        # UNKNOWN results. If no deterministic rule recognizes them, they must
        # reach the semantic classifier just like any other unrecognized turn.
        # This preserves the deterministic -> BERT fallback contract.

        # High-precision current-turn patterns are deterministic routing rules,
        # not a substitute for the semantic classifier. They prevent a small
        # classifier model from turning an obvious support request into UNKNOWN
        # and let the graph make a safe capability decision without an LLM.
        deterministic_supported = self._deterministic_supported(text)
        if deterministic_supported is not None:
            return deterministic_supported
        return self._unless_refused(text, await self._from_classifier(text, previous))

    def _unless_refused(self, text: str, intent: Intent) -> Intent:
        """The classifier's intent, unless the customer said they do not want it.

        "I don't want a refund for 10487" reads to a classifier like a refund request.
        A phrase of that intent appearing only negated is the customer's own answer,
        and no score overrules it: the turn is left unplaced (and, naming an order, is
        read as that order's status).
        """
        refused = self._refused_intents(text)
        if intent.name not in refused:
            return intent
        reason = self._settings.intent_router.negation.refused_reason.format(intent=intent.name)
        return self._unknown_for_llm(candidates=list(intent.candidates), reason=reason)

    def _refused_intents(self, text: str) -> set[str]:
        """Intents whose phrases the message uses only negated, and never asserts."""
        return {
            configured.name
            for configured in self._settings.intents
            if any(self._phrases.refused(text, phrase) for phrase in configured.phrases)
            and not any(self._phrases.asserted(text, phrase) for phrase in configured.phrases)
        }

    async def _from_classifier(self, text: str, previous: Intent | None) -> Intent:
        """What the classifier makes of a turn no phrase placed, with context as a fallback."""
        try:
            await self.ensure_ready()
        except Exception:
            return self._clarify(
                candidates=[],
                reason=(
                    "Intent classifier failed to load; the request is not safe to reject "
                    f"deterministically. {self._backend_error}"
                ),
            )

        candidates = await self._classify_with_backend(text)
        if not candidates:
            return self._clarify(
                candidates=[],
                reason=(
                    "Intent classifier returned no candidates; the request is not safe "
                    f"to reject deterministically. {self._backend_error}"
                ),
            )

        mapped_candidates = self._map_model_candidates(candidates)
        if not mapped_candidates:
            return self._clarify(
                candidates=[],
                reason="Intent classifier candidates could not be mapped to the configured taxonomy.",
            )

        top = mapped_candidates[0]
        threshold = self._routing.confidence_threshold
        contextual_followup = self._is_contextual_followup(text)

        # Current-turn evidence always wins. A follow-up marker never gets to
        # overwrite a confident classification for the new message.
        if top.intent in self._intents and top.score >= threshold:
            configured = self._intents[top.intent]
            return Intent(
                IntentAssessment(
                    intent=top.intent,
                    confidence=top.score,
                    disposition=IntentDisposition.SUPPORTED,
                    source=self._source,
                    candidates=mapped_candidates,
                    allowed_tools=list(configured.tools),
                    model=self._backend_model_name(),
                    score_type=self._backend_score_type(),
                    reason="The current turn cleared the configured routing threshold.",
                )
            )

        # Only now may conversational context participate. It is deliberately
        # a fallback, and only for an underspecified follow-up.
        if contextual_followup and previous is not None:
            contextual = self._context_fallback(text, previous, top)
            if contextual is not None:
                return contextual

        # UNKNOWN is uncertainty, not proof that the request is outside the
        # agent's scope. Preserve the candidates and let the LLM use the full
        # conversation context to resolve an ambiguous/follow-up turn.
        if top.intent == self._default:
            return self._unknown_for_llm(
                candidates=mapped_candidates,
                reason="The classifier could not resolve the current turn; LLM context fallback is required.",
            )

        return self._unknown_for_llm(
            candidates=mapped_candidates,
            reason="Classifier score is below the configured routing threshold; LLM context fallback is required.",
        )

    async def rank_with_model(self, message: str) -> list[IntentCandidate]:
        """The classifier's view of a message alone, in this agent's intents, best first.

        No rules, no context: what the model would decide if every turn reached it.
        """
        await self.ensure_ready()
        return self._map_model_candidates(await self._backend.classify(message.strip()))

    def classify(self, message: str, previous: Intent | None = None) -> Intent:
        """Synchronous deterministic-only path, retained for small tests/callers."""
        text = message.strip()
        rejected = self._deterministic_rejection(text)
        if rejected is not None:
            return rejected
        # This synchronous compatibility path has no semantic classifier, so
        # it must not inherit business intent from conversational history.
        if self._is_courtesy(text):
            return self._supported_unknown(
                "Conversational courtesy can be handled without a business tool."
            )
        deterministic_supported = self._deterministic_supported(text)
        if deterministic_supported is not None:
            return deterministic_supported
        return self._clarify([], "Synchronous path does not invoke the BERT classifier.")

    def _deterministic_rejection(self, text: str) -> Intent | None:
        for name, pattern, reason in self._reject_patterns:
            if pattern.search(text):
                self._logger.info(
                    "Customer Support rejected a deterministic out-of-scope request",
                    rule=name,
                )
                return Intent(
                    IntentAssessment(
                        intent=self._default,
                        confidence=1.0,
                        disposition=IntentDisposition.OUT_OF_SCOPE,
                        source="deterministic",
                        matched_rule=name,
                        reason=reason,
                    )
                )
        return None

    def _deterministic_supported(self, text: str) -> Intent | None:
        """Resolve high-precision phrases, or clarify when one phrase maps to multiple intents."""
        if not self._phrase_rules:
            return None
        matches: list[tuple[int, int, str, str]] = []
        for configured in self._settings.intents:
            if configured.name == self._default:
                continue
            for phrase in configured.phrases:
                candidate = phrase.strip().casefold()
                if candidate and self._phrases.asserted(text, candidate):
                    matches.append(
                        (len(candidate.split()), len(candidate), configured.name, phrase.strip())
                    )

        if not matches:
            return None

        # Specific phrases beat generic phrases. If the most-specific phrase
        # belongs to multiple business intents, do not silently grant one of
        # them — return an ambiguity assessment with the competing candidates.
        best_specificity = max((item[0], item[1]) for item in matches)
        best = [item for item in matches if (item[0], item[1]) == best_specificity]
        best_by_intent: dict[str, str] = {}
        for _, _, intent_name, matched_phrase in best:
            best_by_intent.setdefault(intent_name, matched_phrase)

        if len(best_by_intent) > 1:
            intents = list(best_by_intent)
            score = round(1.0 / len(intents), 3)
            candidates = [IntentCandidate(intent=name, score=score) for name in intents]
            return self._unknown_for_llm(
                candidates=candidates,
                reason=(
                    f"The phrase '{best[0][3]}' matches multiple business capabilities; "
                    "LLM context is required to resolve the request."
                ),
            )

        intent_name, matched_phrase = next(iter(best_by_intent.items()))
        configured = self._intents[intent_name]
        self._logger.debug(
            "Customer Support intent resolved by deterministic phrase",
            intent=intent_name,
            phrase=matched_phrase,
        )
        return Intent(
            IntentAssessment(
                intent=intent_name,
                confidence=1.0,
                disposition=IntentDisposition.SUPPORTED,
                source="deterministic",
                matched_rule=f"phrase:{matched_phrase}",
                allowed_tools=list(configured.tools),
                candidates=[IntentCandidate(intent=intent_name, score=1.0)],
                reason="A high-precision current-turn business phrase selected the capability.",
            )
        )

    async def _classify_with_backend(self, text: str) -> list[IntentCandidate]:
        try:
            return await self._backend.classify(text)
        except Exception as exc:
            self._backend_error = f"{type(exc).__name__}: {exc}"
            self._logger.error(
                "Intent classifier inference failed",
                model=self._backend_model_name(),
                error=self._backend_error,
            )
            return []

    def _map_model_candidates(self, candidates: list[IntentCandidate]) -> list[IntentCandidate]:
        """Aggregate the model's native labels into Customer Support intents."""
        mapping = {
            label.casefold(): intent
            for intent, labels in self._routing.model_label_mapping.items()
            for label in labels
        }
        scores = {intent: 0.0 for intent in self._intents}
        scores[self._default] = 0.0

        for candidate in candidates:
            mapped_intent = mapping.get(candidate.intent.casefold(), self._default)
            scores[mapped_intent] = scores.get(mapped_intent, 0.0) + candidate.score

        total = sum(scores.values())
        if total <= 0.0:
            return []

        mapped = [
            IntentCandidate(intent=intent, score=score / total) for intent, score in scores.items()
        ]
        mapped.sort(key=lambda candidate: candidate.score, reverse=True)
        return mapped

    def _backend_model_name(self) -> str:
        return (
            self._backend.model_name
            if self._backend is not None
            else self._app_config.llm.intent_classifier.model
        )

    def _backend_score_type(self) -> str:
        return self._backend.score_type if self._backend is not None else "model_softmax"

    def _is_contextual_followup(self, message: str) -> bool:
        """Return whether the message is plausibly an underspecified follow-up.

        Context is a fallback signal only. Generic one-word terms such as
        ``and`` or ``check`` are intentionally ignored as follow-up markers.
        """
        lowered = message.casefold()

        # Multi-word configured phrases are strong discourse markers. A
        # one-word phrase is too generic to justify inheriting intent.
        strong_phrases = tuple(
            phrase for phrase in self._followup_phrases if len(phrase.split()) >= 2
        )
        has_strong_phrase = any(phrase in lowered for phrase in strong_phrases)
        has_reference = bool(self._reference.search(message))
        has_deictic_reference = bool(
            re.search(r"\b(?:it|this|that|same|those|these|one)\b", lowered)
        )
        return has_strong_phrase or has_reference or has_deictic_reference

    def _context_fallback(
        self,
        message: str,
        previous: Intent,
        top: IntentCandidate,
    ) -> Intent | None:
        """Use previous intent only when the current turn lacks evidence."""
        if previous.disposition is not IntentDisposition.SUPPORTED:
            return None

        previous_config = self._intents.get(previous.name)
        if previous_config is None or previous_config.mode != "read":
            # A vague follow-up must never inherit a write capability. That
            # would silently expand the LLM's authority on the basis of
            # conversational continuity rather than a current-turn request.
            return None

        threshold = self._routing.confidence_threshold
        current_is_unknown = top.intent == self._default
        current_is_uncertain = top.intent in self._intents and top.score < threshold
        if not (current_is_unknown or current_is_uncertain):
            return None

        # The classifier has meaningful evidence for another supported intent.
        # Preserve that evidence and let the normal clarification path run.
        meaningful_competitor = current_is_uncertain and top.score >= (threshold * 0.5)
        if meaningful_competitor:
            return None

        self._logger.debug(
            "Using previous Customer Support intent as constrained follow-up fallback",
            previous_intent=previous.name,
            current_top_intent=top.intent,
            current_top_score=round(top.score, 4),
        )
        return Intent(
            IntentAssessment(
                intent=previous.name,
                confidence=previous.confidence,
                disposition=IntentDisposition.SUPPORTED,
                source="context_fallback",
                candidates=[
                    IntentCandidate(intent=top.intent, score=top.score),
                    IntentCandidate(
                        intent=previous.name,
                        score=max(0.0, 1.0 - top.score),
                    ),
                ],
                allowed_tools=list(previous.allowed_tools),
                matched_rule="continued_context",
                reason=(
                    "The current turn was an underspecified follow-up and did not "
                    "establish a new intent, so the previous supported intent was "
                    "used only as context fallback."
                ),
            )
        )

    def placed_by_order(self, intent: Intent) -> Intent:
        """An unplaced turn that names an order the customer owns, read as that order's status.

        "When will I get my order 10943" is about the order even when no phrase or
        classifier placed it: the configured read intent answers it from the record,
        with no model call. Only an UNKNOWN turn is moved; one that was rejected,
        or placed, keeps its verdict.
        """
        placement = self._settings.intent_router.order_reference
        if intent.name != self._default or intent.disposition == IntentDisposition.OUT_OF_SCOPE:
            return intent
        configured = self._intents[placement.intent]
        return Intent(
            IntentAssessment(
                intent=configured.name,
                confidence=intent.confidence,
                disposition=IntentDisposition.SUPPORTED,
                source=placement.source,
                matched_rule=placement.source,
                allowed_tools=list(configured.tools),
                candidates=intent.candidates,
                model=intent.model,
                score_type=intent.score_type,
                reason=placement.reason,
            )
        )

    @staticmethod
    def _is_courtesy(text: str) -> bool:
        return text.casefold() in {
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "ok",
            "okay",
            "got it",
        }

    @staticmethod
    def _supported_unknown(reason: str) -> Intent:
        return Intent(
            IntentAssessment(
                intent="UNKNOWN",
                confidence=1.0,
                disposition=IntentDisposition.SUPPORTED,
                source="deterministic",
                reason=reason,
            )
        )

    def _unknown_for_llm(self, candidates: list[IntentCandidate], reason: str) -> Intent:
        """Preserve uncertainty and let the LLM resolve it from conversation context."""
        return Intent(
            IntentAssessment(
                intent="UNKNOWN",
                confidence=candidates[0].score if candidates else 0.0,
                disposition=IntentDisposition.SUPPORTED,
                source=self._source if candidates else "classifier_unavailable",
                candidates=candidates,
                allowed_tools=[],
                model=self._backend_model_name(),
                score_type=self._backend_score_type(),
                reason=reason,
            )
        )

    def _clarify(self, candidates: list[IntentCandidate], reason: str) -> Intent:
        """Compatibility helper for callers that explicitly request a clarification."""
        top = [candidate for candidate in candidates if candidate.intent != self._default][
            : self._settings.intent_router.clarification_top_k
        ]
        clarification = self._clarification_for([candidate.intent for candidate in top])
        return Intent(
            IntentAssessment(
                intent="UNKNOWN",
                confidence=candidates[0].score if candidates else 0.0,
                disposition=IntentDisposition.CLARIFY,
                source=self._source if candidates else "classifier_unavailable",
                candidates=candidates,
                model=self._backend_model_name(),
                score_type=self._backend_score_type(),
                clarification=clarification,
                reason=reason,
            )
        )

    @staticmethod
    def _clarification_for(intents: list[str]) -> str:
        key = frozenset(intents)
        templates = {
            frozenset(
                {"ORDER_STATUS", "ORDER_DETAILS"}
            ): "Are you asking about your order status or order details?",
        }
        if key in templates:
            return templates[key]
        if len(intents) == 1:
            return f"Are you asking about {CustomerSupportIntentRouter._friendly(intents[0])}?"
        if len(intents) == 2:
            return f"Are you asking about {CustomerSupportIntentRouter._friendly(intents[0])} or {CustomerSupportIntentRouter._friendly(intents[1])}?"
        return "Could you tell me whether you're asking about an order or a refund?"

    @staticmethod
    def _friendly(intent: str) -> str:
        return {
            "ORDER_STATUS": "your order status",
            "ORDER_DETAILS": "your order details",
            "REQUEST_REFUND": "a refund",
            "REFUND_STATUS": "where your refund stands",
        }.get(intent, "your request")


def intent_choices() -> dict[str, Any]:
    """This agent's intents as a classifier that reads label descriptions is given them."""
    settings = get_agent_settings()
    return {
        "choices": {intent.name: intent.description for intent in settings.intents},
        "instructions": settings.intent_router.instructions,
    }


def load_intent_classifier() -> Any:
    """The shared intent classifier, warmed up, or None if unavailable.

    Called before an event loop opens (native tokenizer start-up inside
    one fails on macOS). A classifier that cannot load narrows the agent
    — the router falls back to its deterministic phrase rules — rather
    than stopping whatever is starting it.
    """
    try:
        classifier = create_intent_classifier(**intent_choices())
        classifier.ensure_loaded_sync()
    except Exception:
        return None
    return classifier
