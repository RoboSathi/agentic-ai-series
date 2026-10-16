"""One security facade for authentication, privacy, injection and moderation."""

from typing import Any

from agent_platform.config import get_settings
from agent_platform.logging import get_logger, instrument
from agent_platform.resilience import ResilientCaller
from agent_platform.security.auth.auth import Authorizer
from agent_platform.security.auth.types import AccessDecision, Principal
from agent_platform.security.content_moderation import (
    ContentModerator,
    ContentSafetyGuard,
    HostedContentSafetyGuard,
)
from agent_platform.security.content_moderation.types import ModerationVerdict
from agent_platform.security.guard_chain import GuardChain
from agent_platform.security.pii.detector import PIIFilterImpl, RedactedSpan
from agent_platform.security.pii.types import RedactionMode
from agent_platform.security.prompt_injection import (
    ChatJailbreakGuard,
    HostedJailbreakGuard,
    InjectionGuard,
    JailbreakGuard,
)
from agent_platform.security.prompt_injection.types import InjectionAssessment
from agent_platform.tools.types import ToolDefinition


def check_names() -> tuple[str, str, str, str]:
    """The checks' names on a card — (rule injection, rule moderation, jailbreak model,
    content-safety model) — from `security.guards`, so a screen can find each verdict."""
    guards = get_settings().security.guards
    return (
        guards.rule_checks.injection,
        guards.rule_checks.moderation,
        guards.jailbreak.check_name,
        guards.content_safety.check_name,
    )


def _advisory(categories: list[str]) -> bool:
    """Whether every category the model named is advisory (an empty list never is)."""
    advisory = {
        c.casefold() for c in get_settings().security.content_moderation.advisory_categories
    }
    return bool(categories) and all(category.casefold() in advisory for category in categories)


def _joined(earlier: str, reason: str) -> str:
    """A reason added to what an assessment already said."""
    return f"{earlier}; {reason}" if earlier else reason[:1].upper() + reason[1:]


@instrument("security")
class SecuritySuite:
    """Auth, PII, deterministic security scans and hosted model guards."""

    def __init__(
        self,
        authorizer: Authorizer | None = None,
        pii: PIIFilterImpl | None = None,
        injection: InjectionGuard | None = None,
        moderator: ContentModerator | None = None,
        jailbreak_guard: JailbreakGuard | None = None,
        content_safety_guard: ContentSafetyGuard | None = None,
        jailbreak_resilient: ResilientCaller | None = None,
        content_safety_resilient: ResilientCaller | None = None,
    ) -> None:
        self.authorizer = authorizer or Authorizer()
        # Each detector is absent when this deployment was not given it.
        # The hosted guards are paired with their deterministic partner:
        # a jailbreak model behind a disabled injection check would be a
        # prompt-injection defence that the configuration says is off.
        self.pii = pii
        self.injection = injection
        self.moderator = moderator
        guard_settings = get_settings().security.guards
        hosted = guard_settings.enabled
        self.jailbreak_guard = jailbreak_guard or (
            _jailbreak_chain(jailbreak_resilient)
            if injection is not None and hosted and guard_settings.jailbreak.enabled
            else None
        )
        self.content_safety_guard = content_safety_guard or (
            _content_safety_chain(content_safety_resilient)
            if moderator is not None and hosted and guard_settings.content_safety.enabled
            else None
        )
        self._logger = get_logger("security")

    def authorize(self, principal: Principal, action: str) -> AccessDecision:
        """Whether this principal may perform this action."""
        return self.authorizer.authorize(principal, action)

    def prepare_untrusted(
        self, instructions: str, data: str, tools: list[ToolDefinition]
    ) -> InjectionAssessment:
        """Assess untrusted content and narrow the toolset accordingly."""
        if self.injection is None:
            # Nothing found, nothing narrowed: with no guard wired, every
            # offered tool stays offered and the text is passed through.
            return InjectionAssessment(
                suspicious=False, allowed_actions=[tool.name for tool in tools]
            )
        return self.injection.assess(instructions, data, tools)

    def pii_model(self) -> tuple[str, str]:
        """The PII model's name, and why it is unavailable ("" for either when not applicable)."""
        return (self.pii.model, self.pii.unavailable) if self.pii else ("", "")

    def pii_tokens(self, text: str) -> list[RedactedSpan]:
        """Each span redaction would replace, by category, recognizer, score and token."""
        return self.pii.tokens(text) if self.pii else []

    def redact(self, text: str, mode: RedactionMode | None = None) -> str:
        """Remove personal data before the text is logged or persisted."""
        return self.pii.redact(text, mode) if self.pii else text

    def minimize(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Strip fields a tool call is not entitled to carry."""
        return self.pii.minimize(payload) if self.pii else dict(payload)

    def moderate(self, text: str) -> ModerationVerdict:
        """Run deterministic content checks synchronously."""
        if self.moderator is None:
            return ModerationVerdict(flagged=False)
        verdict = self.moderator.moderate(text)
        verdict.checks.append(
            {
                "name": get_settings().security.guards.rule_checks.moderation,
                "status": "FAIL" if verdict.flagged else "PASS",
                "reason_code": "BLOCKED_TERM_FOUND" if verdict.flagged else "NO_BLOCKED_TERMS",
                "matched_terms": list(verdict.matched_terms),
                "reason": verdict.reason,
            }
        )
        return verdict

    async def prepare_untrusted_async(
        self, instructions: str, data: str, tools: list[ToolDefinition]
    ) -> InjectionAssessment:
        """Run deterministic injection checks, then the hosted guard when needed."""
        assessment = self.prepare_untrusted(instructions, data, tools)
        settings = get_settings().security.guards
        if assessment.suspicious:
            assessment.checks.append(
                {
                    "name": settings.rule_checks.injection,
                    "status": "FAIL",
                    "reason_code": "DETERMINISTIC_INJECTION_DETECTED",
                    "reason": assessment.reason,
                    "matched_phrases": list(assessment.phrases),
                }
            )
            return assessment

        guard = self.jailbreak_guard
        if not (settings.enabled and settings.jailbreak.enabled and guard is not None):
            return assessment

        verdict = await guard.classify(data)
        assessment.checks.append(
            {
                "name": settings.jailbreak.check_name,
                "status": "FAIL"
                if verdict.flagged
                else "SKIPPED"
                if not verdict.available
                else "PASS",
                "reason_code": (
                    "HOSTED_JAILBREAK_DETECTED"
                    if verdict.flagged
                    else "HOSTED_JAILBREAK_UNAVAILABLE"
                    if not verdict.available
                    else "HOSTED_JAILBREAK_CLEAR"
                ),
                "provider": verdict.provider,
                "model": verdict.model,
                "endpoint": verdict.endpoint,
                "api_called": verdict.api_called,
                "decision": verdict.decision,
                "score": verdict.score,
                "latency_ms": verdict.latency_ms,
                **(
                    {
                        "error_code": verdict.error_code,
                        "http_status": verdict.http_status,
                    }
                    if not verdict.available
                    else {}
                ),
                "reason": verdict.reason,
            }
        )
        if verdict.flagged:
            assessment.suspicious = True
            marker = f"hosted:{verdict.model}"
            if marker not in assessment.phrases:
                assessment.phrases.append(marker)
            assessment.reason = _joined(
                assessment.reason,
                settings.hosted_flagged_reason.format(guard=settings.jailbreak.check_name),
            )
        elif not verdict.available and not settings.fail_open_on_unavailable:
            assessment.suspicious = True
            assessment.reason = _joined(
                assessment.reason,
                settings.fail_closed_reason.format(guard=settings.jailbreak.check_name),
            )
        return assessment

    async def moderate_async(self, text: str) -> ModerationVerdict:
        """Run deterministic moderation, then hosted safety when needed."""
        verdict = self.moderate(text)
        settings = get_settings().security.guards
        if verdict.matched_terms:
            return verdict

        guard = self.content_safety_guard
        if not (settings.enabled and settings.content_safety.enabled and guard is not None):
            return verdict

        model_verdict = await guard.classify(text)
        verdict.checks.append(
            {
                "name": settings.content_safety.check_name,
                "status": "FAIL"
                if model_verdict.flagged
                else "SKIPPED"
                if not model_verdict.available
                else "PASS",
                "reason_code": (
                    "HOSTED_CONTENT_SAFETY_DETECTED"
                    if model_verdict.flagged
                    else "HOSTED_CONTENT_SAFETY_UNAVAILABLE"
                    if not model_verdict.available
                    else "HOSTED_CONTENT_SAFETY_CLEAR"
                ),
                "provider": model_verdict.provider,
                "model": model_verdict.model,
                "endpoint": model_verdict.endpoint,
                "api_called": model_verdict.api_called,
                "decision": model_verdict.decision,
                "categories": list(model_verdict.categories),
                "output": model_verdict.output,
                "latency_ms": model_verdict.latency_ms,
                **(
                    {
                        "error_code": model_verdict.error_code,
                        "http_status": model_verdict.http_status,
                    }
                    if not model_verdict.available
                    else {}
                ),
                "reason": model_verdict.reason,
            }
        )
        if model_verdict.flagged and _advisory(model_verdict.categories):
            # Only categories another screen owns: shown as a warning, the message goes on.
            verdict.checks[-1]["status"] = "ADVISORY"
            verdict.reason = _joined(
                verdict.reason,
                settings.hosted_advisory_reason.format(
                    guard=settings.content_safety.check_name,
                    categories=", ".join(model_verdict.categories),
                ),
            )
        elif model_verdict.flagged:
            verdict.flagged = True
            verdict.advisory_only = False
            for category in model_verdict.categories:
                if category not in verdict.categories:
                    verdict.categories.append(category)
            check = settings.content_safety.check_name
            found = model_verdict.categories
            verdict.reason = _joined(
                verdict.reason,
                settings.hosted_flagged_categories_reason.format(
                    guard=check, categories=", ".join(found)
                )
                if found
                else settings.hosted_flagged_reason.format(guard=check),
            )
        elif not model_verdict.available and not settings.fail_open_on_unavailable:
            verdict.flagged = True
            verdict.advisory_only = False
            verdict.reason = _joined(
                verdict.reason,
                settings.fail_closed_reason.format(guard=settings.content_safety.check_name),
            )
        return verdict

    @property
    def enabled(self) -> bool:
        """Whether any content-level screening was wired at all."""
        return self.injection is not None or self.moderator is not None


def _jailbreak_chain(resilient: ResilientCaller | None) -> GuardChain:
    """The hosted jailbreak classifier, then each configured chat fallback."""
    settings = get_settings().security.guards.jailbreak
    primary = HostedJailbreakGuard(
        base_url=settings.base_url,
        endpoint=settings.endpoint,
        timeout_seconds=settings.timeout_seconds,
        model=settings.model,
        resilient=resilient or ResilientCaller(settings.resilience_key),
    )
    fallbacks = [ChatJailbreakGuard(settings.as_fallback(item)) for item in settings.fallbacks]
    return GuardChain([primary, *fallbacks])


def _content_safety_chain(resilient: ResilientCaller | None) -> GuardChain:
    """The hosted content-safety model, then each configured fallback, asked the same way."""
    settings = get_settings().security.guards.content_safety
    guards = [
        HostedContentSafetyGuard(
            base_url=spec.base_url,
            endpoint=spec.endpoint,
            timeout_seconds=spec.timeout_seconds,
            model=spec.model,
            max_tokens=spec.max_tokens,
            resilient=(resilient if index == 0 else None) or ResilientCaller(spec.resilience_key),
            settings=spec,
        )
        for index, spec in enumerate(
            [settings, *(settings.as_fallback(item) for item in settings.fallbacks)]
        )
    ]
    return GuardChain(guards)
