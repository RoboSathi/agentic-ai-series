"""One security facade for authentication, privacy, injection and moderation."""

from typing import Any

from agentic_commons.config import get_settings
from agentic_commons.logging import get_logger, instrument
from agentic_commons.resilience import ResilientCaller
from agentic_commons.security.auth.auth import Authorizer
from agentic_commons.security.auth.types import AccessDecision, Principal
from agentic_commons.security.content_moderation import (
    ContentModerator,
    ContentSafetyGuard,
    NvidiaHostedContentSafetyGuard,
)
from agentic_commons.security.content_moderation.types import ModerationVerdict
from agentic_commons.security.pii.detector import PIIFilterImpl
from agentic_commons.security.pii.types import RedactionMode
from agentic_commons.security.prompt_injection import (
    JailbreakGuard,
    NvidiaHostedJailbreakGuard,
    PromptInjectionGuard,
)
from agentic_commons.security.prompt_injection.types import InjectionAssessment
from agentic_commons.tools.types import ToolDefinition


@instrument("security")
class SecuritySuite:
    """Auth, PII, deterministic security scans and hosted model guards."""

    def __init__(
        self,
        authorizer: Authorizer | None = None,
        pii: PIIFilterImpl | None = None,
        injection: PromptInjectionGuard | None = None,
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
        self.jailbreak_guard = jailbreak_guard or (
            NvidiaHostedJailbreakGuard(
                base_url=guard_settings.jailbreak.base_url,
                endpoint=guard_settings.jailbreak.endpoint,
                timeout_seconds=guard_settings.jailbreak.timeout_seconds,
                model=guard_settings.jailbreak.model,
                resilient=jailbreak_resilient or ResilientCaller("nvidia-jailbreak"),
            )
            if injection is not None
            and guard_settings.enabled
            and guard_settings.jailbreak.enabled
            else None
        )
        self.content_safety_guard = content_safety_guard or (
            NvidiaHostedContentSafetyGuard(
                base_url=guard_settings.content_safety.base_url,
                endpoint=guard_settings.content_safety.endpoint,
                timeout_seconds=guard_settings.content_safety.timeout_seconds,
                model=guard_settings.content_safety.model,
                max_tokens=guard_settings.content_safety.max_tokens,
                resilient=content_safety_resilient or ResilientCaller("nvidia-content-safety"),
            )
            if moderator is not None
            and guard_settings.enabled
            and guard_settings.content_safety.enabled
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
                "name": "Deterministic blocked-term scan",
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
                    "name": "Deterministic prompt-injection scan",
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
                "name": "NVIDIA NemoGuard JailbreakDetect",
                "status": "FAIL" if verdict.flagged else "SKIPPED" if not verdict.available else "PASS",
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
            assessment.reason = (
                f"{assessment.reason}; hosted jailbreak guard flagged the input"
                if assessment.reason
                else "Hosted jailbreak guard flagged the input"
            )
        elif not verdict.available and not settings.fail_open_on_unavailable:
            assessment.suspicious = True
            assessment.reason = (
                f"{assessment.reason}; hosted jailbreak guard unavailable and security policy is configured to fail closed"
                if assessment.reason
                else "Hosted jailbreak guard unavailable and security policy is configured to fail closed"
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
                "name": "NVIDIA Nemotron Safety Guard 8B v3",
                "status": "FAIL" if model_verdict.flagged else "SKIPPED" if not model_verdict.available else "PASS",
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
        if model_verdict.flagged:
            verdict.flagged = True
            verdict.advisory_only = False
            for category in model_verdict.categories:
                if category not in verdict.categories:
                    verdict.categories.append(category)
            verdict.reason = (
                f"{verdict.reason}; hosted Nemotron Safety Guard flagged the content"
                if verdict.reason
                else "Hosted Nemotron Safety Guard flagged the content"
            )
        elif not model_verdict.available and not settings.fail_open_on_unavailable:
            verdict.flagged = True
            verdict.advisory_only = False
            verdict.reason = (
                f"{verdict.reason}; hosted content-safety guard unavailable and security policy is configured to fail closed"
                if verdict.reason
                else "Hosted content-safety guard unavailable and security policy is configured to fail closed"
            )
        return verdict

    @property
    def enabled(self) -> bool:
        """Whether any content-level screening was wired at all."""
        return self.injection is not None or self.moderator is not None
