"""Presentation labels for the customer-support TUI.

The graph owns the workflow and stage sequence.  This module only converts
canonical stage keys and tool names into compact, human-readable titles.
It contains no workflow logic and no business decisions.
"""

from typing import Protocol, runtime_checkable

from agentic_commons.tools.types import WIRE_SEPARATOR


@runtime_checkable
class StageLabeler(Protocol):
    """Maps a canonical runtime stage key to a display title."""

    def title_for(self, stage: str, **substitutions: str) -> str: ...

    def tool_label(self, tool: str) -> str: ...


# Canonical presentation vocabulary.  This is intentionally a presentation
# map, not a workflow definition.  Unknown stages fall back to a readable
# transformation, so a new graph stage does not break the TUI.
_DISPLAY_NAMES = {
    "intake": "INTAKE",
    "authentication": "AUTH",
    "prompt_injection_check": "PROMPT INJECTION",
    "pii_filtering": "PII FILTER",
    "content_moderation": "CONTENT POLICY",
    "order_lookup": "ORDER LOOKUP",
    "order_status_check": "ORDER STATUS",
    "refund_eligibility_check": "REFUND ELIGIBILITY",
    "intent_classification": "INTENT",
    "working_memory_check": "WORKING MEMORY CHECK",
    "memory_recall": "MEMORY RECALL",
    "memory_write": "MEMORY UPDATED",
    "working_memory_write": "WORKING MEMORY UPDATED",
    "context_assembly": "CONTEXT",
    "llm_call": "LLM CALL",
    "llm_response_received": "LLM RESPONSE",
    "output_parsing": "OUTPUT PARSER",
    "guardrail_validation": "GUARDRAIL",
    "control_loop": "CONTROL LOOP",
    "tool_request": "TOOL REQUEST",
    "tool_execution": "TOOL EXECUTION",
    "tool_result": "TOOL RESULT",
    "executor": "EXECUTOR",
    "tool_suppressed": "TOOL SUPPRESSED",
    "tool_refused": "TOOL REFUSED",
    "gatekeeper_evaluation": "GATEKEEPER",
    "resilience": "RESILIENCE",
    "audit_record": "AUDIT",
    "hallucination_check": "HALLUCINATION CHECK",
    "escalation": "ESCALATION",
    "finalize": "FINALIZE",
}


class StageLabelRegistry:
    """Compact display names for the TUI."""

    def title_for(
        self,
        stage: str,
        **substitutions: str,
    ) -> str:
        """Return a canonical human-facing title for one runtime stage."""
        del substitutions
        return _DISPLAY_NAMES.get(
            stage,
            stage.replace("_", " ").upper(),
        )

    def tool_label(self, tool: str) -> str:
        """Convert wire / qualified tool names to concise display names."""
        bare = (
            tool.rsplit(WIRE_SEPARATOR, maxsplit=1)[-1]
            .rsplit(".", maxsplit=1)[-1]
        )
        return bare.replace("_", " ")

    def known_stages(self) -> set[str]:
        """Return the presentation aliases known to this renderer."""
        return set(_DISPLAY_NAMES)

    def missing(self, emitted_stages: set[str]) -> set[str]:
        """Return emitted stages that have no explicit presentation alias."""
        return emitted_stages - self.known_stages()
