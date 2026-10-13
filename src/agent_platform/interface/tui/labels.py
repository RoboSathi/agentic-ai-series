"""Presentation labels for the TUI (LLD §20).

The graph owns the workflow and stage sequence. This module only
converts canonical stage keys and tool names into compact, human-readable
titles. It contains no workflow logic and no business decisions.

The map below covers the stages *every* agent on this platform emits.
Stages that exist because a particular product has orders or refunds are
that product's vocabulary: its `AgentHost` supplies them, and they are
merged over these. An unknown stage still degrades readably rather than
failing, so a new stage shows up as itself instead of breaking the feed.
"""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from agent_platform.tools.names import display_tool_name


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
    "deterministic_checks": "DETERMINISTIC CHECKS",
    "intent_classification": "INTENT",
    "memory_recall": "MEMORY RECALL",
    "memory_write": "MEMORY UPDATED",
    "skill_activation": "SKILLS",
    "working_memory_write": "WORKING MEMORY UPDATED",
    "context_assembly": "CONTEXT ASSEMBLY",
    # What one model request is made of, section by section, drawn just before it.
    "llm_context": "CONTEXT",
    "llm_request": "LLM REQUEST",
    "llm_call": "LLM CALL",
    "llm_response_received": "LLM RESPONSE",
    # A reasoning loop's steps: the model thinks, acts, reads what came back,
    # checks its answer, and rewrites what the check refused.
    "thought": "THOUGHT",
    "action": "ACTION",
    "observation": "OBSERVATION",
    "reflect": "REFLECT",
    "revise": "REVISE",
    "answer": "ANSWER",
    "output_parsing": "OUTPUT PARSER",
    "guardrail_validation": "GUARDRAIL",
    "loop_control": "LOOP CONTROL",
    "tool_request": "TOOL REQUEST",
    "tool_execution": "TOOL EXECUTION",
    "tool_result": "TOOL RESULT",
    "executor": "EXECUTOR",
    "mcp_server": "MCP SERVER",
    "mcp_call": "MCP CALL",
    "mcp_notification": "MCP NOTIFICATION",
    "mcp_session": "MCP SESSION",
    "tool_suppressed": "TOOL SUPPRESSED",
    "tool_refused": "TOOL REFUSED",
    "gatekeeper_evaluation": "GATEKEEPER",
    "resilience": "RESILIENCE",
    "audit_record": "AUDIT",
    "hallucination_check": "HALLUCINATION CHECK",
    "escalation": "ESCALATION",
    "finalize": "FINALIZE",
    "run_failed_resumable": "RUN FAILED · RESUMABLE",
    "run_resumed": "RESUMED",
    "run_cancelled": "RUN CANCELLED · RESUMABLE",
    "run_interrupted": "WAITING",
    "review_decision": "REVIEW DECISION",
    "turn_summary": "TURN SUMMARY",
    "memory_update": "MEMORY UPDATE",
    "memory_delete": "MEMORY DELETE",
    "memory_erase": "MEMORY ERASE",
    "knowledge_search": "KNOWLEDGE",
    "consolidation": "CONSOLIDATION",
}


class StageLabelRegistry:
    """Compact display names for the TUI.

    Platform labels, with the running agent's own merged over them.
    """

    def __init__(self, agent_labels: Mapping[str, str] | None = None) -> None:
        self._names = {**_DISPLAY_NAMES, **(agent_labels or {})}

    def title_for(
        self,
        stage: str,
        **substitutions: str,
    ) -> str:
        """Return a canonical human-facing title for one runtime stage."""
        del substitutions
        return self._names.get(
            stage,
            stage.replace("_", " ").upper(),
        )

    def tool_label(self, tool: str) -> str:
        """Any spelling of a tool, as the name the model and every card know it by."""
        return display_tool_name(tool).replace("_", " ")

    def known_stages(self) -> set[str]:
        """Return the presentation aliases known to this renderer."""
        return set(self._names)

    def missing(self, emitted_stages: set[str]) -> set[str]:
        """Return emitted stages that have no explicit presentation alias."""
        return emitted_stages - self.known_stages()
