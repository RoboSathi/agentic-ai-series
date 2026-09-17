"""Test the walking skeleton script."""

from customer_support_agent.dev.walking_skeleton import main


def test_walking_skeleton_main_runs() -> None:
    """Test that walking skeleton main function runs."""
    main()


def test_walking_skeleton_imports() -> None:
    """Test that walking skeleton imports work."""
    from agentic_commons.audit.log import AuditLog
    from agentic_commons.audit.types import AuditEvent
    from agentic_commons.gatekeeper.gatekeeper import Gatekeeper
    from agentic_commons.gatekeeper.types import Action
    from agentic_commons.guardrails.guards import PydanticGuard
    from agentic_commons.hallucination_protection.checker import HallucinationChecker
    from agentic_commons.llm.client import LLMClient
    from agentic_commons.memory.episodic.memory import EpisodicMemory
    from agentic_commons.memory.procedural.memory import ProceduralMemory
    from agentic_commons.memory.semantic.memory import SemanticMemory
    from agentic_commons.memory.working.memory import WorkingMemory
    from agentic_commons.security.content_moderation.moderator import ContentModerator
    from agentic_commons.security.pii.detector import PIIDetector
    from agentic_commons.security.prompt_injection.detector import InjectionDetector
    from agentic_commons.tools.registry import ToolRegistry

    assert AuditLog is not None
    assert AuditEvent is not None
    assert Gatekeeper is not None
    assert Action is not None
    assert PydanticGuard is not None
    assert HallucinationChecker is not None
    assert LLMClient is not None
    assert EpisodicMemory is not None
    assert ProceduralMemory is not None
    assert SemanticMemory is not None
    assert WorkingMemory is not None
    assert ContentModerator is not None
    assert PIIDetector is not None
    assert InjectionDetector is not None
    assert ToolRegistry is not None
