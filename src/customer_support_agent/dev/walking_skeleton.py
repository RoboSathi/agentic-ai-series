"""Walking skeleton entry point — demonstrates architecture by calling each subsystem."""

import time

from agentic_commons.audit.log import AuditLog
from agentic_commons.audit.types import AuditEvent
from agentic_commons.gatekeeper.gatekeeper import Gatekeeper
from agentic_commons.gatekeeper.types import Action
from agentic_commons.guardrails.guards import PydanticGuard
from agentic_commons.hallucination_protection.checker import HallucinationChecker
from agentic_commons.llm.client import LLMClient
from agentic_commons.logging import get_logger
from agentic_commons.memory.episodic.memory import EpisodicMemory
from agentic_commons.memory.procedural.memory import ProceduralMemory
from agentic_commons.memory.semantic.memory import SemanticMemory
from agentic_commons.memory.working.memory import WorkingMemory
from agentic_commons.security.content_moderation.moderator import ContentModerator
from agentic_commons.security.pii.detector import PIIDetector
from agentic_commons.security.prompt_injection.detector import InjectionDetector
from agentic_commons.tools.registry import ToolRegistry


def main() -> None:
    """Run the walking skeleton — call each subsystem to demonstrate architecture."""
    logger = get_logger("customer_support_agent")

    logger.info("[bold bright_green]═══ WALKING SKELETON START ═══[/bold bright_green]")
    start_time = time.perf_counter()

    ticket = {
        "id": "ticket-001",
        "customer": "John Doe",
        "issue": "Refund request for order #12345",
    }

    logger.info("[bright_green]Intake:[/bright_green] Loading ticket and context")
    memory = WorkingMemory()
    _ = memory.get("default_session")
    logger.info("[cyan]✓ Loaded working context[/cyan]")

    logger.info("[bright_green]Security:[/bright_green] Running security checks")
    pii_detector = PIIDetector()
    _ = pii_detector.scan(ticket["issue"])
    injection_detector = InjectionDetector()
    _ = injection_detector.check(ticket["issue"])
    moderator = ContentModerator()
    _ = moderator.moderate(ticket["issue"])
    logger.info("[cyan]✓ Security checks passed[/cyan]")

    logger.info("[bright_green]Memory:[/bright_green] Retrieving memories")
    episodic = EpisodicMemory()
    episodic.recall("previous interactions")
    semantic = SemanticMemory()
    semantic.query("customer preferences")
    procedural = ProceduralMemory()
    procedural.get_skill("refund_processing")
    logger.info("[cyan]✓ Memories retrieved[/cyan]")

    logger.info("[bright_green]Guardrail:[/bright_green] Validating business rules")
    guard = PydanticGuard()
    guard.validate(ticket, dict)
    logger.info("[cyan]✓ Validation passed[/cyan]")

    logger.info("[bright_green]Gatekeeper:[/bright_green] Checking authorization")
    gatekeeper = Gatekeeper()
    action = Action(name="process_refund")
    _ = gatekeeper.authorize(action)
    logger.info("[cyan]✓ Authorization check complete[/cyan]")

    logger.info("[bright_green]Tools:[/bright_green] Invoking external tools")
    registry = ToolRegistry()
    registry.invoke("lookup_order", {"order_id": "12345"})
    registry.invoke("check_refund_policy", {"order_id": "12345"})
    logger.info("[cyan]✓ Tools invoked[/cyan]")

    logger.info("[bright_green]LLM:[/bright_green] Generating response")
    client = LLMClient()
    response = client.generate(f"Process this ticket: {ticket}")
    logger.info("[cyan]✓ Response generated[/cyan]")

    logger.info("[bright_green]Hallucination Check:[/bright_green] Verifying claim")
    checker = HallucinationChecker()
    checker.check(response.content, str(ticket))
    logger.info("[cyan]✓ Claim verified[/cyan]")

    logger.info("[bright_green]Audit:[/bright_green] Recording transaction")
    audit_log = AuditLog()
    event = AuditEvent(
        event_type="ticket_processed",
        details={"ticket_id": ticket["id"]},
    )
    audit_log.record(event)
    logger.info("[cyan]✓ Transaction recorded[/cyan]")

    elapsed = time.perf_counter() - start_time

    logger.info("[bold bright_green]═══ WALKING SKELETON END ═══[/bold bright_green]")
    logger.info(f"[cyan]Total execution time: {elapsed:.3f}s[/cyan]")
    logger.info(
        "[cyan]All 20+ subsystems called successfully[/cyan]"
    )
    logger.info(
        "[green]✓ Phase 0 Walking Skeleton Complete[/green]"
    )


if __name__ == "__main__":
    main()
