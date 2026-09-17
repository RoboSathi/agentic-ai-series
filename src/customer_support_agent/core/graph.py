"""LangGraph walking skeleton for customer support agent."""


from langgraph.graph import END, START, StateGraph

from agentic_commons.audit.log import AuditLog
from agentic_commons.gatekeeper.gatekeeper import Gatekeeper
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


class AgentState(dict):  # type: ignore[type-arg]
    """State for the agent graph."""

    def __init__(self) -> None:
        super().__init__()
        self["messages"] = []
        self["context"] = {}
        self["current_ticket"] = {"id": "", "customer": "", "issue": ""}
        self["response"] = ""
        self["final_response"] = ""


def intake_node(state: AgentState) -> AgentState:
    """Intake node — receives ticket and loads context."""
    logger = get_logger("customer_support_agent")
    logger.info("[bright_green]Intake node[/bright_green]")

    # Load working memory
    memory = WorkingMemory()
    context = memory.get("default_session")

    state["context"] = context.model_dump()
    state["current_ticket"] = {
        "id": "ticket-001",
        "customer": "John Doe",
        "issue": "Refund request for order #12345",
    }
    return state


def security_check_node(state: AgentState) -> AgentState:
    """Security check node — runs PII, injection, moderation checks."""
    logger = get_logger("customer_support_agent")
    logger.info("[bright_green]Security check node[/bright_green]")

    issue_text = state["current_ticket"]["issue"]

    pii_detector = PIIDetector()
    _ = pii_detector.scan(issue_text)

    injection_detector = InjectionDetector()
    _ = injection_detector.check(issue_text)

    moderator = ContentModerator()
    _ = moderator.moderate(issue_text)

    return state


def memory_recall_node(state: AgentState) -> AgentState:
    """Memory recall node — retrieves relevant memories."""
    logger = get_logger("customer_support_agent")
    logger.info("[bright_green]Memory recall node[/bright_green]")

    episodic = EpisodicMemory()
    episodic.recall("previous interactions")

    semantic = SemanticMemory()
    semantic.query("customer preferences")

    procedural = ProceduralMemory()
    procedural.get_skill("refund_processing")

    return state


def guardrail_node(state: AgentState) -> AgentState:
    """Guardrail node — validates against business rules."""
    logger = get_logger("customer_support_agent")
    logger.info("[bright_green]Guardrail node[/bright_green]")

    guard = PydanticGuard()
    guard.validate(state["current_ticket"], dict)

    return state


def gatekeeper_node(state: AgentState) -> AgentState:
    """Gatekeeper node — checks authorization."""
    logger = get_logger("customer_support_agent")
    logger.info("[bright_green]Gatekeeper node[/bright_green]")

    from agentic_commons.gatekeeper.types import Action

    gatekeeper = Gatekeeper()
    action = Action(name="process_refund")
    _ = gatekeeper.authorize(action)

    return state


def tools_node(state: AgentState) -> AgentState:
    """Tools node — invokes external tools."""
    logger = get_logger("customer_support_agent")
    logger.info("[bright_green]Tools node[/bright_green]")

    registry = ToolRegistry()
    registry.invoke("lookup_order", {"order_id": "12345"})
    registry.invoke("check_refund_policy", {"order_id": "12345"})

    return state


def llm_node(state: AgentState) -> AgentState:
    """LLM node — generates response using language model."""
    logger = get_logger("customer_support_agent")
    logger.info("[bright_green]LLM node[/bright_green]")

    client = LLMClient()
    prompt = f"Process this ticket: {state['current_ticket']}"
    response = client.generate(prompt)

    state["response"] = response.content
    return state


def hallucination_check_node(state: AgentState) -> AgentState:
    """Hallucination check node — verifies LLM output."""
    logger = get_logger("customer_support_agent")
    logger.info("[bright_green]Hallucination check node[/bright_green]")

    checker = HallucinationChecker()
    checker.check(state.get("response", ""), str(state["context"]))

    return state


def audit_node(state: AgentState) -> AgentState:
    """Audit node — records the transaction."""
    logger = get_logger("customer_support_agent")
    logger.info("[bright_green]Audit node[/bright_green]")

    from agentic_commons.audit.types import AuditEvent

    audit_log = AuditLog()
    event = AuditEvent(
        event_type="ticket_processed",
        details={"ticket_id": state["current_ticket"]["id"]},
    )
    audit_log.record(event)

    return state


def response_node(state: AgentState) -> AgentState:
    """Response node — formats and returns the final response."""
    logger = get_logger("customer_support_agent")
    logger.info("[bright_green]Response node[/bright_green]")

    state["final_response"] = state.get("response", "[no response generated]")
    return state


def build_graph() -> StateGraph:
    """Build the LangGraph walking skeleton."""
    graph = StateGraph(AgentState)

    graph.add_node("intake", intake_node)
    graph.add_node("security_check", security_check_node)
    graph.add_node("memory_recall", memory_recall_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("gatekeeper", gatekeeper_node)
    graph.add_node("tools", tools_node)
    graph.add_node("llm", llm_node)
    graph.add_node("hallucination_check", hallucination_check_node)
    graph.add_node("audit", audit_node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "security_check")
    graph.add_edge("security_check", "memory_recall")
    graph.add_edge("memory_recall", "guardrail")
    graph.add_edge("guardrail", "gatekeeper")
    graph.add_edge("gatekeeper", "tools")
    graph.add_edge("tools", "llm")
    graph.add_edge("llm", "hallucination_check")
    graph.add_edge("hallucination_check", "audit")
    graph.add_edge("audit", "response")
    graph.add_edge("response", END)

    return graph
