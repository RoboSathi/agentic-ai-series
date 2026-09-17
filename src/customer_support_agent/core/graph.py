"""LangGraph customer support agent.

Every optional safety/governance step (PII, prompt injection, content
moderation, memory recall, guardrails, gatekeeper, hallucination
protection, audit) is checked against config.yaml in this exact sequence
and only runs if enabled — with everything but llm/tools/persistence off
by default, a query goes straight to the LLM's tool-calling loop.
"""

import json
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic_commons.audit.log import AuditLog
from agentic_commons.audit.types import AuditEvent
from agentic_commons.config import get_settings, is_enabled
from agentic_commons.gatekeeper.gatekeeper import Gatekeeper
from agentic_commons.gatekeeper.types import Action
from agentic_commons.guardrails.guards import PydanticGuard
from agentic_commons.hallucination_protection.checker import HallucinationChecker
from agentic_commons.llm.client import LLMClient, create_llm_client
from agentic_commons.llm.types import Message
from agentic_commons.logging import get_logger
from agentic_commons.memory.episodic.memory import EpisodicMemory
from agentic_commons.memory.procedural.memory import ProceduralMemory
from agentic_commons.memory.semantic.memory import SemanticMemory
from agentic_commons.memory.working.memory import WorkingMemory
from agentic_commons.persistence.database import Database
from agentic_commons.security.content_moderation.moderator import ContentModerator
from agentic_commons.security.pii.detector import PIIDetector
from agentic_commons.security.prompt_injection.detector import InjectionDetector
from agentic_commons.tools.registry import ToolRegistry
from customer_support_agent.core.schema import ensure_schema
from customer_support_agent.core.tools import build_tool_registry
from customer_support_agent.dummy_data import DB_PATH as DEFAULT_DB_PATH
from customer_support_agent.prompts.system_prompt import build_system_prompt

_LOGGER_NAME = "customer_support_agent"


class AgentState(TypedDict):
    """State threaded through the customer support agent graph."""

    messages: Annotated[list[Message], operator.add]
    query: str
    final_response: str
    tool_call_count: int


def _skip(step: str, toggle_path: str) -> None:
    get_logger(_LOGGER_NAME).info(
        f"[dim]Skipping {step} ({toggle_path}=False)[/dim]"
    )


def intake_node(state: AgentState) -> dict[str, Any]:
    """Intake — log receipt of the query; messages are already seeded."""
    get_logger(_LOGGER_NAME).info(f"[bright_green]Intake:[/bright_green] {state['query']}")
    return {}


def pii_check_node(state: AgentState) -> dict[str, Any]:
    """PII check — toggle-gated on security.pii.enabled."""
    if is_enabled("security.pii"):
        get_logger(_LOGGER_NAME).info("[bright_green]Security:[/bright_green] PII scan")
        PIIDetector().scan(state["query"])
    else:
        _skip("PII scan", "security.pii.enabled")
    return {}


def prompt_injection_check_node(state: AgentState) -> dict[str, Any]:
    """Prompt injection check — toggle-gated on security.prompt_injection.enabled."""
    if is_enabled("security.prompt_injection"):
        get_logger(_LOGGER_NAME).info(
            "[bright_green]Security:[/bright_green] Prompt injection check"
        )
        InjectionDetector().check(state["query"])
    else:
        _skip("prompt injection check", "security.prompt_injection.enabled")
    return {}


def content_moderation_check_node(state: AgentState) -> dict[str, Any]:
    """Content moderation check — toggle-gated on security.content_moderation.enabled."""
    if is_enabled("security.content_moderation"):
        get_logger(_LOGGER_NAME).info(
            "[bright_green]Security:[/bright_green] Content moderation"
        )
        ContentModerator().moderate(state["query"])
    else:
        _skip("content moderation", "security.content_moderation.enabled")
    return {}


def memory_recall_node(state: AgentState) -> dict[str, Any]:
    """Memory recall — toggle-gated on memory.enabled."""
    if is_enabled("memory"):
        get_logger(_LOGGER_NAME).info("[bright_green]Memory:[/bright_green] Recalling")
        WorkingMemory().get("default_session")
        EpisodicMemory().recall(state["query"])
        SemanticMemory().query(state["query"])
        ProceduralMemory().get_skill("refund_processing")
    else:
        _skip("memory recall", "memory.enabled")
    return {}


def guardrail_check_node(state: AgentState) -> dict[str, Any]:
    """Guardrail check — toggle-gated on guardrails.enabled."""
    if is_enabled("guardrails"):
        get_logger(_LOGGER_NAME).info("[bright_green]Guardrail:[/bright_green] Validating")
        PydanticGuard().validate({"query": state["query"]}, dict)
    else:
        _skip("guardrail validation", "guardrails.enabled")
    return {}


def _make_agent_node(llm_client: LLMClient, tool_registry: ToolRegistry) -> Any:
    """Build the agent node: one LLM call with tools bound."""

    def agent_node(state: AgentState) -> dict[str, Any]:
        logger = get_logger(_LOGGER_NAME)
        round_num = state["tool_call_count"] + 1
        logger.info(f"[bright_green]LLM:[/bright_green] Thinking (round {round_num})")
        response = llm_client.generate(
            state["messages"], tools=tool_registry.to_openai_tools()
        )

        if response.content:
            logger.info(f"[dim]  says:[/dim] {response.content}")

        assistant_message = Message(
            role="assistant", content=response.content, tool_calls=response.tool_calls
        )
        return {
            "messages": [assistant_message],
            "final_response": response.content,
        }

    return agent_node


def _make_route_after_agent(max_iterations: int) -> Any:
    """Build the conditional-edge function deciding tools vs. finalize vs. finish."""

    def route_after_agent(state: AgentState) -> str:
        last_message = state["messages"][-1]
        if not last_message.tool_calls:
            return "hallucination_check"
        if state["tool_call_count"] < max_iterations:
            return "tools"
        # Hit the iteration cap while the model still wanted a tool: force
        # one text-only turn so the user gets an actual answer instead of
        # whatever (often empty) content came with that last tool request.
        return "finalize"

    return route_after_agent


def _make_finalize_node(llm_client: LLMClient) -> Any:
    """Build the finalize node: one tool-free LLM call to force a text answer.

    Only reached when max_tool_iterations was hit while the model still
    wanted to call a tool.
    """

    def finalize_node(state: AgentState) -> dict[str, Any]:
        logger = get_logger(_LOGGER_NAME)
        logger.info(
            "[yellow]Tool call limit reached — forcing a final answer[/yellow]"
        )
        nudge = Message(
            role="user",
            content=(
                "You've reached the tool call limit. Give your final answer "
                "now based on everything above — no more tools are available."
            ),
        )
        response = llm_client.generate(state["messages"] + [nudge])
        if response.content:
            logger.info(f"[dim]  says:[/dim] {response.content}")
        assistant_message = Message(role="assistant", content=response.content)
        return {
            "messages": [nudge, assistant_message],
            "final_response": response.content,
        }

    return finalize_node


def _make_tools_node(tool_registry: ToolRegistry) -> Any:
    """Build the tools node: executes every requested tool call.

    process_refund specifically requires gatekeeper authorization when
    gatekeeper is enabled — every other tool just runs.
    """

    def tools_node(state: AgentState) -> dict[str, Any]:
        logger = get_logger(_LOGGER_NAME)
        last_message = state["messages"][-1]
        tool_messages: list[Message] = []

        for call in last_message.tool_calls or []:
            logger.info(f"[bright_green]Tools:[/bright_green] calling {call.name}({call.arguments})")
            try:
                args = json.loads(call.arguments)
            except json.JSONDecodeError:
                output = f"Invalid arguments for {call.name}: {call.arguments}"
            else:
                if call.name == "process_refund" and is_enabled("gatekeeper"):
                    logger.info("[dim]  checking gatekeeper authorization...[/dim]")
                    decision = Gatekeeper().authorize(Action(name="process_refund"))
                    output = (
                        tool_registry.invoke(call.name, args).output
                        if decision.authorized
                        else "Not authorized to process this refund"
                    )
                else:
                    output = tool_registry.invoke(call.name, args).output

            logger.info(f"[dim]  →[/dim] {output}")
            tool_messages.append(
                Message(role="tool", tool_call_id=call.id, content=output)
            )

        return {
            "messages": tool_messages,
            "tool_call_count": state["tool_call_count"] + 1,
        }

    return tools_node


def hallucination_check_node(state: AgentState) -> dict[str, Any]:
    """Hallucination check — toggle-gated on hallucination_protection.enabled."""
    if is_enabled("hallucination_protection"):
        get_logger(_LOGGER_NAME).info(
            "[bright_green]Hallucination Check:[/bright_green] Verifying"
        )
        HallucinationChecker().check(state["final_response"], state["query"])
    else:
        _skip("hallucination check", "hallucination_protection.enabled")
    return {}


def audit_node(state: AgentState) -> dict[str, Any]:
    """Audit — toggle-gated on audit.enabled."""
    if is_enabled("audit"):
        get_logger(_LOGGER_NAME).info("[bright_green]Audit:[/bright_green] Recording")
        AuditLog().record(
            AuditEvent(event_type="query_handled", details={"query": state["query"]})
        )
    else:
        _skip("audit recording", "audit.enabled")
    return {}


def build_graph(
    llm_client: LLMClient | None = None,
    database: Database | None = None,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Build and compile the customer support agent graph."""
    db = database or Database(db_path=DEFAULT_DB_PATH)
    ensure_schema(db)
    tool_registry = build_tool_registry(db)
    client = llm_client or create_llm_client()
    max_iterations = get_settings().customer_support_agent.max_tool_iterations

    graph: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)
    graph.add_node("intake", intake_node)
    graph.add_node("pii_check", pii_check_node)
    graph.add_node("prompt_injection_check", prompt_injection_check_node)
    graph.add_node("content_moderation_check", content_moderation_check_node)
    graph.add_node("memory_recall", memory_recall_node)
    graph.add_node("guardrail_check", guardrail_check_node)
    graph.add_node("agent", _make_agent_node(client, tool_registry))
    graph.add_node("tools", _make_tools_node(tool_registry))
    graph.add_node("finalize", _make_finalize_node(client))
    graph.add_node("hallucination_check", hallucination_check_node)
    graph.add_node("audit", audit_node)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "pii_check")
    graph.add_edge("pii_check", "prompt_injection_check")
    graph.add_edge("prompt_injection_check", "content_moderation_check")
    graph.add_edge("content_moderation_check", "memory_recall")
    graph.add_edge("memory_recall", "guardrail_check")
    graph.add_edge("guardrail_check", "agent")
    graph.add_conditional_edges(
        "agent",
        _make_route_after_agent(max_iterations),
        {
            "tools": "tools",
            "finalize": "finalize",
            "hallucination_check": "hallucination_check",
        },
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", "hallucination_check")
    graph.add_edge("hallucination_check", "audit")
    graph.add_edge("audit", END)

    return graph.compile()


def run_agent(
    query: str,
    llm_client: LLMClient | None = None,
    database: Database | None = None,
) -> str:
    """Run one customer support query through the full agent graph."""
    compiled = build_graph(llm_client=llm_client, database=database)
    initial_state: AgentState = {
        "messages": [
            Message(role="system", content=build_system_prompt()),
            Message(role="user", content=query),
        ],
        "query": query,
        "final_response": "",
        "tool_call_count": 0,
    }
    result = compiled.invoke(initial_state)
    return str(result["final_response"])
