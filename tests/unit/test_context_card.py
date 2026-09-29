"""The context stage keeps its work and loses its card.

Assembly (retrieval, budget fit, health check) is real work and stays. Its
card printed a redacted display copy of working memory, not what the model
receives, and counters nobody acted on. The model's input is on the LLM
Request card; what assembly cost is one number there.
"""

from types import SimpleNamespace
from typing import Any

from agentic_commons.context.types import AssembledContext, Chunk, ContextFinding
from agentic_commons.llm.types import CompletionRequest, Message
from customer_support_agent.graph.nodes.context import ContextNodes
from customer_support_agent.graph.state import AgentState
from customer_support_agent.presentation.reporting import llm_request_fields


class _Nodes(ContextNodes):
    """`ContextNodes` with the reporting seams recorded and the collaborators stubbed."""

    def __init__(self, findings: list[ContextFinding] | None = None) -> None:
        self.done: list[dict[str, Any]] = []
        self.warned: list[dict[str, Any]] = []
        window = AssembledContext(
            chunks=[Chunk(content="Order 10975: T-shirt", kind="retrieved", source="10975")],
            budget_tokens=3000,
            used_tokens=486,
        )

        async def assemble(_task: Any) -> AssembledContext:
            return window

        self._context: Any = SimpleNamespace(assemble=assemble)
        self._strategies: Any = SimpleNamespace(fit=lambda assembled: assembled)
        self._context_health = None
        self._findings = findings or []

    def _check_health(self, fitted: Any, state: Any) -> tuple[Any, list[ContextFinding]]:
        return fitted, self._findings

    def _report(self, stage: str, *args: Any, **kwargs: Any) -> None:
        return None

    def _done(self, stage: str, *args: Any, **kwargs: Any) -> None:
        self.done.append({"stage": stage, **kwargs})

    def _reporter_warning(self, stage: str, **kwargs: Any) -> None:
        self.warned.append({"stage": stage, **kwargs})


def _state() -> AgentState:
    state = AgentState(query="refund", session_id="s", customer_id=89)
    state.working_memory.set("verified_customer", "Linus Hoare (account #89)")
    return state


class TestAHealthyWindowIsQuiet:
    async def test_no_visible_card_and_no_working_memory_copy(self) -> None:
        nodes = _Nodes()
        await nodes.assemble_context(_state())
        [event] = nodes.done
        assert event["quiet"] is True
        assert not any(key.startswith("working_memory") for key in event["fields"])

    async def test_the_log_keeps_what_assembly_cost(self) -> None:
        nodes = _Nodes()
        await nodes.assemble_context(_state())
        fields = nodes.done[0]["fields"]
        assert (fields["used_tokens"], fields["budget_tokens"]) == (486, 3000)

    async def test_the_decision_line_survives(self) -> None:
        result = await _Nodes().assemble_context(_state())
        assert "context: 486/3000 tokens" in result["decisions"]


class TestAPoisonedWindowIsNot:
    async def test_a_finding_that_is_not_routine_draws_one_warning(self) -> None:
        finding = ContextFinding(
            mode="poisoning",
            source="10975",
            rule="injected instruction",
            detail="x",
            action="dropped",
        )
        nodes = _Nodes([finding])
        await nodes.assemble_context(_state())
        [warning] = nodes.warned
        assert "poisoning" in warning["summary"]
        assert nodes.done == []


class TestTheLlmRequestSaysHowBigItIs:
    def test_the_context_size_is_in_tokens(self) -> None:
        request = CompletionRequest(
            messages=[
                Message(role="system", content="x" * 4000),
                Message(role="user", content="refund"),
            ]
        )
        assert llm_request_fields(request, model="m")["context_tokens"] >= 1000
