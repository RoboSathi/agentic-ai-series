"""The planner: contract in, validated DAG out — or a clean failure.

One repair attempt, then stop. A planner that cannot produce a valid plan
twice is not going to on the fifth try, and every retry is spent budget.
"""

from agent_platform.logging import get_logger, instrument
from research_agent.domain.plan import PlanDag, PlanValidation, SubQuestion
from research_agent.domain.rubric import Contract
from research_agent.llm.structured import StructuredCaller
from research_agent.planning.dag_validation import validate_plan
from research_agent.planning.ids import renumber
from research_agent.prompts import planner as prompt
from research_agent.prompts.base import RolePrompt


class PlanRejectedError(RuntimeError):
    """The plan failed validation, and so did its one repair."""

    def __init__(self, validation: PlanValidation) -> None:
        super().__init__("; ".join(validation.problems))
        self.validation = validation


@instrument("research_agent")
class Planner:
    """Asks the planner role for a DAG and holds it to the validation gate."""

    def __init__(
        self, caller: StructuredCaller, max_queries_per_node: int, node_id_prefix: str
    ) -> None:
        self._caller = caller
        self._max_queries = max_queries_per_node
        self._prefix = node_id_prefix
        self._logger = get_logger("research_agent")

    async def plan(self, contract: Contract) -> tuple[PlanDag, PlanValidation, bool]:
        """(plan, its validation, whether a repair was needed); raises if both attempts fail."""
        request = prompt.build(contract, contract.budget.max_dag_nodes, self._max_queries)
        plan = await self._ask(request)
        validation = validate_plan(plan, contract.rubric, contract.budget.max_dag_nodes)
        if validation.valid:
            return self._numbered(plan), validation, False
        self._logger.warning("Plan rejected; one repair attempt", problems=validation.problems)
        repair = request.with_instruction(
            prompt.REPAIR_INSTRUCTION.format(problems="\n".join(validation.problems))
        )
        plan = await self._ask(repair)
        validation = validate_plan(plan, contract.rubric, contract.budget.max_dag_nodes)
        if not validation.valid:
            raise PlanRejectedError(validation)
        return self._numbered(plan), validation, True

    def _numbered(self, plan: PlanDag) -> PlanDag:
        """The validated plan with readable ids (validated first: duplicates must be seen)."""
        return plan.model_copy(update={"nodes": renumber(plan.nodes, self._prefix)})

    async def _ask(self, request: RolePrompt) -> PlanDag:
        answer = await self._caller.call(request, prompt.PlanAnswer)
        return PlanDag(
            nodes=tuple(
                SubQuestion(
                    node_id=node.node_id,
                    question=node.question,
                    depends_on=tuple(node.depends_on),
                    rubric_items=tuple(node.rubric_items),
                    queries=tuple(node.queries[: self._max_queries]),
                    subject=node.subject,
                )
                for node in answer.nodes
            ),
            rationale=answer.rationale,
        )
