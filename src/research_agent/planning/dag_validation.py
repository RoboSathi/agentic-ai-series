"""The plan validation gate: a plan runs only if it is acyclic, bounded and traced.

Three rules, all deterministic:
- acyclic (and closed: no dependency on an undeclared node, no duplicate id);
- at most `max_dag_nodes` nodes;
- every node traces to at least one Rubric Item that exists.

A plan that serves no acceptance criterion is research for its own sake,
which is the open-ended search this agent exists not to do.
"""

from agentic_commons.scheduling import InvalidTaskGraphError, TaskGraph
from research_agent.domain.plan import PlanDag, PlanValidation
from research_agent.domain.rubric import Rubric


def validate_plan(plan: PlanDag, rubric: Rubric, max_nodes: int) -> PlanValidation:
    problems: list[str] = []
    cycle: list[str] = []
    if not plan.nodes:
        problems.append("the plan has no nodes")
    if len(plan.nodes) > max_nodes:
        problems.append(f"{len(plan.nodes)} nodes exceeds the limit of {max_nodes}")
    try:
        TaskGraph(plan.tasks())
    except InvalidTaskGraphError as exc:
        problems.append(str(exc))
        cycle = exc.cycle
    # With the rubric withheld there is nothing to trace to (an ablation):
    # the rule applies whenever acceptance criteria exist.
    problems.extend(_untraced(plan, rubric) if rubric.items else _queryless(plan))
    return PlanValidation(problems=problems, cycle=cycle)


def _queryless(plan: PlanDag) -> list[str]:
    return [f"node {node.node_id} has no queries" for node in plan.nodes if not node.queries]


def _untraced(plan: PlanDag, rubric: Rubric) -> list[str]:
    known = set(rubric.ids)
    problems = []
    for node in plan.nodes:
        valid = [item_id for item_id in node.rubric_items if item_id in known]
        if not valid:
            problems.append(f"node {node.node_id} traces to no rubric item")
        if not node.queries:
            problems.append(f"node {node.node_id} has no queries")
    return problems
