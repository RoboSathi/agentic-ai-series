"""Sub-question ids a person can read: Q1, Q2, ... in the order they were asked.

The planner model names its own nodes (`n1`, `N1`, whatever it likes) and a reflection round
names its follow-ups by round. Both are renumbered here so a card, a log line and the report
all say "Q3" for the same question, and a follow-up carries on from where the plan stopped.
"""

from collections.abc import Sequence

from research_agent.domain.plan import SubQuestion


def renumber(nodes: Sequence[SubQuestion], prefix: str, start: int = 1) -> tuple[SubQuestion, ...]:
    """The nodes with ids `prefix{start}`, `prefix{start+1}`, ...; dependencies follow."""
    names = {node.node_id: f"{prefix}{number}" for number, node in enumerate(nodes, start=start)}
    return tuple(
        node.model_copy(
            update={
                "node_id": names[node.node_id],
                "depends_on": tuple(names.get(dep, dep) for dep in node.depends_on),
            }
        )
        for node in nodes
    )
