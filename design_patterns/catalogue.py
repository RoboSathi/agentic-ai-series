"""The pattern-selection decision record (E04 and E07's exit artifact).

Sixteen patterns: eight where a person fixes the path at design time (WF)
and eight where a model chooses it at run time (AG). The distinction is
not sophistication — it is *who decides the control flow*, and that is an
authority question before it is an engineering one.

The `depth_episode` column is the map the rest of the series hangs off:
each pattern is introduced here at shape level and built out in full
exactly once, somewhere later. Nothing is taught twice.

This table is data, not prose. `design_patterns/workflow/` and
`design_patterns/agentic/` hold one runnable implementation per row.
"""

from design_patterns.types import PatternCard

WORKFLOW_PATTERNS: list[PatternCard] = [
    PatternCard(
        name="sequence",
        family="WF",
        summary="Fixed steps, in order, each feeding the next.",
        use_when="The steps and their order are known and never vary.",
        avoid_when="The right next step depends on what the last one returned.",
        depth_episode="E04",
        module="M0",
    ),
    PatternCard(
        name="branch",
        family="WF",
        summary="A deterministic predicate picks one of several fixed paths.",
        use_when="The condition is expressible as a rule someone can read.",
        avoid_when="The condition needs judgement about unstructured input.",
        depth_episode="E04",
        module="M0",
    ),
    PatternCard(
        name="fixed_loop",
        family="WF",
        summary="Repeat a step a bounded, known number of times.",
        use_when="The iteration count is known before the loop starts.",
        avoid_when="Only the model can tell when the work is done.",
        depth_episode="E04",
        module="M0",
    ),
    PatternCard(
        name="fan_out_fan_in",
        family="WF",
        summary="Run independent branches concurrently, then join their results.",
        use_when="Sub-tasks are genuinely independent and all are needed.",
        avoid_when="Branches must see each other's output to be correct.",
        depth_episode="E04",
        module="M0",
    ),
    PatternCard(
        name="map_reduce",
        family="WF",
        summary="Apply one operation across many items, then combine.",
        use_when="The corpus exceeds one context window but each item fits.",
        avoid_when="Items need to be interpreted relative to one another.",
        depth_episode="E04",
        module="M0",
    ),
    PatternCard(
        name="router",
        family="WF",
        summary="Classify the request, then dispatch to a fixed handler.",
        use_when="The categories are stable and enumerable.",
        avoid_when="New categories appear faster than the table is updated.",
        depth_episode="E04",
        module="M0",
    ),
    PatternCard(
        name="human_in_the_loop",
        family="WF",
        summary="Pause at a defined point for a person to decide.",
        use_when="The action is irreversible, or its blast radius is wide.",
        avoid_when="Volume would exceed reviewer capacity — check Little's Law.",
        depth_episode="E38",
        module="M15",
    ),
    PatternCard(
        name="rag_pipeline",
        family="WF",
        summary="Retrieve, then generate — retrieval fixed at design time.",
        use_when="Every request needs the same kind of grounding.",
        avoid_when="What to look up depends on partial reasoning.",
        depth_episode="E19",
        module="M5",
    ),
]

AGENTIC_PATTERNS: list[PatternCard] = [
    PatternCard(
        name="tool_use",
        family="AG",
        summary="The model chooses which tool to call, and with what arguments.",
        use_when="The right action depends on the request's content.",
        avoid_when="One fixed call always serves — a sequence is cheaper and safer.",
        depth_episode="E05",
        module="M1",
    ),
    PatternCard(
        name="react",
        family="AG",
        summary="Reason, act, observe — repeatedly, until the task resolves.",
        use_when="A step's result changes what should be done next.",
        avoid_when="The task has no natural stopping point (see E20).",
        depth_episode="E05",
        module="M1",
    ),
    PatternCard(
        name="planning",
        family="AG",
        summary="Commit to an explicit decomposition before executing it.",
        use_when="The work has several steps whose order matters.",
        avoid_when="The first step's result would invalidate the rest of the plan.",
        depth_episode="E06",
        module="M1",
    ),
    PatternCard(
        name="reflection",
        family="AG",
        summary="Critique a finished attempt and revise it.",
        use_when="An external signal grounds the critique.",
        avoid_when="Nothing grounds it — self-critique alone raises confidence, not accuracy.",
        depth_episode="E06",
        module="M1",
    ),
    PatternCard(
        name="evaluator_optimizer",
        family="AG",
        summary="A generator paired with a separate, independent scorer.",
        use_when="Quality is measurable by something other than the generator.",
        avoid_when="The scorer is the generator — that is reflection wearing a hat.",
        depth_episode="E06",
        module="M1",
    ),
    PatternCard(
        name="orchestrator_subagent",
        family="AG",
        summary="One agent plans and delegates bounded sub-tasks to others.",
        use_when="Sub-tasks are separable and each needs its own context.",
        avoid_when="Coordination costs more than the parallelism saves (E43).",
        depth_episode="E41",
        module="M17",
    ),
    PatternCard(
        name="agentic_retrieval",
        family="AG",
        summary="Retrieval as a tool the model chooses to call, mid-reasoning.",
        use_when="What to look up is only clear part-way through the task.",
        avoid_when="A fixed retrieval step would serve — then it is a RAG pipeline.",
        depth_episode="E19",
        module="M5",
    ),
    PatternCard(
        name="memory_systems",
        family="AG",
        summary="The agent decides what to remember and what to recall.",
        use_when="Behaviour must persist across sessions.",
        avoid_when="Nothing verifies what gets written — memory is a security boundary (E16).",
        depth_episode="E13",
        module="M5",
    ),
]

ALL_PATTERNS: list[PatternCard] = [*WORKFLOW_PATTERNS, *AGENTIC_PATTERNS]


def by_name(name: str) -> PatternCard | None:
    """Look up one pattern's decision-record row."""
    return next((card for card in ALL_PATTERNS if card.name == name), None)


def by_family(family: str) -> list[PatternCard]:
    """Every pattern in one family."""
    return [card for card in ALL_PATTERNS if card.family == family]


def depth_map() -> dict[str, str]:
    """Pattern → the episode that builds it out in full.

    The map the rest of the series hangs off (E07's exit artifact).
    """
    return {card.name: card.depth_episode for card in ALL_PATTERNS}
