"""Surface classification and the Three Gates (E12/E15, LLD §11).

Four surface classes, distinguished by what *may* be written to them:

  locked           — never writable at runtime. Not "requires approval":
                     there is no approval that grants it, because the
                     agent's own policy lives here. An agent that can
                     edit its own constraints has none.
  editable         — writable under normal authorization.
  append_only      — may be added to, never altered or removed. Audit
                     trails and queues: history you can extend but not
                     rewrite.
  human_controlled — only a human writes here. An agent may propose.

The Three Gates then decide whether an action is a candidate for
automation at all: how many paths lead to the outcome, how far the damage
reaches if it is wrong, and whether the result can be checked afterwards.
"""

from agentic_commons.config import get_settings
from agentic_commons.gatekeeper.types import GateEvaluation
from agentic_commons.logging import get_logger
from agentic_commons.tools.types import BlastRadius, SurfaceClass

# Surfaces no runtime authority can write to, whatever rank asks.
_NEVER_WRITABLE: frozenset[SurfaceClass] = frozenset({"locked"})

# Surfaces only a human may write to.
_HUMAN_ONLY: frozenset[SurfaceClass] = frozenset({"human_controlled"})

# Blast radii too wide to automate without a certifiable result.
_WIDE_RADII: frozenset[BlastRadius] = frozenset({"tenant", "global"})


class SurfaceClassifier:
    """Maps an action to its surface, and says who may write there."""

    def __init__(self) -> None:
        settings = get_settings().gatekeeper.surface_policy
        self._enabled = settings.enabled
        self._surfaces = settings.surfaces
        self._logger = get_logger("gatekeeper")

    def classify(self, action: str) -> SurfaceClass:
        """The surface class an action writes to.

        An unmapped action is treated as `human_controlled`, not
        `editable`: an action nobody has classified is one nobody has
        decided is safe to automate.
        """
        bare = action.rsplit(".", maxsplit=1)[-1]
        surface = self._surfaces.get(bare)
        if surface is None:
            self._logger.warning(
                "Unclassified action; defaulting to human-controlled", action=action
            )
            return "human_controlled"
        return surface

    def is_writable_by_agent(self, surface: SurfaceClass) -> tuple[bool, str]:
        """Whether an agent may write to a surface, and why not if not."""
        if not self._enabled:
            return True, "surface policy disabled"
        if surface in _NEVER_WRITABLE:
            return False, "locked surfaces are never writable at runtime"
        if surface in _HUMAN_ONLY:
            return False, "human-controlled surfaces accept writes only from a human"
        return True, f"'{surface}' is writable under authorization"

    def is_append_only(self, surface: SurfaceClass) -> bool:
        """Whether writes to this surface may only add, never modify."""
        return surface == "append_only"


class ThreeGatesEvaluator:
    """Path cardinality × blast radius × certifiability (E15).

    All three must pass. An open-ended path with a wide blast radius and
    no way to check the result is the profile of work that should not be
    automated — and naming which gate failed is what turns "this feels
    risky" into a decision someone can argue with.
    """

    def __init__(self) -> None:
        self._logger = get_logger("gatekeeper")

    def evaluate(
        self,
        path_cardinality: str = "bounded",
        blast_radius: BlastRadius = "single_record",
        certifiable: bool = True,
    ) -> GateEvaluation:
        """Assess one action against all three gates."""
        reasons: list[str] = []
        if path_cardinality == "open" and blast_radius in _WIDE_RADII:
            reasons.append("open-ended path with a wide blast radius")
        if not certifiable and blast_radius in _WIDE_RADII:
            reasons.append("wide blast radius with no certifiable result")
        if not certifiable and path_cardinality == "open":
            reasons.append("open-ended path with no certifiable result")
        passes = not reasons
        return GateEvaluation(
            path_cardinality=path_cardinality,  # type: ignore[arg-type]
            blast_radius=blast_radius,
            certifiable=certifiable,
            passes=passes,
            reason="; ".join(reasons) if reasons else "all three gates pass",
        )
