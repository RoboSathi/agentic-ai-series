"""Surface classification and the Three Gates (E12/E15, LLD §11).

Four surface classes, distinguished by what *may* be written to them:

  locked           — never writable at runtime.
  editable         — writable under normal authorization.
  append_only      — may be added to, never altered or removed.
  human_controlled — only a human writes here.

The Three Gates then evaluate the declared execution profile.  The evaluator
keeps the existing policy semantics and now exposes each dimension separately
so operators can see exactly why an aggregate result passed or failed.
"""

from agentic_commons.config import get_settings
from agentic_commons.gatekeeper.types import CheckReport, GateEvaluation
from agentic_commons.logging import get_logger
from agentic_commons.tools.types import BlastRadius, PathCardinality, SurfaceClass

_NEVER_WRITABLE: frozenset[SurfaceClass] = frozenset({"locked"})
_HUMAN_ONLY: frozenset[SurfaceClass] = frozenset({"human_controlled"})
_WIDE_RADII: frozenset[BlastRadius] = frozenset({"tenant", "global"})


class SurfaceClassifier:
    """Maps an action to a surface, and says who may write there."""

    def __init__(self) -> None:
        settings = get_settings().gatekeeper.surface_policy
        self._enabled = settings.enabled
        self._surfaces = settings.surfaces
        self._logger = get_logger("gatekeeper")

    def classify(self, action: str) -> SurfaceClass:
        """The surface class an action writes to.

        An unmapped action is treated as `human_controlled`, not `editable`:
        an action nobody has classified is one nobody has decided is safe to
        automate.
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

    The aggregate rules intentionally remain the same as the existing
    evaluator.  What changes is the report: the operator can now see the
    concrete values and the individual dimensions implicated by a failure.
    """

    def evaluate(
        self,
        path_cardinality: PathCardinality = "bounded",
        blast_radius: BlastRadius = "single_record",
        certifiable: bool = True,
    ) -> GateEvaluation:
        """Assess one action against all three gates."""
        reasons: list[str] = []
        path_failed = False
        blast_failed = False
        certifiability_failed = False

        if path_cardinality == "open" and blast_radius in _WIDE_RADII:
            reasons.append("open-ended path with a wide blast radius")
            path_failed = True
            blast_failed = True
        if not certifiable and blast_radius in _WIDE_RADII:
            reasons.append("wide blast radius with no certifiable result")
            blast_failed = True
            certifiability_failed = True
        if not certifiable and path_cardinality == "open":
            reasons.append("open-ended path with no certifiable result")
            path_failed = True
            certifiability_failed = True

        path_check = CheckReport(
            check_id="three_gates.path_cardinality",
            name="Path cardinality",
            status="FAIL" if path_failed else "PASS",
            reason_code="THREE_GATES_PATH_RISK" if path_failed else "PATH_CARDINALITY_ACCEPTED",
            reason=(
                "open path + wide blast radius"
                if path_failed
                else "accepted"
            ),
            observed={"value": path_cardinality},
        )
        blast_check = CheckReport(
            check_id="three_gates.blast_radius",
            name="Blast radius",
            status="FAIL" if blast_failed else "PASS",
            reason_code="THREE_GATES_BLAST_RADIUS_RISK" if blast_failed else "BLAST_RADIUS_ACCEPTED",
            reason=(
                "wide blast radius contributes to a failing rule"
                if blast_failed
                else "accepted"
            ),
            observed={"value": blast_radius},
        )
        cert_check = CheckReport(
            check_id="three_gates.certifiability",
            name="Certifiability",
            status="FAIL" if certifiability_failed else "PASS",
            reason_code=(
                "THREE_GATES_NOT_CERTIFIABLE"
                if certifiability_failed
                else "CERTIFIABLE_RESULT"
            ),
            reason=(
                "result is not deterministically certifiable"
                if certifiability_failed
                else "deterministically certifiable"
            ),
            observed={"value": certifiable},
        )

        passes = not reasons
        result = GateEvaluation(
            path_cardinality=path_cardinality,
            blast_radius=blast_radius,
            certifiable=certifiable,
            passes=passes,
            reason="; ".join(reasons) if reasons else "all three gates pass",
            path_check=path_check,
            blast_radius_check=blast_check,
            certifiability_check=cert_check,
        )
        return result
