"""Hallucination detection, diagnosis and trust calibration (LLD §10)."""

from agentic_commons.hallucination_protection.checker import (
    COMBINED_TIER,
    COMBINED_TIER_JOINER,
    OBSERVATION_SEQUENCE,
    ClaimDecomposer,
    ClaimResolver,
    DetectionLadder,
    GroundingReasoningDiagnoser,
    GroundingResolver,
    HallucinationDetector,
    TrustCalibrationTracker,
)
from agentic_commons.hallucination_protection.types import (
    AtomicClaim,
    DetectionVerdict,
    EvidenceCheck,
    FailureKind,
    GenerationAssessment,
    TrustMeasurement,
    TrustReport,
)

__all__ = [
    "COMBINED_TIER",
    "COMBINED_TIER_JOINER",
    "OBSERVATION_SEQUENCE",
    "AtomicClaim",
    "ClaimDecomposer",
    "ClaimResolver",
    "DetectionLadder",
    "EvidenceCheck",
    "GroundingResolver",
    "DetectionVerdict",
    "FailureKind",
    "GenerationAssessment",
    "GroundingReasoningDiagnoser",
    "HallucinationDetector",
    "TrustCalibrationTracker",
    "TrustMeasurement",
    "TrustReport",
]
