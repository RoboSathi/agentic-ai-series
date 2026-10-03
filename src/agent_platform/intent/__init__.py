"""Reusable intent-classification contracts and providers."""

from agent_platform.intent.classifier import HuggingFaceIntentClassifier, create_intent_classifier
from agent_platform.intent.types import (
    IntentAssessment,
    IntentCandidate,
    IntentClassifierInterface,
    IntentDisposition,
)

__all__ = [
    "HuggingFaceIntentClassifier",
    "IntentAssessment",
    "IntentCandidate",
    "IntentClassifierInterface",
    "IntentDisposition",
    "create_intent_classifier",
]
