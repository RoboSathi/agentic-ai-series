"""Confirming that a side effect actually happened (LLD §24).

"The executor returned success" and "the money moved" are different
claims, and nothing in this harness used to check the second one. This
package does: after a side-effecting call, the thing that should have
changed is read back and compared against what the action said it did.

A result that cannot be verified never becomes authoritative, so no
customer-facing claim is grounded on it.
"""

from agent_platform.runtime.verification.types import IntendedEffect, VerificationReport
from agent_platform.runtime.verification.verifier import (
    EffectSource,
    EffectVerifier,
    ReadBackVerifier,
)

__all__ = [
    "EffectSource",
    "EffectVerifier",
    "IntendedEffect",
    "ReadBackVerifier",
    "VerificationReport",
]
