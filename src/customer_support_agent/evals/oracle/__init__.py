"""The Customer Support oracle: this agent's rules on the platform's `InvariantOracle`."""

import re
from functools import partial

from agent_platform.evals.core.contracts import GoldenCase, Transcript
from agent_platform.evals.oracle import InvariantOracle
from customer_support_agent.config.settings import CsaEvalSettings, get_agent_settings
from customer_support_agent.evals.dataset.schema import CsaCasePayload
from customer_support_agent.evals.fixture import FixtureFacts
from customer_support_agent.evals.oracle.rules import RULES, OracleContext

ORACLE_NAME = "csa_oracle"


def csa_oracle(
    facts: FixtureFacts, settings: CsaEvalSettings | None = None
) -> InvariantOracle[OracleContext]:
    """The oracle over these fixture facts; any change to them, the rules or the patterns regrades."""
    evals = settings or get_agent_settings().evals
    return InvariantOracle(
        ORACLE_NAME,
        RULES,
        partial(_context, facts, evals),
        version_inputs=[
            facts.model_dump(mode="json"),
            evals.order_reference_pattern,
            evals.money_pattern,
            sorted(evals.money_tools),
        ],
    )


def _context(
    facts: FixtureFacts, evals: CsaEvalSettings, case: GoldenCase, transcript: Transcript
) -> OracleContext:
    payload = CsaCasePayload.of(case)
    return OracleContext(
        case=case,
        expected=payload.expected,
        transcript=transcript,
        facts=facts,
        customer=facts.user(payload.customer),
        order_pattern=re.compile(evals.order_reference_pattern),
        money_pattern=re.compile(evals.money_pattern),
        money_tools=frozenset(evals.money_tools),
    )


__all__ = ["ORACLE_NAME", "RULES", "OracleContext", "csa_oracle"]
