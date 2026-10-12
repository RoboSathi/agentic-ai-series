"""The deterministic oracle: an agent's rules over the outcome, zero model calls."""

from agent_platform.evals.oracle.oracle import InvariantOracle, Rule, RuleVerdict
from agent_platform.evals.oracle.report import OracleReport

__all__ = ["InvariantOracle", "OracleReport", "Rule", "RuleVerdict"]
