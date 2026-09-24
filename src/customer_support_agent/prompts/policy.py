"""The refund policy this agent is asked to enforce.

Resolved package-relative rather than from the shell's working
directory: a CWD-relative path works until somebody runs the agent from
anywhere but the repository root.
"""

from customer_support_agent.config import get_agent_settings


def policy_text() -> str:
    """The refund policy, as written."""
    return get_agent_settings().resolved_policy_path.read_text()
