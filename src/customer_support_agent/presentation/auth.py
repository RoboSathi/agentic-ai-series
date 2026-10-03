"""The fixed wording of verification: the prompt for a code, and its refusal.

Configured, not written here (`customer_support_agent.messages.auth`), and
rendered by the graph's authentication node. The authenticator decides who
is speaking; it never decides what they read.
"""

from customer_support_agent.config import get_agent_settings


def ask_for_code() -> str:
    """The prompt shown until a code arrives."""
    settings = get_agent_settings()
    return settings.messages.auth.ask_for_code.format(length=settings.user_code_length)


def unknown_code() -> str:
    """The one reply for a malformed or unknown code — they must read alike."""
    settings = get_agent_settings()
    return settings.messages.auth.unknown_code.format(length=settings.user_code_length)


__all__ = ["ask_for_code", "unknown_code"]
