"""The support agent's memory customisations (ADR 0011).

Everything memory does is the platform's; this package holds only what
is the support agent's own: how its turns are scoped for skills
(`signals`), and how its sessions are summarised (`profile`).
"""

from customer_support_agent.memory.profile import support_memory_profile
from customer_support_agent.memory.signals import turn_signals

__all__ = ["support_memory_profile", "turn_signals"]
