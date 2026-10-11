"""Wiping long-term memory back to empty — `make clean-memory`.

Long-term only, by decision: the four CoALA stores' durable halves, the
curator queue and the dream inbox. Conversation checkpoints stay, so a
paused run (a refund waiting on a reviewer) can still resume; business
data and the knowledge index are not memory and are never touched.

Each store purges its own rows — table names stay with the class that
owns them — and this module only decides which stores a reset covers.
"""

from agent_platform.memory.dream.inbox import FileDreamInbox
from agent_platform.memory.episodic.memory import EpisodicStore
from agent_platform.memory.governance import CuratorQueue
from agent_platform.memory.procedural.store import SkillStore
from agent_platform.memory.semantic.memory import SemanticStore
from agent_platform.persistence import PlatformStore


def clear_long_term_memory(store: PlatformStore) -> dict[str, int]:
    """Purge every long-term memory store for this agent; returns what each removed."""
    return {
        "episodic session logs": EpisodicStore().purge(),
        "semantic facts": SemanticStore().purge(),
        "skills": SkillStore().purge(),
        "curator queue entries": CuratorQueue(store).purge(),
        "dream events": FileDreamInbox().purge(),
    }
