"""Wiping long-term memory back to empty — `make clean-memory`.

Long-term only, by decision: the four CoALA stores' durable halves, the
curator queue and the dream inbox. Conversation checkpoints stay, so a
paused run (a refund waiting on a reviewer) can still resume; business
data and the knowledge index are not memory and are never touched.

Each store purges its own rows — table names stay with the class that
owns them — and this module only decides which stores a reset covers.
"""

from pathlib import Path

from agentic_commons.memory.episodic.memory import EpisodicStore
from agentic_commons.memory.events import DreamInbox
from agentic_commons.memory.governance import CuratorQueue
from agentic_commons.memory.procedural.memory import ProceduralStore
from agentic_commons.memory.semantic.memory import SemanticStore
from agentic_commons.persistence import PlatformStore


def clear_long_term_memory(store: PlatformStore, episodic_root: Path) -> dict[str, int]:
    """Purge every long-term memory store for this agent; returns what each removed."""
    return {
        "episodic session logs": EpisodicStore(episodic_root).purge(),
        "semantic facts": SemanticStore(store).purge(),
        "procedures": ProceduralStore(store).purge(),
        "curator queue entries": CuratorQueue(store).purge(),
        "dream inbox events": DreamInbox(store).purge(),
    }
