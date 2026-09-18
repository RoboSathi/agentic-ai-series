"""Demo data shared by the episode applications.

Episodes own their demo data here rather than reaching into the running
product's database: an episode should be understandable without first
understanding a refund agent, and it must not depend on data another
folder is free to regenerate.
"""

import json
from pathlib import Path

from agentic_commons.context import Chunk

DUMMY_DATA_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE = DUMMY_DATA_DIR / "knowledge_base.json"


def knowledge_chunks() -> list[Chunk]:
    """The small support corpus, as retrievable chunks."""
    entries = json.loads(KNOWLEDGE_BASE.read_text())
    return [
        Chunk(content=entry["text"], source=entry["topic"], kind="retrieved") for entry in entries
    ]
