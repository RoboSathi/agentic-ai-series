"""Demo data for the pattern library.

Each folder that needs demo data owns its own. The patterns are meant to
be readable in isolation, so they do not reach into the running
product's database — a reader should be able to understand a pattern
without first understanding a refund agent.
"""

from pathlib import Path

DUMMY_DATA_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE = DUMMY_DATA_DIR / "knowledge_base.md"
TICKETS = DUMMY_DATA_DIR / "tickets.json"
