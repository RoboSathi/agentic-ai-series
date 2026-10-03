"""`make clean-memory`: empty this agent's long-term memory.

Episodic logs, semantic facts, procedures, the curator queue and the dream
inbox. Checkpoints stay, so paused runs can resume; orders, refunds,
reviews, audit, policy and the knowledge index are not memory.
"""

from agent_platform.config import get_settings
from agent_platform.memory.reset import clear_long_term_memory
from agent_platform.presentation.text_tree import TreeNode, tree_lines
from customer_support_agent.admin._store import agent_store
from customer_support_agent.admin.show_memory import print_lines

# What a reset deliberately leaves alone, said out loud so nobody wonders.
_KEPT = "Kept: conversation checkpoints, orders, refunds, reviews, audit, policy, knowledge index"
_RESTART = "Restart a running TUI: its working memory lives in the process."


def main() -> None:
    store = agent_store()
    removed = clear_long_term_memory(store, get_settings().memory.episodic.resolved_path)
    root = TreeNode(f"Cleared long-term memory · {store.agent_id} · {store.db_path}")
    for label, count in removed.items():
        root.add(f"{label}: {count} removed")
    print_lines([*tree_lines(root), _KEPT, _RESTART])


if __name__ == "__main__":
    main()
