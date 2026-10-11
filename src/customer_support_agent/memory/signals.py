"""What a support turn tells the skill selector — and the dream — about itself.

Its intent and its route, written the way a SKILL.md trigger names them
(`intent:REQUEST_REFUND`). The skill selector reads them to choose a
turn's skills; the turn's dream event carries them as its scope, so a
skill learned from refusals in refund turns triggers on refund turns.
"""

from collections.abc import Mapping
from typing import Any

from customer_support_agent.config import get_agent_settings


def turn_signals(metadata: Mapping[str, Any]) -> frozenset[str]:
    """The turn's signals; a metadata key the turn did not set contributes nothing."""
    templates = get_agent_settings().instruction_signals
    return frozenset(
        template.format(value=metadata[key])
        for key, template in templates.items()
        if metadata.get(key)
    )
