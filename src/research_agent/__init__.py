"""The research agent — the platform's second product.

An autonomous researcher that bounds a question with the user before it
searches, freezes what "done" means, researches against that contract
under hard budgets, and has an independent pass audit the result.

Importing this package declares its configuration layer, exactly as the
customer support agent does: the dependency points from the agent to the
platform, so the agent names its own overlay. `AGENTIC_OVERLAYS` still
wins when a deployment, the Makefile or a test has already decided.
"""

import os
from pathlib import Path

_CONFIG_FILE = Path(__file__).resolve().parent / "config" / "config.yaml"
_OVERLAYS_ENV = "AGENTIC_OVERLAYS"

CONFIG_FILE = _CONFIG_FILE


def _declare_config_layer() -> None:
    """Name this agent's overlay, unless the environment already did."""
    if not os.environ.get(_OVERLAYS_ENV):
        os.environ[_OVERLAYS_ENV] = str(_CONFIG_FILE)


_declare_config_layer()
