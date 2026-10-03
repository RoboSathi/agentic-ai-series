"""The customer support agent — the series' running product (LLD §21).

Importing this package declares its configuration layer.

That declaration used to live the other way round: `agent_platform`'
own `config.yaml` listed this agent's overlay, which meant the platform
could not be read without this agent installed, and that every future
agent would inherit this one's vocabulary. The dependency has to point
from the agent to the platform, so the agent names its own config here.

`AGENTIC_OVERLAYS` still wins when it is already set. A deployment, an
episode or a test that has decided which layers apply has decided; this
is the default for the ordinary case where nobody said otherwise.
"""

import os
from pathlib import Path

_CONFIG_FILE = Path(__file__).resolve().parent / "config" / "config.yaml"
_OVERLAYS_ENV = "AGENTIC_OVERLAYS"


def _declare_config_layer() -> None:
    """Name this agent's overlay, unless the environment already did."""
    if not os.environ.get(_OVERLAYS_ENV):
        os.environ[_OVERLAYS_ENV] = str(_CONFIG_FILE)


_declare_config_layer()
