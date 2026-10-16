"""What an unmarked test may not reach: a remote host, or a local model's weights.

The default run is the fast, offline one (`pyproject.toml` deselects the `models` and
`network` markers). A test that strays from it used to wait on a socket until its client's
own timeout, or spend minutes loading — or downloading — a checkpoint, and the suite took a
quarter of an hour. Now the stray call fails at once, naming what it reached, and the test is
either given a fake collaborator or marked for the run that allows it.
"""

import socket
import sys
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from ipaddress import ip_address
from pathlib import Path
from typing import Any

import pytest

MODELS = "models"
NETWORK = "network"
# Library seams that open a local checkpoint or fetch one, by dotted path.
MODEL_SEAMS = (
    "agent_platform.llm.local_models.LocalSequenceClassifier._load_or_download",
    "agent_platform.llm.local_laya.fetch_checkpoint",
    "agent_platform.llm.tokenizer.LocalTokenizer._load",
    "agent_platform.security.pii.presidio._build_analyzer",
    "agent_platform.security.pii.presidio.ensure_pipeline",
)
# The Laya package, imported only inside `local_laya.resident_laya`: stood in for by a module
# whose `load` refuses, so a test that installs its own fake `laya` still can.
LAYA_MODULE = "laya"
_LOCAL_NAMES = frozenset({"localhost"})


class UnmarkedNetworkAccessError(RuntimeError):
    """An unmarked test opened a connection to a remote host."""


class UnmarkedModelLoadError(RuntimeError):
    """An unmarked test reached a local model's load or download."""


@contextmanager
def hermetic(item: pytest.Item) -> Iterator[None]:
    """Refuse remote sockets and model loads for the length of one test, unless marked."""
    with pytest.MonkeyPatch.context() as patch:
        if item.get_closest_marker(NETWORK) is None:
            _refuse_remote_sockets(patch)
        if item.get_closest_marker(MODELS) is None:
            for seam in MODEL_SEAMS:
                patch.setattr(seam, _refusing(seam))
            patch.setitem(sys.modules, LAYA_MODULE, _refusing_module(LAYA_MODULE))
        yield


def _refuse_remote_sockets(patch: pytest.MonkeyPatch) -> None:
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex

    def guarded(original: Any) -> Any:
        def call(self: socket.socket, address: Any) -> Any:
            if self.family in (socket.AF_INET, socket.AF_INET6) and not _is_local(address):
                raise UnmarkedNetworkAccessError(
                    f"{address!r}: mark the test @pytest.mark.network, or fake the client"
                )
            return original(self, address)

        return call

    patch.setattr(socket.socket, "connect", guarded(connect))
    patch.setattr(socket.socket, "connect_ex", guarded(connect_ex))


def _is_local(address: Any) -> bool:
    host = str(address[0]) if isinstance(address, tuple) else str(address)
    if host in _LOCAL_NAMES:
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _refusing(seam: str) -> Any:
    def refuse(*_: Any, **__: Any) -> Any:
        raise UnmarkedModelLoadError(
            f"{seam}: mark the test @pytest.mark.models, or fake the model"
        )

    return refuse


def _refusing_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.load = _refusing(f"{name}.load")  # type: ignore[attr-defined]
    return module


def needs_model(artifact: Path, reason: str) -> Callable[[Any], Any]:
    """Mark a test `models`, and skip it where the artifact was never downloaded."""
    skip = pytest.mark.skipif(not artifact.exists(), reason=f"{reason} (make models)")

    def mark(test: Any) -> Any:
        return pytest.mark.models(skip(test))

    return mark
