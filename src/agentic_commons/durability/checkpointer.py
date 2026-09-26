"""The checkpointer: one SQLite file of graph state, optionally encrypted.

Why a bridge over LangGraph's *sync* `SqliteSaver` instead of its
`AsyncSqliteSaver`: the async saver runs aiosqlite on a worker thread
whose lock is bound to the first event loop that touches it, and whose
thread keeps a process alive until it is closed. This platform opens
event loops in several places — the console's, `asyncio.run` in start-up
work, one per test — and a checkpointer that works in only one of them,
or hangs the process on exit, is not durable in any useful sense.
SQLite writes to a local file take milliseconds, so running them inline
is the same trade the rest of the platform's persistence already makes.

Encryption is `EncryptedSerializer` with AES from the key named in
config. Checkpoints hold the whole conversation, so plaintext is allowed
only where the configuration says a missing key is acceptable.
"""

import os
import sqlite3
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from agentic_commons.config import get_settings
from agentic_commons.config.settings import DurabilitySettings
from agentic_commons.logging import get_logger

# AES key lengths the cipher accepts, in bytes.
_AES_KEY_LENGTHS = (16, 24, 32)


class CheckpointKeyError(RuntimeError):
    """Encryption is required and no usable key was provided."""


class SqliteCheckpointer(SqliteSaver):
    """LangGraph's SQLite saver, callable from any event loop (see module docstring)."""

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)

    async def aget_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, Any]:
        return self.get_delta_channel_history(config=config, channels=channels)


def create_checkpointer(
    settings: DurabilitySettings | None = None,
    *,
    state_types: Iterable[type[BaseModel]] = (),
    path: Path | None = None,
) -> SqliteCheckpointer:
    """Build the configured checkpointer (Factory).

    `state_types` are the pydantic models the graph state is made of.
    They are registered with the serializer so reading a checkpoint back
    never deserializes a type nobody declared.
    """
    resolved = settings or get_settings().durability
    target = path or resolved.resolved_db_path
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(target), check_same_thread=False)
    saver = SqliteCheckpointer(connection, serde=_serializer(resolved, state_types))
    saver.setup()
    return saver


def model_types(root: type[BaseModel]) -> set[type[BaseModel]]:
    """Every pydantic model reachable from a state model's field annotations."""
    found: set[type[BaseModel]] = set()
    pending: list[Any] = [root]
    while pending:
        annotation = pending.pop()
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if annotation in found:
                continue
            found.add(annotation)
            pending.extend(field.annotation for field in annotation.model_fields.values())
        pending.extend(getattr(annotation, "__args__", ()))
    return found


def _serializer(
    settings: DurabilitySettings, state_types: Iterable[type[BaseModel]]
) -> SerializerProtocol:
    allowed = sorted({(cls.__module__, cls.__name__) for cls in state_types})
    base = JsonPlusSerializer(allowed_msgpack_modules=allowed or True)
    key = os.environ.get(settings.encryption.key_env, "").encode("utf-8")
    if len(key) in _AES_KEY_LENGTHS:
        return EncryptedSerializer.from_pycryptodome_aes(serde=base, key=key)
    reason = (
        f"{settings.encryption.key_env} is unset"
        if not key
        else f"{settings.encryption.key_env} must be {_AES_KEY_LENGTHS} bytes, got {len(key)}"
    )
    if settings.encryption.required:
        raise CheckpointKeyError(f"Checkpoint encryption is required but {reason}")
    get_logger("persistence").warning(
        "Checkpoints are stored UNENCRYPTED — set the key to encrypt them at rest", reason=reason
    )
    return base
