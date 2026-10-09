"""LangSmith export: one trace per turn, redacted before it leaves the process.

A turn opens one root run, whose id *is* the turn's log `trace_id`, so a
line in the file log and a run on the LangSmith dashboard are found by the
same id. Everything the turn causes nests under that root without being
told to: the LangGraph nodes (langchain_core reuses the root's client), the
model calls (`instrument_sdk_client`) and the tool executions (`tool_run`).
Outside a turn, or with export withheld, those hooks do nothing.

Two rules hold on every path:

- Nothing unredacted leaves the machine. The client's input and output
  hooks run the agent's PII redactor over every string of every run; the
  file log keeps the full payload, as it always has.
- Export never fails a turn. A missing key, a dead endpoint or a rejected
  batch is logged and the turn carries on — the local trace and audit
  trail are the record, this is a view of it.
"""

import atexit
import json
import random
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from agent_platform.config import get_settings
from agent_platform.config.settings import LangSmithSettings
from agent_platform.logging import bound_log_context, get_logger, new_trace_id
from agent_platform.observability.units import turn_bracket

T = TypeVar("T")
TextRedactor = Callable[[str], str]


class TurnTrace(Protocol):
    """The root run of one turn, as the agent sees it."""

    def finish(self, outputs: Mapping[str, Any]) -> None:
        """Record what the turn produced (redacted on export)."""

    def annotate(self, metadata: Mapping[str, Any]) -> None:
        """Facts about the turn to filter and group by on the dashboard (intent, outcome)."""


class TraceExporter(Protocol):
    """Opens a turn's root run; the agent never learns which backend, if any."""

    def turn(
        self, trace_id: str, inputs: Mapping[str, Any], metadata: Mapping[str, Any]
    ) -> AbstractContextManager[TurnTrace]:
        """Scope one turn's root run to a block."""

    def url_for(self, trace_id: str) -> str | None:
        """Where the turn's trace can be opened, once the backend says so."""

    def flush(self) -> None:
        """Send whatever is still queued."""


_current_turn: ContextVar[tuple[TraceExporter, str] | None] = ContextVar(
    "langsmith_turn", default=None
)


class _NoTurn:
    def finish(self, outputs: Mapping[str, Any]) -> None:
        return None

    def annotate(self, metadata: Mapping[str, Any]) -> None:
        return None


class NullTraceExporter:
    """Export withheld: every turn runs exactly as it would with no exporter."""

    def turn(
        self, trace_id: str, inputs: Mapping[str, Any], metadata: Mapping[str, Any]
    ) -> AbstractContextManager[TurnTrace]:
        return nullcontext(_NoTurn())

    def url_for(self, trace_id: str) -> str | None:
        return None

    def flush(self) -> None:
        return None


class _RootRun:
    """Adapts the SDK's run tree to `TurnTrace`."""

    def __init__(self, run: Any) -> None:
        self._run = run

    def finish(self, outputs: Mapping[str, Any]) -> None:
        self._run.end(outputs=dict(outputs))

    def annotate(self, metadata: Mapping[str, Any]) -> None:
        self._run.add_metadata(dict(metadata))


class _ProjectLink:
    """The workspace and project ids a run URL needs, resolved once, off the turn's path.

    Both come from the backend (the project is created by the first run that
    names it), so they are looked up on a daemon thread the first time a turn
    starts. Until they arrive a turn simply has no URL; it never waits for one.
    """

    def __init__(self, client: Any, settings: LangSmithSettings, project: str) -> None:
        self._client = client
        self._settings = settings
        self._project = project
        self._ids: tuple[str, str] | None = None
        self._started = threading.Event()
        self._logger = get_logger("observability")

    def resolve_in_background(self) -> None:
        if not self._started.is_set():
            self._started.set()
            threading.Thread(target=self._resolve, daemon=True).start()

    def url(self, trace_id: str) -> str | None:
        if self._ids is None:
            return None
        tenant_id, project_id = self._ids
        run_id = str(uuid.UUID(hex=trace_id))
        return self._settings.run_url.format(
            web_url=self._settings.web_url,
            tenant_id=tenant_id,
            project_id=project_id,
            run_id=run_id,
        )

    def _resolve(self) -> None:
        try:
            project = self._client.create_project(self._project, upsert=True)
            self._ids = (str(project.tenant_id), str(project.id))
        except Exception as exc:
            self._logger.warning("LangSmith project lookup failed; no trace links", error=str(exc))


class LangSmithExporter:
    """Exports each turn to one LangSmith project, redacting every payload first."""

    def __init__(
        self,
        client: Any,
        settings: LangSmithSettings,
        agent_id: str,
    ) -> None:
        self._client = client
        self._settings = settings
        self._project = settings.project.format(agent_id=agent_id)
        self._root_name = settings.root_run_name.format(agent_id=agent_id)
        self._link = _ProjectLink(client, settings, self._project)
        self._sample_rate = settings.sample_rate
        # Turns in flight that were sampled out: they have no trace, so no link.
        self._skipped: set[str] = set()
        self._logger = get_logger("observability")

    @contextmanager
    def turn(
        self, trace_id: str, inputs: Mapping[str, Any], metadata: Mapping[str, Any]
    ) -> Iterator[TurnTrace]:
        from langsmith import trace, tracing_context

        if random.random() >= self._sample_rate:
            self._skipped.add(trace_id)
            try:
                yield _NoTurn()
            finally:
                self._skipped.discard(trace_id)
            return
        self._link.resolve_in_background()
        with (
            tracing_context(enabled=True, client=self._client, project_name=self._project),
            trace(
                self._root_name,
                inputs=dict(inputs),
                run_id=uuid.UUID(hex=trace_id),
                metadata=dict(metadata),
                project_name=self._project,
                client=self._client,
            ) as run,
        ):
            yield _RootRun(run)

    def url_for(self, trace_id: str) -> str | None:
        return None if trace_id in self._skipped else self._link.url(trace_id)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as exc:
            self._logger.warning("LangSmith flush failed", error=str(exc))


def redact_tree(value: Any, redact: TextRedactor) -> Any:
    """Every string in a payload passed through the redactor; the shape is kept.

    The payload is made plain JSON first. A graph node's input is the agent's
    state *model*, not a dict, and a walk that only knew dicts and lists
    passed it — and every customer string inside it — through untouched.
    Whatever type arrives, nothing reaches the walk that it cannot see into.
    """
    return _redact_plain(json.loads(json.dumps(value, default=_plain)), redact)


def _plain(value: Any) -> Any:
    """A JSON-encodable stand-in for an object `json` cannot encode."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (set, frozenset)):
        return list(value)
    return str(value)


def _redact_plain(value: Any, redact: TextRedactor) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: _redact_plain(item, redact) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_plain(item, redact) for item in value]
    return value


def is_routing_run(name: str, run_type: str, extra: Mapping[str, Any] | None) -> bool:
    """Whether a run is a graph's conditional edge rather than a step.

    LangGraph traces every routing function (`_after_reason`) as a chain run
    of the node it leaves, carrying that node's whole state again. The
    decision it made is already visible as the next step, so it is noise:
    a chain run inside a node (`langgraph_node` set) that is not the node.
    """
    node = _node_of(extra)
    return run_type == "chain" and node is not None and name != node


def is_node_run(name: str, run_type: str, extra: Mapping[str, Any] | None) -> bool:
    """Whether a run is one step of a graph (its name is the node it runs)."""
    return run_type == "chain" and name == _node_of(extra)


def _node_of(extra: Mapping[str, Any] | None) -> Any:
    return ((extra or {}).get("metadata") or {}).get("langgraph_node")


def _routing_free_client() -> type[Any]:
    """The SDK client, minus the graph's routing runs, and steps sent as what they changed.

    A step's input is the whole state so far, which every earlier step's
    output already shows; its output is exactly what it changed. With
    `full_node_state` off a step is sent with its output alone, so the trace
    reads as a sequence of changes instead of the same state again and again.
    """
    from langsmith import Client

    class RoutingFreeClient(Client):
        def __init__(self, *, full_node_state: bool, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._full_node_state = full_node_state
            self._dropped: set[str] = set()

        def create_run(self, *args: Any, **kwargs: Any) -> Any:
            name, run_type = kwargs.get("name", ""), kwargs.get("run_type", "")
            extra = kwargs.get("extra")
            if is_routing_run(name, run_type, extra):
                self._dropped.add(str(kwargs.get("id")))
                return None
            if not self._full_node_state and is_node_run(name, run_type, extra):
                kwargs["inputs"] = {}
            return super().create_run(*args, **kwargs)

        def update_run(self, run_id: Any, **kwargs: Any) -> Any:
            if str(run_id) in self._dropped:
                self._dropped.discard(str(run_id))
                return None
            # The tracer sends a step's input again when it closes the run.
            name, run_type = kwargs.get("name", ""), kwargs.get("run_type", "")
            if not self._full_node_state and is_node_run(name, run_type, kwargs.get("extra")):
                kwargs["inputs"] = None
            return super().update_run(run_id, **kwargs)

    return RoutingFreeClient


def build_trace_exporter(agent_id: str, redactor: Callable[[], TextRedactor]) -> TraceExporter:
    """The configured exporter for one agent — a no-op when withheld or keyless.

    Called once, in the composition root. `redactor` is built only when
    export is on, so a withheld exporter costs no PII model. No network is
    touched here: the client batches in the background and the project link
    resolves on the first turn, so building an agent (a test, `make verify`)
    stays offline.
    """
    settings = get_settings().observability.langsmith
    logger = get_logger("observability")
    if not settings.enabled:
        return NullTraceExporter()
    api_key = settings.api_key()
    if not api_key:
        logger.warning("LangSmith export enabled but no key is set; exporting nothing")
        return NullTraceExporter()
    redact = redactor()

    def hide(payload: dict[str, Any]) -> dict[str, Any]:
        return dict(redact_tree(payload, redact))

    def on_error(exc: Exception) -> None:
        logger.warning("LangSmith export failed", error=str(exc))

    client = _routing_free_client()(
        full_node_state=settings.full_node_state,
        api_url=settings.endpoint,
        api_key=api_key,
        web_url=settings.web_url,
        hide_inputs=hide,
        hide_outputs=hide,
        hide_metadata=hide,
        tracing_error_callback=on_error,
    )
    exporter = LangSmithExporter(client, settings, agent_id)
    # Runs are sent in batches; the last turn's must not die with the process.
    atexit.register(exporter.flush)
    return exporter


@contextmanager
def turn_trace(
    exporter: TraceExporter,
    *,
    agent: str,
    session: str,
    inputs: Mapping[str, Any],
    message: bool = True,
) -> Iterator[TurnTrace]:
    """One turn: its log context, its root run and its analytics bracket, under one id.

    While the block runs, `current_trace_url()` answers with the turn's link,
    which the turn summary carries; the link is logged once the turn ends. The
    bracket (`units.turn_bracket`) times the turn and appends its turn and task
    facts however it ends, so no agent writes them. `message` is False for a turn
    with no message from a person in it (a resumed run).
    """
    trace_id = new_trace_id()
    with (
        bound_log_context(agent=agent, session=session, trace_id=trace_id),
        turn_bracket(agent=agent, session=session, turn_id=trace_id, message=message),
        exporter.turn(trace_id, inputs, _turn_metadata(agent, session)) as root,
    ):
        token = _current_turn.set((exporter, trace_id))
        try:
            yield root
        finally:
            _current_turn.reset(token)
            url = exporter.url_for(trace_id)
            if url is not None:
                get_logger("observability").info("LangSmith trace", url=url)


def _turn_metadata(agent: str, session: str) -> dict[str, Any]:
    """What every turn's root run carries: the conversation it belongs to, and what produced it.

    The session id under the configured thread key is what makes LangSmith
    group a conversation's turns into one thread. The versions are the audit
    trail's, so a regression can be bisected on the dashboard by what moved.
    """
    versions = get_settings().audit
    return {
        get_settings().observability.langsmith.thread_key: session,
        "agent": agent,
        "policy_version": versions.policy_version,
        "prompt_version": versions.prompt_version,
        "data_version": versions.data_version,
        "tool_schema_version": versions.tool_schema_version,
    }


def current_trace_url() -> str | None:
    """The link of the turn running in this context — None outside a turn or with no export."""
    turn = _current_turn.get()
    return turn[0].url_for(turn[1]) if turn is not None else None


def instrument_sdk_client(client: T) -> T:
    """The provider SDK client, reporting each call as a model run inside a turn.

    Inert outside a turn's root run, so it is safe to apply unconditionally.
    """
    from langsmith.wrappers import wrap_openai

    return wrap_openai(client)


def tool_run(name: str, func: Callable[..., T]) -> Callable[..., T]:
    """`func` reported as one tool run named for the action, inside a turn."""
    from langsmith import traceable

    return traceable(run_type="tool", name=name)(func)


def traced_tool(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator: a method reported as one tool run named for itself, inside a turn."""
    from langsmith import traceable

    return traceable(run_type="tool")(func)


__all__ = [
    "LangSmithExporter",
    "NullTraceExporter",
    "TraceExporter",
    "TurnTrace",
    "build_trace_exporter",
    "current_trace_url",
    "instrument_sdk_client",
    "redact_tree",
    "tool_run",
    "traced_tool",
    "turn_trace",
]
