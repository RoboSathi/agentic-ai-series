"""A request is built from named sections, and a CONTEXT card says what each costs.

What is under test is the harness's own accounting: which section a piece of text lands in,
how many tokens it is counted as (by a fixed four-characters-per-token estimate, so counts
are expectations rather than measurements), what the stable prefix is, and that exactly one
CONTEXT card is drawn per request, before its LLM REQUEST card. No model is called.
"""

from types import SimpleNamespace
from typing import Any

from agent_platform.config import get_settings
from agent_platform.config.settings import TokenizerSettings
from agent_platform.context.types import Chunk
from agent_platform.llm.call_log import SteppedClient
from agent_platform.llm.client import OpenAICompatibleClient
from agent_platform.llm.prompt import (
    CONTEXT_WINDOW,
    HISTORY,
    POLICY,
    PROCEDURES,
    PROFILE,
    TOOL_LOOP,
    USER_QUERY,
    PromptBuilder,
    PromptPart,
    TurnLayout,
    manifest_from_messages,
)
from agent_platform.llm.tokenizer import CharEstimateTokenizer, LocalTokenizer
from agent_platform.llm.types import CompletionRequest, Message, ModelToolCall, TokenUsage
from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter
from agent_platform.observability.context_card import CONTEXT_STAGE, context_fields
from agent_platform.observability.model_observer import StageReporterModelObserver
from agent_platform.observability.step_events import LLM_REQUEST, StepEvents
from agent_platform.observability.types import StatusEvent
from tests.conftest import ScriptedLLM

FOUR = CharEstimateTokenizer(4)
INSTRUCTIONS = [
    PromptPart(PROFILE, "You are a support agent.", "AGENT.md"),
    PromptPart(PROCEDURES, "# Procedure: refund\nsteps", "refund@1"),
    PromptPart(PROCEDURES, "# Procedure: lookup\nsteps", "lookup@1"),
]
CHUNKS = [
    Chunk(content="customer_id: 1", kind="memory", source="working_memory", pinned=True),
    Chunk(content="order 10001 delivered", kind="retrieved", source="order:10001"),
    Chunk(content="asked about shipping", kind="memory", source="memory:episodic"),
]
TOOLS = [{"type": "function", "function": {"name": "process_refund", "parameters": {}}}]


def _built() -> Any:
    return (
        PromptBuilder(FOUR)
        .instructions(INSTRUCTIONS)
        .system(POLICY, "Refund within 30 days.")
        .window("# Retrieved context", "\n\n".join(c.content for c in CHUNKS), CHUNKS)
        .message(USER_QUERY, "user", "refund 10001")
        .tools(TOOLS)
        .build()
    )


def _feed() -> tuple[StageReporter, list[StatusEvent]]:
    feed = InMemoryLiveStatusFeed(max_events=0)
    shown: list[StatusEvent] = []
    feed.subscribe(shown.append)
    return StageReporter(feed, "t"), shown


class TestSectionsAreNamedAndCounted:
    def test_sections_keep_the_order_the_model_reads_them(self) -> None:
        names = [section.name for section in _built().manifest.sections]
        assert names == [PROFILE, PROCEDURES, POLICY, CONTEXT_WINDOW, USER_QUERY]

    def test_the_system_message_is_every_system_section_joined(self) -> None:
        messages = _built().messages
        assert [m.role for m in messages] == ["system", "user"]
        assert messages[0].content.startswith("You are a support agent.\n\n# Procedure: refund")
        assert messages[0].content.endswith("asked about shipping")

    def test_each_section_is_counted_by_the_tokenizer(self) -> None:
        policy = _built().manifest.sections[2]
        assert policy.tokens == FOUR.count("Refund within 30 days.") == 6

    def test_the_total_includes_the_tool_schemas(self) -> None:
        manifest = _built().manifest
        sections = sum(section.tokens for section in manifest.sections)
        assert manifest.tool_schema_tokens > 0
        assert manifest.total_tokens == sections + manifest.tool_schema_tokens
        assert manifest.tools == ["process_refund"]

    def test_each_procedure_is_its_own_part(self) -> None:
        procedures = _built().manifest.sections[1]
        assert [part.source for part in procedures.parts] == ["refund@1", "lookup@1"]

    def test_the_window_is_broken_down_by_what_it_holds(self) -> None:
        window = _built().manifest.sections[3]
        names = [part.name for part in window.parts]
        assert names == ["working_memory", "retrieved", "long_term_memory"]

    def test_empty_text_adds_no_section(self) -> None:
        manifest = PromptBuilder(FOUR).system(POLICY, "  ").build().manifest
        assert manifest.sections == []


class TestTheStablePrefix:
    def test_it_stops_at_the_first_section_that_changes_per_turn(self) -> None:
        manifest = _built().manifest
        prefix = [s.tokens for s in manifest.sections[:3]]
        assert manifest.prefix_tokens == sum(prefix)

    def test_equal_prefixes_sign_equally_and_a_changed_one_does_not(self) -> None:
        first = _built().manifest.prefix_signature
        changed = (
            PromptBuilder(FOUR).instructions(INSTRUCTIONS).system(POLICY, "Other policy.").build()
        )
        assert first == _built().manifest.prefix_signature
        assert first != changed.manifest.prefix_signature

    def test_no_stable_section_means_no_signature(self) -> None:
        manifest = PromptBuilder(FOUR).message(USER_QUERY, "user", "hi").build().manifest
        assert manifest.prefix_signature == ""


class TestATurnLayoutSectionsEveryStep:
    def test_history_question_and_tool_loop_are_told_apart(self) -> None:
        layout = TurnLayout(tuple(INSTRUCTIONS[:1]), history=2)
        messages = [
            Message(role="system", content="ignored: rebuilt from the layout"),
            Message(role="user", content="earlier question"),
            Message(role="assistant", content="earlier answer"),
            Message(role="user", content="now"),
            Message(role="assistant", tool_calls=[ModelToolCall(id="1", name="t", arguments="{}")]),
            Message(role="tool", content="result", tool_call_id="1"),
        ]
        built = layout.build(messages)
        names = [section.name for section in built.manifest.sections]
        assert names == [PROFILE, HISTORY, USER_QUERY, TOOL_LOOP]
        assert built.messages[0].content == "You are a support agent."
        assert built.messages[1:] == messages[1:]


class TestARequestNoBuilderMade:
    def test_is_broken_down_by_role(self) -> None:
        manifest = manifest_from_messages(
            [Message(role="system", content="be brief"), Message(role="user", content="hi")]
        )
        assert [s.name for s in manifest.sections] == ["instructions", "user_query"]


class TestTheContextCard:
    def test_the_tree_names_every_section_and_its_share(self) -> None:
        fields = context_fields(_built().manifest, "m")
        tree = fields[get_settings().observability.context_card.sections_title]
        assert tree[0].startswith("├─ AGENT.md profile · ")
        assert any("Working memory" in line for line in tree)
        assert tree[-1].startswith("└─ Tool schemas")

    def test_every_section_is_shown_in_full(self) -> None:
        fields = context_fields(_built().manifest, "m")
        policy = next(title for title in fields if title.startswith("Policy · "))
        assert fields[policy] == ["Refund within 30 days."]

    def test_it_is_drawn_once_and_before_the_request_card(self) -> None:
        reporter, shown = _feed()
        steps = StepEvents(reporter, max_steps=3, model="m")
        request = CompletionRequest(messages=_built().messages, manifest=_built().manifest)
        steps.llm_request(request, purpose="next_action")
        steps.context(request, purpose="next_action")
        assert [event.stage for event in shown] == [CONTEXT_STAGE, LLM_REQUEST]

    def test_the_response_sets_the_estimate_beside_the_providers_count(self) -> None:
        reporter, shown = _feed()
        steps = StepEvents(reporter, max_steps=3, model="m")
        request = CompletionRequest(messages=_built().messages, manifest=_built().manifest)
        steps.llm_request(request, purpose="next_action")
        result = SimpleNamespace(
            usage=TokenUsage(prompt_tokens=100, completion_tokens=1, total_tokens=101),
            latency_ms=1.0,
            finish_reason="stop",
            tool_calls=None,
            content="ok",
        )
        steps.llm_response(result, purpose="next_action")
        fields = shown[-1].fields
        assert fields["estimated_input_tokens"] == request.manifest.total_tokens
        assert fields["estimate_drift"].endswith("%")


class TestTheChainDrawsWhatNoStepDrew:
    async def test_a_bare_request_gets_a_card(self) -> None:
        reporter, shown = _feed()
        client = SteppedClient(ScriptedLLM(), StageReporterModelObserver(lambda: reporter), "judge")
        await client.complete(CompletionRequest(messages=[Message(role="user", content="hi")]))
        assert [event.stage for event in shown] == [CONTEXT_STAGE]

    async def test_a_request_a_step_already_drew_gets_no_second(self) -> None:
        reporter, shown = _feed()
        steps = StepEvents(reporter, max_steps=3, model="m")
        request = CompletionRequest(messages=[Message(role="user", content="hi")])
        steps.context(request, purpose="reply")
        client = SteppedClient(ScriptedLLM(), StageReporterModelObserver(lambda: reporter), "m")
        await client.complete(request)
        assert [event.stage for event in shown] == [CONTEXT_STAGE]


class TestTheTokenizer:
    def test_disabled_it_counts_by_characters_and_says_so(self) -> None:
        settings = TokenizerSettings(
            enabled=False, model="x", local_dir="nowhere", file="t.json", download_if_missing=False
        )
        tokenizer = LocalTokenizer(settings, CharEstimateTokenizer(4))
        assert tokenizer.count("abcdefgh") == 2
        assert tokenizer.exact is False and tokenizer.name == "chars/4"

    def test_a_missing_file_latches_to_the_estimate(self) -> None:
        settings = TokenizerSettings(
            enabled=True, model="x", local_dir="nowhere", file="t.json", download_if_missing=False
        )
        tokenizer = LocalTokenizer(settings, CharEstimateTokenizer(4))
        assert tokenizer.count("abcd") == 1
        assert tokenizer.exact is False


class TestCachedPromptTokens:
    def test_are_read_from_the_providers_usage(self) -> None:
        raw = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
            prompt_tokens_details=SimpleNamespace(cached_tokens=60),
        )
        assert OpenAICompatibleClient._usage_from(raw).cached_prompt_tokens == 60

    def test_are_priced_at_their_own_rate(self) -> None:
        provider = SimpleNamespace(
            input_cost_per_1k_tokens=1.0,
            cached_input_cost_per_1k_tokens=0.5,
            output_cost_per_1k_tokens=2.0,
        )
        client: Any = SimpleNamespace(_provider=provider)
        usage = TokenUsage(
            prompt_tokens=1000, completion_tokens=1000, total_tokens=2000, cached_prompt_tokens=500
        )
        assert OpenAICompatibleClient.cost_for(client, usage) == 0.5 + 0.25 + 2.0
