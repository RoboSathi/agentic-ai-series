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
    PROFILE,
    SKILLS,
    TOOL_LOOP,
    TURN_CONTEXT,
    USER_QUERY,
    PromptBuilder,
    PromptPart,
    TurnLayout,
    Volatility,
    manifest_from_messages,
)
from agent_platform.llm.tokenizer import CharEstimateTokenizer, LocalTokenizer
from agent_platform.llm.types import (
    CompletionRequest,
    CompletionResult,
    Message,
    ModelToolCall,
    TokenUsage,
)
from agent_platform.observability import InMemoryLiveStatusFeed, StageReporter
from agent_platform.observability.context_card import CONTEXT_STAGE, context_fields
from agent_platform.observability.llm_fields import cache_fields
from agent_platform.observability.model_observer import StageReporterModelObserver
from agent_platform.observability.step_events import LLM_REQUEST, StepEvents
from agent_platform.observability.turn_tally import TurnTally
from agent_platform.observability.types import StatusEvent
from tests.conftest import ScriptedLLM

FOUR = CharEstimateTokenizer(4)


def _raw_usage(details: object) -> SimpleNamespace:
    """A provider's usage block, with or without its prompt_tokens_details."""
    return SimpleNamespace(
        prompt_tokens=100, completion_tokens=10, total_tokens=110, prompt_tokens_details=details
    )


INSTRUCTIONS = [
    PromptPart(PROFILE, "You are a support agent.", "AGENT.md"),
    PromptPart(SKILLS, "# Skill: refund\nsteps", "refund@1"),
    PromptPart(SKILLS, "# Skill: lookup\nsteps", "lookup@1"),
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
        assert names == [PROFILE, SKILLS, POLICY, CONTEXT_WINDOW, USER_QUERY]

    def test_the_system_message_is_every_system_section_joined(self) -> None:
        messages = _built().messages
        assert [m.role for m in messages] == ["system", "user"]
        assert messages[0].content.startswith("You are a support agent.\n\n# Skill: refund")
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

    def test_each_skill_is_its_own_part(self) -> None:
        skills = _built().manifest.sections[1]
        assert [part.source for part in skills.parts] == ["refund@1", "lookup@1"]

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
        # The tool schemas count too: a chat template renders them at the head of the prompt.
        assert manifest.prefix_tokens == sum(prefix) + manifest.tool_schema_tokens

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


class TestTheLayoutIsOrderedForThePrefixCache:
    """Least volatile first, whatever order an agent adds its sections in."""

    def test_a_fixed_section_goes_ahead_of_a_per_turn_one_added_before_it(self) -> None:
        parts = [
            PromptPart(PROFILE, "Profile.", "AGENT.md"),
            PromptPart(SKILLS, "# Skill: refund", "refund@1", Volatility.TURN),
            PromptPart(POLICY, "Refund within 30 days."),
        ]
        built = PromptBuilder(FOUR).instructions(parts).build()
        names = [section.name for section in built.manifest.sections]
        assert names == [PROFILE, POLICY, SKILLS]
        assert built.manifest.sections[1].stable and not built.manifest.sections[2].stable

    def test_two_skill_choices_share_one_prefix(self) -> None:
        def signed(skill: str) -> str:
            parts = [
                PromptPart(PROFILE, "Profile."),
                PromptPart(SKILLS, skill, volatility=Volatility.TURN),
                PromptPart(POLICY, "Policy."),
            ]
            return (
                PromptBuilder(FOUR)
                .instructions(parts)
                .tools(TOOLS)
                .build()
                .manifest.prefix_signature
            )

        assert signed("# Skill: refund") == signed("# Skill: lookup")

    def test_the_tools_offered_are_part_of_the_prefix(self) -> None:
        other = [{"type": "function", "function": {"name": "get_order", "parameters": {}}}]
        builder = PromptBuilder(FOUR).system(POLICY, "Policy.")
        assert builder.tools(TOOLS).build().manifest.prefix_signature != (
            builder.tools(other).build().manifest.prefix_signature
        )

    def test_with_history_the_turns_context_follows_it(self) -> None:
        history = [Message(role="user", content="before"), Message(role="assistant", content="ok")]
        built = (
            PromptBuilder(FOUR)
            .system(PROFILE, "Profile.")
            .system("today", "Today is Thursday.", volatility=Volatility.TURN)
            .messages(HISTORY, history)
            .message(USER_QUERY, "user", "now")
            .build()
        )
        settings = get_settings().context.prompt.turn_context
        assert [m.content for m in built.messages[:3]] == ["Profile.", "before", "ok"]
        assert built.messages[3].role == settings.role
        assert built.messages[3].content == f"{settings.heading}\n\nToday is Thursday."
        assert built.messages[4].content == "now"
        names = [section.name for section in built.manifest.sections]
        assert names == [PROFILE, HISTORY, TURN_CONTEXT, "today", USER_QUERY]

    def test_without_history_the_turns_context_closes_the_system_message(self) -> None:
        built = (
            PromptBuilder(FOUR)
            .system("today", "Today is Thursday.", volatility=Volatility.TURN)
            .system(PROFILE, "Profile.")
            .message(USER_QUERY, "user", "now")
            .build()
        )
        assert built.messages[0].content == "Profile.\n\nToday is Thursday."
        assert len(built.messages) == 2


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
        usage = OpenAICompatibleClient._usage_from(_raw_usage(SimpleNamespace(cached_tokens=60)))
        assert usage.cached_prompt_tokens == 60
        assert usage.cache_reported and usage.cache_hit

    def test_a_reported_zero_is_a_miss(self) -> None:
        usage = OpenAICompatibleClient._usage_from(_raw_usage(SimpleNamespace(cached_tokens=0)))
        assert usage.cache_reported and not usage.cache_hit

    def test_a_provider_that_does_not_say_is_not_reported(self) -> None:
        usage = OpenAICompatibleClient._usage_from(_raw_usage(None))
        assert not usage.cache_reported and not usage.cache_hit

    def test_cache_hit_is_the_last_thing_a_response_card_says(self) -> None:
        result = CompletionResult(
            content="hi",
            usage=TokenUsage(prompt_tokens=10, cached_prompt_tokens=8, cache_reported=True),
        )
        feed, shown = _feed()
        StepEvents(feed, max_steps=1).llm_response(result, purpose="reply")
        fields = list(shown[-1].fields.items())
        assert fields[-2:] == [("cached_input_tokens", 8), ("cache_hit", True)]

    def test_an_unreported_count_says_so(self) -> None:
        fields = cache_fields(TokenUsage(prompt_tokens=10))
        not_reported = get_settings().observability.cache_not_reported
        assert fields == {"cached_input_tokens": not_reported, "cache_hit": False}

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


class TestTheTurnSummarySaysHowMuchWasCached:
    def test_it_adds_up_every_call_of_the_turn(self) -> None:
        tally = TurnTally()
        tally.model_call(TokenUsage(prompt_tokens=1000, total_tokens=1010))
        tally.model_call(
            TokenUsage(
                prompt_tokens=1200, total_tokens=1210, cached_prompt_tokens=900, cache_reported=True
            )
        )
        line = tally.fields("completed", 1.0)["prompt_cache"]
        expected = get_settings().observability.prompt_cache_line.format(
            hits=1, calls=2, cached=900, prompt=2200
        )
        assert line == expected


class TestTheLayoutsEdges:
    def test_a_window_of_only_stable_chunks_is_all_prefix(self) -> None:
        stable = [Chunk(content="policy", kind="policy", source="p", metadata={"stable": True})]
        manifest = PromptBuilder(FOUR).window("# Context", "policy", stable).build().manifest
        assert [s.name for s in manifest.sections] == ["window_stable"]
        assert manifest.sections[0].stable

    def test_history_with_nothing_after_it_still_gets_the_turn_context(self) -> None:
        built = (
            PromptBuilder(FOUR)
            .system(PROFILE, "Profile.")
            .system("today", "Thursday.", volatility=Volatility.TURN)
            .messages(HISTORY, [Message(role="user", content="before")])
            .build()
        )
        assert [s.name for s in built.manifest.sections][-2:] == [TURN_CONTEXT, "today"]
        assert built.messages[-1].content.endswith("Thursday.")
