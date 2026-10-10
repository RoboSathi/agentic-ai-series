"""What made a correct research answer read as invented: each cause, pinned."""

import pytest

from agent_platform.config import get_settings
from agent_platform.config.settings import RoleModelSettings
from agent_platform.evals.entailment import LexicalEntailmentModel, negation_pattern
from agent_platform.hallucination_check.claims import ClaimDecomposer, specifics
from agent_platform.hallucination_check.direct import amounts
from agent_platform.llm.roles import RoleModelRouter
from agent_platform.llm.types import CompletionRequest, Message
from agent_platform.llm.typography import TypographyNormaliser
from tests.conftest import ScriptedLLM


class TestNegation:
    def test_a_negation_word_inside_another_word_is_not_one(self) -> None:
        pattern = negation_pattern({"no", "not", "n't"})
        assert not pattern.search("openai announced new technology for the economy")
        assert pattern.search("there is no rain") and pattern.search("it isn't raining")

    @pytest.mark.asyncio
    async def test_an_exact_restatement_is_not_read_as_its_opposite(self) -> None:
        result = await LexicalEntailmentModel().entails(
            "Now: wind speed 5.3km/h", "light wind at 5.3 km/h now"
        )
        assert result.label != "contradiction"


class TestSpecifics:
    def test_an_iso_timestamp_sources_the_day_and_year_a_sentence_names(self) -> None:
        claim = specifics("As of 09:00 IST on 2 October 2026, it is 27.1 °C")
        evidence = specifics("as of 2026-10-02T09:00 temperature 27.1°C")
        assert claim <= evidence

    @pytest.mark.parametrize("text", ["saves $1.2 M a year", "saves $1.2M", "$1.2 million"])
    def test_a_scaled_figure_is_not_an_exact_amount(self, text: str) -> None:
        assert amounts(text) == set()

    def test_an_exact_amount_still_is(self) -> None:
        assert {str(a) for a in amounts("a refund of $18.70.")} == {"$18.70"}


def test_a_models_citation_markup_is_stripped_where_text_enters() -> None:
    normaliser = TypographyNormaliser(get_settings().llm.output_normalisation)
    assert normaliser.normalise("70 linked accounts 【0†L10-L20】.")[0] == "70 linked accounts."


def test_the_connectives_that_split_a_claim_are_configuration() -> None:
    settings = get_settings().model_copy(deep=True)
    sentence = "OpenAI shipped models and agents because demand grew."
    default = [c.text for c in ClaimDecomposer().decompose(sentence)]
    assert len(default) == 3  # the platform splits on "and" and on "because"
    settings.hallucination_check.claim_conjunctions = ["because"]
    from agent_platform.config import settings as module

    original = module._config_instance
    module._config_instance = settings
    try:
        narrow = [c.text for c in ClaimDecomposer().decompose(sentence)]
    finally:
        module._config_instance = original
    assert narrow[0] == "OpenAI shipped models and agents"


@pytest.mark.asyncio
async def test_a_roles_request_fields_ride_on_every_request() -> None:
    llm = ScriptedLLM([{"content": "ok"}])
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    router = RoleModelRouter(
        {
            "interviewer": RoleModelSettings(provider="nim", model="m", extra_body=extra),
            "answerer": RoleModelSettings(provider="nim", model="m"),
        },
        factory=lambda provider, model: llm,
    )
    request = CompletionRequest(messages=[Message(role="user", content="q")])
    await router.client_for("interviewer").complete(request)
    await router.client_for("answerer").complete(request)
    # One adapter for the model; only the interviewer's requests carry its fields.
    assert llm.requests[0].extra_body == extra and llm.requests[1].extra_body == {}
    assert router.model_key("interviewer") == router.model_key("answerer")
