"""The research agent admits a message as tokens before anything holds it (ADR P-40).

The research agent keeps names (research subjects) but tokenises contact details, so an email
in the question reaches neither the model nor the PII card as itself. A scripted answerer;
no network.
"""

from tests.research_agent.conftest import BuildHarness

# The offline harness withholds `pii`; these tests are about it, so they declare it.
ANSWERED = {"direct_answer": True, "pii": True}
EMAIL = "someone@example.com"
QUESTION = f"what is the capital of France? write to {EMAIL}"


def _sent(h: object) -> str:
    requests = getattr(h, "roles").models["answerer"].requests
    return " ".join(m.content for request in requests for m in request.messages)


async def test_the_model_never_sees_the_email(harness: BuildHarness) -> None:
    h = harness({"answerer": [{"content": "Paris."}]}, ANSWERED)
    await h.agent.handle(QUESTION, "s")
    assert EMAIL not in _sent(h)
    [card] = [e for e in h.feed.events if e.stage == "pii_filtering" and e.status != "started"]
    assert "EMAIL_ADDRESS" in card.fields["categories"] and EMAIL not in str(card.fields)


async def test_a_turn_with_no_reporter_is_admitted_the_same(harness: BuildHarness) -> None:
    h = harness({"answerer": [{"content": "Paris."}]}, ANSWERED, reported=False)
    result = await h.agent.handle(QUESTION, "s")
    assert result.response
    assert EMAIL not in _sent(h)
