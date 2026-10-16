"""The per-session facts the agent carries between turns, and their bound."""

from customer_support_agent.domain.identity import Customer
from customer_support_agent.graph.conversation import ConversationState, SessionMap


def _customer(number: int) -> Customer:
    return Customer(customer_id=number, name=f"c{number}", email=f"c{number}@example.com")


def test_past_the_bound_the_least_recently_written_session_goes() -> None:
    sessions: SessionMap[int] = SessionMap(2)
    sessions["a"], sessions["b"] = 1, 2
    sessions["a"] = 3
    sessions["c"] = 4
    assert list(sessions) == ["a", "c"]


def test_a_rebuilt_agent_adopts_every_session_fact() -> None:
    previous, current = ConversationState(max_sessions=5), ConversationState(max_sessions=5)
    previous.sessions["s"] = _customer(1)
    current.adopt(previous)
    assert current.sessions["s"].customer_id == 1


def test_adopting_respects_the_bound() -> None:
    previous, current = ConversationState(max_sessions=5), ConversationState(max_sessions=2)
    for number in range(4):
        previous.sessions[f"s{number}"] = _customer(number)
    current.adopt(previous)
    assert list(current.sessions) == ["s2", "s3"]
