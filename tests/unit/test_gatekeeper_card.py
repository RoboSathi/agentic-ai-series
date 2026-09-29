"""The gatekeeper card says what was ruled and why, and nothing that only repeats a check.

`cumulative_total`, `threshold` and three ranks sat on the decision purely
to be printed; the checks already carry the numbers a reviewer needs
(`observed` on the authority and cumulative checks), and no business
logic read them.
"""

from types import SimpleNamespace

from agentic_commons.gatekeeper.types import AuthorityDecision, CheckReport
from customer_support_agent.presentation.reporting import gatekeeper_fields

_REMOVED = {
    "cumulative_total",
    "threshold",
    "authority_source",
    "authority_rank",
    "required_authority_rank",
    "max_granting_rank",
}


class TestTheDecisionCarriesOnlyWhatIsRead:
    def test_no_field_that_exists_only_to_be_printed(self) -> None:
        assert not _REMOVED & set(AuthorityDecision.model_fields)

    def test_the_card_does_not_print_them_either(self) -> None:
        decision = AuthorityDecision(
            verdict="deny",
            action="support.process_refund",
            actor="a",
            reason="over the rolling limit",
            reason_code="CUMULATIVE_LIMIT_EXCEEDED",
            question="cumulative",
            checks=[
                CheckReport(
                    check_id="gatekeeper.cumulative_limit",
                    name="Cumulative budget",
                    status="FAIL",
                    reason_code="CUMULATIVE_LIMIT_EXCEEDED",
                    reason="over",
                    observed={"projected": "$300.00", "threshold": "$250.00"},
                )
            ],
        )
        call = SimpleNamespace(side_effecting=True, params={"order_id": 10975, "amount": "23.79"})
        fields = gatekeeper_fields(decision, call)
        assert not _REMOVED & set(fields)
        assert fields["reason_code"] == "CUMULATIVE_LIMIT_EXCEEDED"
        assert fields["checks"][0]["observed"]["threshold"] == "$250.00"
