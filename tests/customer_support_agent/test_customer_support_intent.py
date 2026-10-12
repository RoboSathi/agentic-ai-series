from agent_platform.intent import IntentDisposition
from customer_support_agent.intent.router import CustomerSupportIntentRouter


def test_refund_request_phrases_are_deterministically_supported():
    router = CustomerSupportIntentRouter()

    cases = [
        "refund request",
        "request refund",
        "refund for order 10227",
        "request a refund for order 10227",
        "please refund order 10227",
    ]

    for message in cases:
        intent = router.classify(message)
        assert intent.name == "REQUEST_REFUND", message
        assert intent.disposition is IntentDisposition.SUPPORTED, message
        assert "process_refund" in intent.allowed_tools, message
