"""Authenticating a customer by their code (LLD §21).

The first thing a conversation establishes is who is speaking. Until a
valid `user_code` arrives, the agent will not look anything up — not
because a tool would leak (lookups are already owner-scoped) but because
without an identity there is no owner to scope to, and "show me my
orders" has no answer.

Nothing here involves a model. The code either matches a row or it does
not, and a deterministic check is faster, cheaper, and cannot be talked
into a different answer. The details read back are a fixed format for the
same reason: a model asked to render a customer's own record is a model
that can get it subtly wrong.
"""

import re

from pydantic import BaseModel

from agentic_commons.logging import get_logger, logged
from customer_support_agent.config import get_agent_settings
from customer_support_agent.core.identity import Customer
from customer_support_agent.core.repositories import UsersRepository
from customer_support_agent.core.schema import User


class AuthResult(BaseModel):
    """The outcome of an authentication attempt."""

    authenticated: bool
    customer: Customer | None = None
    message: str = ""
    code_attempted: bool = False


class CustomerAuthenticator:
    """Turns a typed code into an authenticated customer, or a prompt."""

    def __init__(self, users: UsersRepository) -> None:
        settings = get_agent_settings()
        self._users = users
        self._length = settings.user_code_length
        self._alphabet = settings.user_code_alphabet
        # A code is a run of exactly the configured length drawn from the
        # configured alphabet. Anchored on word boundaries so it is found
        # inside a sentence ("my code is HEDZTVA") without matching a
        # longer token that merely contains one.
        self._pattern = re.compile(
            rf"\b[{re.escape(self._alphabet)}]{{{self._length}}}\b", re.IGNORECASE
        )
        self._logger = get_logger("customer_support_agent")

    def extract_code(self, message: str) -> str | None:
        """The candidate code in a message, if one is present."""
        match = self._pattern.search(message.upper())
        return match.group(0) if match else None

    @logged("customer_support_agent")
    def authenticate(self, message: str) -> AuthResult:
        """Try to authenticate from whatever the customer just typed."""
        code = self.extract_code(message)
        if code is None:
            return AuthResult(
                authenticated=False,
                message=self._ask_for_code(),
                code_attempted=False,
            )
        user = self._users.get_by_code(code)
        if user is None:
            # The same message whether the code is malformed or simply
            # wrong: distinguishing them tells an attacker which codes
            # exist.
            self._logger.warning("Authentication failed", code_length=len(code))
            return AuthResult(
                authenticated=False,
                message=(
                    "I couldn't find an account with that code. Please check it and "
                    "try again — it's the "
                    f"{self._length}-character code from your order confirmation email."
                ),
                code_attempted=True,
            )
        self._logger.info("Customer authenticated", customer_id=user.user_id)
        return AuthResult(
            authenticated=True,
            customer=Customer(customer_id=user.user_id, name=user.name, email=user.email),
            message=self.describe(user),
            code_attempted=True,
        )

    def describe(self, user: User) -> str:
        """The fixed-format read-back of a customer's own details.

        Rendered here rather than by a model. These are the customer's
        own records; a generated version can be subtly wrong, and being
        subtly wrong about someone's account is worse than being terse.
        The email is masked because a support transcript is not a place
        to restate a full address.
        """
        order_count = self._users.order_count_for(user.user_id)
        return (
            f"Thanks, {user.name} — you're verified.\n"
            f"  Account   : #{user.user_id}\n"
            f"  Email     : {user.masked_email}\n"
            f"  Customer since: {user.created_at[:10]}\n"
            f"  Orders    : {order_count}\n"
            "How can I help with your orders today?"
        )

    def _ask_for_code(self) -> str:
        """The prompt shown until a code arrives."""
        return (
            "Before I can look anything up, I need to verify your account. "
            f"Please share your {self._length}-character customer code — it's in "
            "your order confirmation email."
        )
