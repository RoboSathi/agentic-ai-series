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

This module decides *who*; it never says anything. The words a customer
reads — the prompt for a code, the verified read-back — are rendered by
the graph's authentication node from `presentation`, so security does not
depend on how a greeting looks.
"""

import re

from pydantic import BaseModel

from agent_platform.logging import get_logger, instrument, logged
from customer_support_agent.config import get_agent_settings
from customer_support_agent.domain.identity import Customer
from customer_support_agent.domain.models import User
from customer_support_agent.persistence.repositories import UsersRepository


class AuthResult(BaseModel):
    """The outcome of an authentication attempt."""

    authenticated: bool
    customer: Customer | None = None
    # The account row behind `customer`, for the verified read-back.
    user: User | None = None
    code_attempted: bool = False


@instrument("customer_support_agent")
class CustomerAuthenticator:
    """Turns a typed code into an authenticated customer, or a prompt."""

    def __init__(self, users: UsersRepository) -> None:
        settings = get_agent_settings()
        self._users = users
        self._length = settings.user_code_length
        self._alphabet = settings.user_code_alphabet
        self._placeholder = settings.user_code_placeholder
        # A code is a run of exactly the configured length drawn from the
        # configured alphabet, containing at least one digit. Anchored on
        # word boundaries so it is found inside a sentence ("my code is
        # HED8TVA") without matching a longer token that contains one.
        #
        # The digit requirement is what stops a seven-letter English word
        # being read as a credential: without it "what's the weather"
        # authenticates as WEATHER and the customer is told their code was
        # not found, which is both wrong and baffling.
        alphabet = re.escape(self._alphabet)
        self._pattern = re.compile(
            rf"\b(?=[{alphabet}]{{{self._length}}}\b)"
            rf"[{alphabet}]*\d[{alphabet}]*\b",
            re.IGNORECASE,
        )
        self._logger = get_logger("customer_support_agent")

    def customer_for(self, customer_id: int) -> Customer | None:
        """The customer a checkpointed run was for, read back from the record.

        Used only to resume a run this customer already authenticated
        for; it is not a way in, because nothing a customer types can
        name a checkpoint.
        """
        user = self._users.get(customer_id)
        if user is None:
            return None
        return Customer(customer_id=user.user_id, name=user.name, email=user.email)

    def extract_code(self, message: str) -> str | None:
        """The candidate code in a message, if one is present."""
        match = self._pattern.search(message.upper())
        return match.group(0) if match else None

    def mask_code(self, message: str) -> str:
        """The message with any typed code replaced, for everything that keeps it.

        A code is a credential. It is read once, to authenticate; after
        that the history, the episodic record and the next prompt need
        only know that one was given.
        """
        return self._pattern.sub(self._placeholder, message)

    @logged("customer_support_agent")
    def authenticate(self, message: str) -> AuthResult:
        """Try to authenticate from whatever the customer just typed."""
        code = self.extract_code(message)
        if code is None:
            return AuthResult(authenticated=False, code_attempted=False)
        user = self._users.get_by_code(code)
        if user is None:
            # One outcome whether the code is malformed or simply wrong:
            # distinguishing them tells an attacker which codes exist.
            self._logger.warning("Authentication failed", code_length=len(code))
            return AuthResult(authenticated=False, code_attempted=True)
        self._logger.info("Customer authenticated", customer_id=user.user_id)
        return AuthResult(
            authenticated=True,
            customer=Customer(customer_id=user.user_id, name=user.name, email=user.email),
            user=user,
            code_attempted=True,
        )
