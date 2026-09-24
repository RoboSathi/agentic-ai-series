"""Proving who the customer is, before anything acts on their behalf."""

from customer_support_agent.security.authentication import AuthResult, CustomerAuthenticator

__all__ = ["AuthResult", "CustomerAuthenticator"]
