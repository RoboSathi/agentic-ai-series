# security — proving who the customer is

`CustomerAuthenticator` verifies a customer by their seven-character user code before anything acts on their behalf; identity never comes from anything the model says.

- **Entry points:** `CustomerAuthenticator`, `AuthResult` (`authentication.py`)
- **Configuration:** `customer_support_agent.user_code_length`, `user_code_alphabet`
- **Design:** [ADR C-04](../../../design_docs/ADR.md#c-04--authentication-is-the-hard-front-door)
