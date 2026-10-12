# guardrails — structured output and the response guardrail

Makes model output conform before use (validate, repair or reject, or constrain generation); `ResponseGuardrail` runs a reply through the hallucination check with a bounded repair loop that fails closed to a person.

- **Entry points:** `PydanticValidator`, `LlmConstrainedGenerator`, `ResponseGuardrail`, `GuardrailViolationError`
- **Configuration:** `guardrails:`
- **Design:** [LLD §A18](../../../design_docs/LLD.md#a18-guardrails)
