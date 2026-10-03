# verification — did the side effect happen?

After a side-effecting call, reads back what should have changed and
compares it with what the action claimed. A result that cannot be verified
never becomes authoritative evidence for a customer-facing claim.

## Vocabulary

- **IntendedEffect** — what an action said it did
- **EffectSource** — an agent's way of reading the real state back

## Public interface

- `EffectVerifier` (Protocol), `ReadBackVerifier` (`verifier.py`)
- `EffectSource` (Protocol), `IntendedEffect`, `VerificationReport` (`types.py`)

## Configuration

No section of its own.

## Seams

- An agent supplies its `EffectSource`.

## Collaborators

- Feeds the evidence tiers of `hallucination_protection`.

## Episodes

Effect verification (LLD §24).

## Tests

Covered through the agent's refund tests; dedicated unit tests are an open item.
