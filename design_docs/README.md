# Design docs — platform-wide

This folder holds only what is true of the **platform** (`src/agent_platform`)
and of the monolith as a whole. Each agent's own design lives inside its
package:

| agent | design docs |
|---|---|
| `customer_support_agent` (csa) | [`src/customer_support_agent/design_docs/`](../src/customer_support_agent/design_docs/README.md) |
| `research_agent` (ra) | [`src/research_agent/design_docs/`](../src/research_agent/design_docs/overview.md) |

## Index

| document | what it is |
|---|---|
| [`adr/0001-pattern-ownership.md`](adr/0001-pattern-ownership.md) | Pattern *composition* is agent-owned; only domain-free mechanisms go in commons |
| [`adr/0002-agent-registry-and-make.md`](adr/0002-agent-registry-and-make.md) | `make <target> <agent>`: the Makefile agent registry |
| [`adr/0003-commons-mechanisms-domain-free.md`](adr/0003-commons-mechanisms-domain-free.md) | The mechanisms added for a second agent, and the vocabulary guard |
| [`adr/0004-host-hooks-panels-cancel.md`](adr/0004-host-hooks-panels-cancel.md) | Optional `AgentHost` hooks: greeting, side panels, cancel, forwarded turns |
| [`adr/0005-baseline-repair.md`](adr/0005-baseline-repair.md) | The behaviour-preserving repair that made `make verify` green before ra work began |
| [`adr/0006-retire-episodes-and-pattern-library.md`](adr/0006-retire-episodes-and-pattern-library.md) | `episodes/` and `design_patterns/` removed; episodes run the agents under config |
| `PRD.md`, `HLD.md`, `LLD.md`, `Program_Design.md`, `Episodes.md` | The series' original product/design documents (see below) |

## The earlier PRD / HLD / LLD / Program Design

The build brief for the research agent assumed these documents were not in
this repository. They are: `PRD.md`, `HLD.md`, `LLD.md` and
`Program_Design.md` sit in this folder and predate the research agent.
They describe the platform and the customer-support agent only, and they
were **not edited** as part of adding ra. Where they and the code disagree
(for example PRD G1, "every tool call passes the Gatekeeper", which
`CLAUDE.md` records as retired now that the Gatekeeper is a toggleable
feature), the code and the ADRs are current. **A human must reconcile
them** — decide what to keep, what to supersede with ADRs, and where ra
belongs in them.

## Conventions

- **ADRs** use Context / Decision / Consequences / Alternatives rejected.
  Platform ADRs are numbered `0001…` here; an agent's are `ra-0001…` (or
  the agent's prefix) in its own `design_docs/adr/`.
- **Dependencies point from agent to platform, never back.** Commons holds
  no agent vocabulary (enforced for ra's words by
  `tests/agent_platform/test_roles_and_boundary.py`).
- **Agent docs describe the code as it exists.** csa's folder records only
  what can be verified from its code; nothing is invented after the fact.
- **Each agent's `design_docs/patterns.md`** records where it uses each
  pattern the series teaches; edit it when the code moves.
