# Design docs

Every design document for the repository — the platform and all three agents — lives in this
folder. Agent packages hold no design documents; each package's `README.md` is a short pointer
to the section here that describes it.

## Read in this order

| Document | What it answers |
|---|---|
| [PRD.md](PRD.md) | Why the system exists, who it is for, every requirement and its build status, what is parked |
| [HLD.md](HLD.md) | What it is made of: the platform's component map, how an agent is composed, the three agents' graphs, request lifecycles, deployment, stack |
| [LLD.md](LLD.md) | How each component works: interfaces, configuration, seams — platform packages, then each agent, then the cross-subsystem invariants |
| [ADR.md](ADR.md) | Why it is this way: the single decision log (consolidates platform ADRs 0001–0013 and research ADRs ra-0001–0024) |
| [ubiquitous_language.md](ubiquitous_language.md) | What the words mean: one glossary for the platform, analytics and every agent |
| [patterns.md](patterns.md) | Where each agent uses each WF/AG and LLD pattern the series teaches |
| [threat_model.md](threat_model.md) | Assets, trust boundaries, threats and the control that answers each |
| [evals.md](evals.md) | How each agent is evaluated: three graders, the shared judge, each suite |
| [analytics_schema.md](analytics_schema.md) | The analytics contract column by column: facts, the metric catalogue, agent-owned tables |
| [Program_Design.md](Program_Design.md) | Repository layout, the design process, coding standards, verification |

## Conventions

- **The code and these documents agree.** A change that moves the design updates the relevant
  section here in the same change, and adds or amends an entry in [ADR.md](ADR.md).
- **Decisions** are written once, in `ADR.md`, as the current decision; superseded versions are
  listed in its appendix rather than kept as separate files.
- **Dependencies point from agent to platform, never back.** The platform's documents name no
  agent; agent-specific design is in the agent sections of each document.
- **Diagrams** are Mermaid, so they render on GitHub and diff as text.
- **Screenshots** for the root README live in [`images/`](images/).
