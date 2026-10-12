# Building a Good Agentic AI Harness

**Status:** COMPLETE

## Findings

An agent harness is the runtime scaffolding that turns a language model into an agent by driving model and tool calls, managing conversation state and context, applying approval policies, and keeping the agent progressing through multistep tasks [E4]. The harness composes building blocks including a chat client, chat pipeline, agent and context providers, middleware, decorators, and application UX [E5]. Default capabilities encompass function invocation with configurable iteration limits, per-service-call history persistence, token compaction, todo tracking, plan-and-execute modes, session memory, tool approval with standing approvals and auto-approval rules, OpenTelemetry observability, and optional background agents and shell execution [E6].

## Details

For production agentic applications, the agent's knowledge base should be managed in external persistent storage such as Memory Bank to ensure durability, scalability, and accessibility across multiple instances [E3]. To accelerate agent development in Google Cloud, use ADK [E2].

## Known unknowns

- None declared during scoping.

## Coverage & Limits

- Every must-pass criterion was met.

## Citations

- [E4] https://learn.microsoft.com/en-us/agent-framework/concepts/harness — retrieved 2026-10-03T18:15:12.182817+00:00 — page sha256 df1f9fe4908b482a… — “An agent harness, sometimes called an AI harness, is the runtime scaffolding that turns a language model into an agent that can perform work. It drives model and tool calls, manages conversation state and context, applies approval policies, and can keep the agent progressing through a multistep task.”
- [E5] https://learn.microsoft.com/en-us/agent-framework/concepts/harness — retrieved 2026-10-03T18:15:12.182817+00:00 — page sha256 df1f9fe4908b482a… — “The Harness composes existing Agent Framework building blocks rather than defining a separate agent runtime:

Chat client — connects the agent to a model.

Chat pipeline — adds function invocation, message injection, per-service-call history persistence, and optional compaction.

Agent and context providers — add session-scoped instructions, tools, memory, todo state, operating modes, and optional capabilities.

Middleware and decorators — add approval handling, observability, and optional bounded looping.

Application UX — streams responses, displays progress, and collects input such as tool approvals.”
- [E6] https://learn.microsoft.com/en-us/agent-framework/concepts/harness — retrieved 2026-10-03T18:15:12.182817+00:00 — page sha256 df1f9fe4908b482a… — “Harness capability matrix

Capability
Harness behavior
Canonical guidance

Function invocation
Enabled with a configurable per-request iteration limit.
Function tools

Per-service-call history persistence
Persists history after each model call in a tool-calling run.
Sessions

Compaction
Enabled when token limits or a custom strategy are supplied.
Compaction

Todo tracking
Enabled by default.
Planning and todos

Agent modes
Plan and execute modes are enabled by default.
Planning and todos

File memory and file access
Session file memory is enabled by default; shared file access is opt-in.
Context providers

Tool approval
Standing approvals and auto-approval rules are enabled by default.
Tool approval

OpenTelemetry
Agent observability is enabled by default.
Observability

Web search
Added by default where the selected chat client supports it.
Web search

Agent Skills
Enabled by default in .NET; opt-in through a provider or paths in Python.
Agent Skills

Background agents
Optional parallel delegation to named child agents.
Background agents

Shell execution
Composed from the shell package; the Python factory can wire it automatically.
Shell tools

Looping
Optional bounded re-invocation driven by evaluators or predicates.
Agent looping”
- [E3] https://docs.cloud.google.com/architecture/choose-agentic-ai-architecture-components — retrieved 2026-10-03T18:15:12.820287+00:00 — page sha256 3e90f0e759ebc7d0… — “External storage: For production applications, manage your agent's
knowledge base in an external, persistent storage service. An external
storage service ensures that your agent's knowledge is durable, scalable,
and accessible across multiple application instances. Use
Memory Bank
for long-term storage with any agent runtime on Google Cloud.”
- [E2] https://docs.cloud.google.com/architecture/choose-agentic-ai-architecture-components — retrieved 2026-10-03T18:15:12.820287+00:00 — page sha256 3e90f0e759ebc7d0… — “To accelerate agent development in Google Cloud, we recommend that you
use
ADK.”
