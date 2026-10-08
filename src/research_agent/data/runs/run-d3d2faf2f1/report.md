# Memory Architecture of Agentic AI and the CoALA Framework

**Status:** COMPLETE

## Summary

The CoALA framework defines a language agent with modular memory components including working memory, long-term memory (episodic, semantic, procedural), and procedures as a higher-order layer [E2]. It also specifies three core components: Memory, Decision Processes, and Actions [E3], and implements memory stores such as a notepad for short-term, a knowledge base for facts, and an episode log for experiences, alongside a decision loop for reflective action selection [E4]. When trying to add every event into episodic memory, the system struggles with organizing the information coherently, and conceptualizing semantic memory can be problematic because it quickly accumulates vast amounts of data; basic clustering and segmentation based solely on episodic and semantic categories often prove insufficient, and as the volume and complexity of data grow, a need for even finer-grained organization emerges [E5]. Independent research identifies five distinct long-term memory types in agentic AI: short-term session context, semantic facts, episodic events, procedural behaviors, and cross-session persistence [E12], with structured knowledge stored separately from raw chat logs [E13] and the agent autonomously managing memory via tool calls [E15].

## Concepts

CoALA structures memory into working memory, long-term memory subdivided into episodic, semantic, and procedural memory, and a procedural layer for higher-order processes [E2][E6]. Long-term memory in agentic systems is categorized into episodic memory for recalling specific events with temporal and outcome details [E17], semantic memory for factual knowledge [E12], and procedural memory for learned behaviors [E16]. The agent itself controls memory persistence across sessions indefinitely [E14] and decides what to store, update, or discard through explicit tool calls [E15]. Structured knowledge such as facts, preferences, and learned behaviors is kept distinct from raw conversation logs [E13].

## Examples

CoALA exemplifies memory stores with a notepad for short-term context, a knowledge base for semantic facts, and an episode log for episodic experiences [E4]. In practice, an agent might use add_memory to record a user preference and later retrieve it across sessions without repetition [E15][E14]. Episodic memory allows AI agents to recall specific events and experiences from their operational history, storing what happened, when it happened, and what the outcomes were [E17].

## Known unknowns

- None declared during scoping.

## Coverage & Limits

- Every must-pass criterion was met.
- Budget: Q2: fetches exhausted.

## Citations

- [E2] https://www.cognee.ai/cognitive-architectures-for-language-agents-explained — retrieved 2026-10-07T09:05:01.543753+00:00 — page sha256 aaf785130effbc62… — “The paper proposes a framework called CoALA (Cognitive Architectures for Language Agents) designed to make language models function more like human problem-solvers. CoALA breaks down memory into several key components:

Working Memory: Acts as a short-term scratchpad that holds the immediate context—such as recent chat messages or partial solutions.

Long-Term Memory: Stores information over extended periods and is subdivided into:

Episodic Memory: Keeps records of past events (for example, "What happened the last time I tried solution X?").

Semantic Memory: Contains factual knowledge about the world (for instance, "Birds can fly, except for ostriches").

Procedural Memory: Remembers how to perform tasks, which might be embedded in the agent's code or the LLM's parameters.

Procedures: Higher-order functions, performing as a layer above storage, that can retrieve, manipulate, and apply stored information, enabling the system to perform complex, context-aware actions.”
- [E3] https://www.cognee.ai/cognitive-architectures-for-language-agents-explained — retrieved 2026-10-07T09:05:01.543753+00:00 — page sha256 aaf785130effbc62… — “In simple terms, the paper suggests that, like the human cognitive system, a robust AI system should have three core components:

Memory: A place to store both short-term and long-term information.

Decision Processes: Mechanisms for determining what to do next—essentially, a set of "if this, then that" rules.

Actions: The actual operations or tasks the system can perform.”
- [E4] https://www.cognee.ai/cognitive-architectures-for-language-agents-explained — retrieved 2026-10-07T09:05:01.543753+00:00 — page sha256 aaf785130effbc62… — “Cognitive architectures help address these issues by:

Adding Memory Stores, like a "notepad" for short-term tasks, a "knowledge base" for facts, and an "episode log" for experiences.

Implementing a Decision Loop so the LLM reflects on what actions to take next, rather than just generate text randomly.

Structuring Decision Processes to help the system decide whether to retrieve a memory, execute a piece of code, interact with a user, or even control a robot arm.”
- [E5] https://www.cognee.ai/cognitive-architectures-for-language-agents-explained — retrieved 2026-10-07T09:05:01.543753+00:00 — page sha256 aaf785130effbc62… — “Unfortunately, it tends to break down under certain conditions. For example, when trying to add every event into episodic memory, the system struggles with organizing the information coherently. Similarly, conceptualizing semantic memory can be problematic because it quickly accumulates vast amounts of data.

Basic clustering and segmentation based solely on episodic and semantic categories often prove insufficient. As the volume and complexity of data grow, a need for even finer-grained organization emerges. For instance, if you want to group all books about New York into a dedicated sub-graph, the current model doesn't offer clear guidance on how to achieve that.”
- [E12] https://www.patronus.ai/ai-agent-development/agentic-memory — retrieved 2026-10-07T09:05:29.042803+00:00 — page sha256 19295baa308b6da2… — “Five distinct memory types:

Short-term (session context)

Semantic (facts)

Episodic (past events)

Procedural (learned behaviors), and

Long-term (cross-session persistence)”
- [E13] https://www.patronus.ai/ai-agent-development/agentic-memory — retrieved 2026-10-07T09:05:29.042803+00:00 — page sha256 19295baa308b6da2… — “Structured knowledge extracted from enterprise knowledge stores and past conversations: facts, preferences, events, and learned behaviors, stored separately from raw chat logs.”
- [E15] https://www.patronus.ai/ai-agent-development/agentic-memory — retrieved 2026-10-07T09:05:29.042803+00:00 — page sha256 19295baa308b6da2… — “The agent itself decides what to store, update, or discard via tool calls such as add_memory or delete_memory.”
- [E6] https://arxiv.org/html/2309.02427v3 — retrieved 2026-10-07T09:05:01.931850+00:00 — page sha256 5811aaeba380512d… — “In this paper, we draw on the rich history of cognitive science and symbolic artificial intelligence to propose Cognitive Architectures for Language Agents (CoALA). CoALA describes a language agent with modular memory components, a structured action space to interact with internal memory and external environments, and a generalized decision-making process to choose actions.”
- [E17] https://machinelearningmastery.com/beyond-short-term-memory-the-3-types-of-long-term-memory-ai-agents-need — retrieved 2026-10-07T09:05:33.901094+00:00 — page sha256 316520bdf1725ad5… — “Episodic memory allows AI agents to recall specific events and experiences from their operational history. This stores what happened, when it happened, and what the outcomes were.”
- [E16] https://machinelearningmastery.com/beyond-short-term-memory-the-3-types-of-long-term-memory-ai-agents-need — retrieved 2026-10-07T09:05:33.901094+00:00 — page sha256 316520bdf1725ad5… — “AI agents need the same thing. Building agents that can learn from experience, accumulate knowledge, and execute complex tasks requires implementing three distinct types of long-term memory: episodic, semantic, and procedural.”
- [E14] https://www.patronus.ai/ai-agent-development/agentic-memory — retrieved 2026-10-07T09:05:29.042803+00:00 — page sha256 19295baa308b6da2… — “Persists across sessions indefinitely. The agent recalls information from days or weeks ago without the user repeating it.”
