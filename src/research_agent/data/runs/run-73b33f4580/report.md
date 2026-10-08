# Reflection Design Pattern in Agentic AI

**Status:** PARTIAL

## Summary

The reflection design pattern automates self-criticism so an agent can iteratively improve its own outputs [E1]. It enables AI systems to examine their outputs, reasoning, and decisions to identify errors and areas for improvement [E5]. These patterns implement self-awareness and self-correction capabilities, allowing systems to iteratively refine their responses, validate their reasoning, and adapt their approach based on self-assessment [E6].

## Concepts

Core mechanics involve a single agent generating output, evaluating it against explicit criteria, identifying weaknesses, and producing a revised attempt in a self-improvement loop [E9]. The agent checks its output against a success criteria tool (tests or rubric) that returns pass/fail signals [E10], and a router node acts as a conditional gate to either route back for revision or forward to final output [E11]. The pattern automates critical feedback so the model automatically criticizes its own output and improves its response [E1], and enables systems to examine their own outputs and reasoning to identify errors [E5]. Techniques such as Self-Critique involve the agent evaluating its own outputs through systematic analysis and generating targeted improvements [E7]. A router node called 'Satisfactory?' acts as a conditional gate: if self-assessment is not satisfactory it routes back for revision with critique; if satisfactory it routes to final output [E11].

## Examples

Reflection can be extended by giving the LLM tools such as unit tests to check code correctness or web search to double-check text output, then reflecting on any errors found [E4]. Available techniques include Producer-Critic Pattern where separate producer and critic agents collaborate, LLM as Judge where an LLM acts as critic, and Reflexion where agents learn from experience through verbal self-reflection and episodic memory [E7]. Best practices include defining clear criteria and metrics, implementing multiple reflection cycles, balancing depth with efficiency, using diverse evaluation perspectives, maintaining logs, designing stopping criteria, and integrating human feedback [E8].

## Known unknowns

- None declared during scoping.

## Coverage & Limits

- R4: The sub-concept 'core mechanics' is explained and cross-checked in two sources. — 0 independent source(s), scored 0.00 of the 1 required; run stopped: maximum rounds reached; the critique found the answer incomplete: The claims cover the three sub-concepts (core mechanics, prompt templates, evaluation loops) with specific details like prompt structures, the 'Satisfactory?' router, and tool-augmented reflection. However, the cross-check requirement for core mechanics and prompt templates is not met: R4 and R5 show only 0 and 1 independent sources respectively, while R6 (evaluation loops) passes with 3 sources. Without at least two independent sources for core mechanics and prompt templates, the evidence base is too thin to support a reliable explainer.. Tried: reflection design pattern agentic AI core mechanics prompt templates evaluation loops, self-reflection LLM agent architecture prompt structure evaluation loop.
- R5: The sub-concept 'prompt templates' is explained and cross-checked in two sources. — 0 independent source(s), scored 0.00 of the 1 required; run stopped: maximum rounds reached; the critique found the answer incomplete: The claims cover the three sub-concepts (core mechanics, prompt templates, evaluation loops) with specific details like prompt structures, the 'Satisfactory?' router, and tool-augmented reflection. However, the cross-check requirement for core mechanics and prompt templates is not met: R4 and R5 show only 0 and 1 independent sources respectively, while R6 (evaluation loops) passes with 3 sources. Without at least two independent sources for core mechanics and prompt templates, the evidence base is too thin to support a reliable explainer.. Tried: reflection design pattern agentic AI core mechanics prompt templates evaluation loops, self-reflection LLM agent architecture prompt structure evaluation loop.
- 2 source(s) rejected — blocked or paywalled. e.g. https://medium.com/@zhaoyi0113/agentic-ai-reflection-design-patterns-5e46291ef6eb, https://medium.com/@vpatil_80538/reflective-loop-pattern-the-llm-powered-self-improving-ai-architecture-7b41b7eacf69

## Citations

- [E1] https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection — retrieved 2026-10-04T17:36:04.179075+00:00 — page sha256 8768380b0795d076… — “What if you automate the step of delivering critical feedback, so the model automatically criticizes its own output and improves its response? This is the crux of Reflection.”
- [E5] https://agentic-design.ai/patterns/reflection — retrieved 2026-10-04T17:36:06.568379+00:00 — page sha256 28e6b2cb54760d15… — “Reflection patterns enable AI systems to examine their own outputs, reasoning processes, and decision-making to identify errors, inconsistencies, or areas for improvement.”
- [E6] https://agentic-design.ai/patterns/reflection — retrieved 2026-10-04T17:36:06.568379+00:00 — page sha256 28e6b2cb54760d15… — “These patterns implement self-awareness and self-correction capabilities, allowing systems to iteratively refine their responses, validate their reasoning, and adapt their approach based on self-assessment.”
- [E9] https://agenticdiagrams.com/patterns/reflection-pattern/ — retrieved 2026-10-04T17:36:06.858131+00:00 — page sha256 391bac2592d034c0… — “Single-agent self-improvement pattern. After producing output the agent evaluates it against its own success criteria, identifies weaknesses, and generates a revised attempt. Based on the Reflexion paper (Shinn et al., 2023).”
- [E10] https://agenticdiagrams.com/patterns/reflection-pattern/ — retrieved 2026-10-04T17:36:06.858131+00:00 — page sha256 391bac2592d034c0… — “criteria:
    type: tool
    sub_type: function
    label: Success Criteria
    sub_title: tests / rubric
    edges:
      - to: agent
        type: return
        label: pass / fail signals”
- [E11] https://agenticdiagrams.com/patterns/reflection-pattern/ — retrieved 2026-10-04T17:36:06.858131+00:00 — page sha256 391bac2592d034c0… — “gate:
    type: router
    sub_type: conditional
    label: Satisfactory?
    edges:
      - to: agent
        label: no — revise with critique
      - to: final
        label: yes — ship”
- [E7] https://agentic-design.ai/patterns/reflection — retrieved 2026-10-04T17:36:06.568379+00:00 — page sha256 28e6b2cb54760d15… — “Self-Critique(SC)
medium
Agent evaluates its own outputs through systematic analysis and generates targeted improvements

Producer-Critic Pattern(PC)
medium
Architectural pattern where separate producer and critic agents collaborate for quality improvement

LLM as Judge(LJ)
low
Specific Producer-Critic implementation where an LLM acts as the critic to evaluate outputs

Reflexion(RX)
high
Agents learn from experience through verbal self-reflection and episodic memory”
- [E4] https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection — retrieved 2026-10-04T17:36:04.179075+00:00 — page sha256 8768380b0795d076… — “And we can go beyond self-reflection by giving the LLM tools that help evaluate its output; for example, running its code through a few unit tests to check whether it generates correct results on test cases or searching the web to double-check text output. Then it can reflect on any errors it found and come up with ideas for improvement.”
- [E8] https://agentic-design.ai/patterns/reflection — retrieved 2026-10-04T17:36:06.568379+00:00 — page sha256 28e6b2cb54760d15… — “Best Practices
Define clear criteria and metrics for self-evaluation
Implement multiple reflection cycles for complex tasks
Balance reflection depth with computational efficiency
Use diverse evaluation perspectives to avoid blind spots
Maintain logs of reflection processes for analysis and improvement
Design stopping criteria to prevent infinite reflection loops
Integrate human feedback to calibrate reflection effectiveness”
