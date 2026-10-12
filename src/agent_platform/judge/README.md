# judge — the LLM judge every grader builds on

One skeleton for asking a model to decide about someone else's output: validated answers, shared grading rules, and abstention instead of a silent pass or fail.

- **Entry points:** `create_judge_client`, `LlmJudge` (Template Method), `ModelJudge`, `ask_validated`, `ask_text`, `JudgeCalibrator`
- **Configuration:** `judge:`
- **Design:** [LLD §A5](../../../design_docs/LLD.md#a5-judge)
