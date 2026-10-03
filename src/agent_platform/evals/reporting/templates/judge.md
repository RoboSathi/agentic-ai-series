# LLM judge · `${run_id}`

Judge model: `${judge_model}`. Each case gets one call that scores the
rubric criteria and checks every factual claim in the reply against the
evidence the agent was given. Scores are normalised to 0–1.

## Criteria

${criteria}

## Judge vs the runtime hallucination guard

${agreement}

## Unsupported claims

${claims}

## Lowest-scoring cases

${lowest}
