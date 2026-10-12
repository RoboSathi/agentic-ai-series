# DeepEval report · `${run_id}`

**Gate: ${verdict}** — ${reasons}

The golden set, graded twice. The **oracle** reads the state each case left
behind and checks the agent's hard business rules — zero model calls; one
failure fails the run. **DeepEval** has the judge model (`${judge_model}`)
grade the conversation itself, one dimension per metric; those are soft and
gate at a ${soft_threshold} pass rate. Every rate carries its 95% Wilson
interval: on ${case_count} cases, a few points either way is noise, a broken
invariant is not.

${run_facts}

## What changed since the previous run

${changes}

## Oracle — hard business invariants (deterministic)

${oracle}

### By category (failing / checked)

${oracle_matrix}

## DeepEval — conversation quality (model-graded)

${deepeval}

### By category (failing / checked)

${deepeval_matrix}

## Reliability across trials

${trials}

## Judge vs the runtime hallucination guard

${agreement}

## Failing cases

${failures}

## Cost and budget

${cost}
