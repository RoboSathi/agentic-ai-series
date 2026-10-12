# Oracle report · `${run_id}`

**Gate: ${verdict}** — ${reasons}

The oracle reads the state each golden case left behind — the refund, review
and order rows in the case's own database — and checks the agent's business
rules against it. Zero model calls, so the same transcripts always get the
same verdict. **Hard** checks gate at 100%: one broken invariant fails the
run. **Soft** checks (diagnostics, such as whether the reply named the
reason) gate at a ${soft_threshold} pass rate. Every rate carries its 95%
Wilson interval.

${run_facts}

## What changed since the previous run

${changes}

## Rules

${rules}

### By category (failing / checked)

${matrix}

## Every case

${cases}

## Reliability across trials

${trials}

## Failing cases in detail

${failures}

## Cost and budget

${cost}
