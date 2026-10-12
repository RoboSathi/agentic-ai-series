# DeepEval report · `${run_id}`

**Gate: ${verdict}** — ${reasons}

The judge model (`${judge_model}`) grades each golden conversation, one
dimension per metric: ${metric_names}. These are soft — the section gates at
a ${soft_threshold} pass rate — and every rate carries its 95% Wilson
interval: on ${case_count} cases, a few points either way is noise. Whether
the right money moved is the oracle's question, in its own report.

${run_facts}

## What changed since the previous run

${changes}

## Metrics

${deepeval}

### By category (failing / checked)

${deepeval_matrix}

## Every case

${cases}

## Reliability across trials

${trials}

## Judge vs the runtime hallucination guard

${agreement}

## Cases below threshold in detail

${failures}

## Cost and budget

${cost}
