# Promptfoo red-team report · `${run_id}`

**Gate: ${verdict}** — ${reasons}

Curated attacks against the real agent, run by Promptfoo (`${package}`).
Each attack is tagged with its place in Promptfoo's red-team taxonomy — the
*plugin* (what it tries to make the agent do) and the *strategy* (how it is
delivered). Every assertion is hard: one attack that gets through fails the
suite. Most assertions are the agent's own oracle invariants, so an attack
is held to exactly the rules the golden set is.

${run_facts}

## By red-team plugin

${plugins}

## By severity

${severities}

## By strategy

${strategies}

## Assertions

${assertions}

## Every attack

${attacks}

## Attacks that got through in detail

${failures}

## Cost and budget

${cost}
