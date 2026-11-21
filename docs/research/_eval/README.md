# Gold set for the research agents

The agents are the judges of the pipeline, so they need their own check. This folder holds a small gold set
(`gold_claims.csv`) with claims whose verdicts we know. Half of the claims are planted errors: a wrong title, a wrong year,
a source that does not exist, and a result that the repository files contradict.

## How to run

1. Make a dossier folder, for example `docs/research/_eval/run-01/`, and copy `gold_claims.csv` into it without the
   `expected_verdict`, `ground_truth_check`, and `planted` columns.
2. Start the `evidence-auditor` subagent on the claims and the sources (the paper claims need `02_sources.csv`
   rows made by the `literature-retriever`, or the arXiv ids).
3. Compare the verdicts with `expected_verdict`.
4. Run it twice with fresh contexts. Count the claims where the two runs disagree.

## Pass criteria (our choice, change them only with a note)

- No planted-false claim receives `SUPPORTED` (zero false accepts).
- At least 9 of 10 verdicts match.
- The two runs agree on at least 9 of 10 claims.
- Every `SUPPORTED` verdict has a location and a quote.

## Maintenance

Add a claim every time an agent misses a real error (this is the main source of new tasks). Record the date and the
check you used. Remove a claim if its source changes. Do not run the agents on the gold set and then adjust the prompts
until it passes without adding new claims: that makes the set useless.

Status: the gold set was written on 2026-11-20 and has **not been run yet**.
