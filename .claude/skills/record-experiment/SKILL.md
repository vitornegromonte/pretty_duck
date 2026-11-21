---
name: record-experiment
description: Record an experiment result in the ADR, README, and paper with the same numbers. Use after a run finishes or when a decision changes.
argument-hint: [experiment id or result file]
---

Record this result: $ARGUMENTS

1. Read the saved JSON. Take every number from the file. Do not type numbers from memory.
2. Update `docs/adr/` (status `Accepted` or `Rejected`, numbers, caveats, artifacts) with the setup, the result, and a verdict.
3. Update the "Findings so far" section of `README.md`.
4. Update `docs/paper/method.tex` (tables, summary, discussion). Build it with `make paper` and check for errors.
5. Check that the three documents show the same numbers and the same verdict.
6. State what the result does not show. A hypothesis is not a result.
7. Do not commit unless the user asks.
