---
name: research
description: Run the multi-agent research pipeline (scout, retriever, auditor, adversarial reviewer, strategist, engineer, evaluator) on one research question. Use for question-level work, not for small tasks.
disable-model-invocation: true
argument-hint: [research question]
---

Run the research pipeline for this question: $ARGUMENTS

You are the orchestrator. You dispatch subagents, check their files, and stop at the human gates. Subagents do not see this conversation. Pass them the **file paths** and the stage, not summaries.

Cost warning: this pipeline starts many subagents. Tell the user the plan before stage 1. Do not use it for a question that you can answer by reading one file.

## Resume

The pipeline is long and the context fills up. After `/clear`, or in a new session, read `docs/research/<slug>/INDEX.md` and continue from the first stage that is not `done`. Never restart a stage that is `done` unless the user asks. Record in `INDEX.md` the stage, its status, the date, and the gate that was passed.

## Setup

1. Make a slug (short, lowercase, hyphens). Create `docs/research/<slug>/`.
2. Write `00_question.md` with: the question, the scope, the constraints (models, data, GPU time), the date window, and what the user expects. Create `INDEX.md` with one row per stage and its status.
3. Read `.claude/rules/research.md` and keep to it.

## Preflight (before stage 1)

1. Check that the subagent types exist (`/agents`). If one is missing, stop and tell the user to restart the session.
2. Check that the web tools work: run one search and open one page. `WebFetch` can be missing in some sessions. If it is missing, tell the user. The retriever and the auditor then need the texts downloaded by you.
3. Tell the user the plan and the expected cost: stages 2, 3, and 4 use the most tokens. Models: the scout and the retriever use `sonnet`; the others use the session model. Check `/usage` after stage 3.

## Stages

Run the stages in order. After each stage, check `git status`: only the expected files may change. Update `INDEX.md`.

| # | Subagent | Input | Output |
|---|---|---|---|
| 1 | `scout` | `00_question.md` | `01_scout.md` |
| 2 | `literature-retriever` | `01_scout.md` | `02_sources.csv`, `02_retrieval.md` |
| 3 | `evidence-auditor` | claims and `02_sources.csv` | `03_evidence.md` |
| 4 | `adversarial-reviewer` (mode A) | `03_evidence.md`, `01_scout.md`, the project claim | `04_adversarial.md` |
| 5 | `research-strategist` | `01` to `04`, the ADR | `05_strategy.md` |
| 5b | `adversarial-reviewer` (mode B) | `05_strategy.md` | `05b_redteam.md` |
| 6a | `experiment-engineer` | the approved ADR entry | code, tests, results, `06_manifest.md` |
| 6b | `experiment-evaluator` | ADR entry, `06_manifest.md`, results | `06_evaluation.md` |

Notes for each stage:

- **Stage 2.** If the topic is broad, start up to 3 retrievers in parallel, one per group of sub-questions. Give each its own output name (`02_sources_<group>.csv`). Merge them yourself and remove duplicates by arXiv id or DOI.
- **Stage 3.** Build the claim list from the retriever notes and from the project documents (`docs/paper/method.tex`, the ADR). Start one auditor per group of papers when the list is long. Give each claim and its source id.
- **Stage 3, high-risk claims.** For a claim that the strategy will depend on, start **two** independent auditors with the same input and compare. A disagreement goes to the human.
- **Stage 4.** Pass the project's own claim. Tell the reviewer to try to find prior art.
- **Stage 5b.** Run it on every strategy. If the verdict is `REVISE`, send the issues back to the strategist (new file `05_strategy_v2.md`; do not edit the old one). Allow at most 2 rounds.
- **Stage 6.** Never let the engineer grade the work. Start the evaluator as a new subagent with fresh context.

## Human gates (stop and ask the user)

1. After stage 1: confirm the scope and the queries.
2. After stage 5b: approve the experiments and the decision rules. The user may change them now, never after data.
3. Before any remote GPU run and before any long run.
4. Before any commit or push. Use `/record-experiment` after the evaluator finishes.

## Stop conditions

- The adversarial review finds prior art that closes the gap: stop. Report the paper and what is still different.
- The strategy has no experiment that can change a decision: stop and say so.
- The evaluator returns `INVALID`: fix the setup. Do not interpret the numbers.
- A subagent reports a failed check: do not continue until the user decides.

## Final report to the user

Give: the question, the gap in one sentence, the evidence counts per level and verdict, the main threats from the adversarial review, the experiments with their verdicts, and what the work does not show. Link the dossier files.
