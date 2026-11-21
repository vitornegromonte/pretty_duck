# Research dossiers

Each folder here is one research question, worked through the multi-agent pipeline. The pipeline is the skill `/research`
(`.claude/skills/research/SKILL.md`). The agents are in `.claude/agents/`. The evidence rules are in `.claude/rules/research.md`.

## The pipeline

```text
00 question ──► 1 scout ──► 2 literature-retriever ──► 3 evidence-auditor ──► 4 adversarial-reviewer (A)
   (human)        │                  │                        │                        │
   gate 1 ◄───────┘                  └── sources (E0..E3) ────┴── claims audited ─────┘
                                                                                      ▼
                      6b experiment-evaluator ◄── 6a experiment-engineer ◄── gate 2 ◄── 5b adversarial-reviewer (B) ◄── 5 research-strategist
                              │                                            (human)
                         verdict ──► /record-experiment ──► gate 4 (commit)
```

| Stage | Agent | Question it answers | Writes |
|---|---|---|---|
| 1 | `scout` | Which terms, methods, fields, and opposing queries cover the question? | `01_scout.md` |
| 2 | `literature-retriever` | Which papers exist, in which versions, and at which evidence level? | `02_sources.csv`, `02_retrieval.md` |
| 3 | `evidence-auditor` | Does each paper really support the claim we attribute to it? | `03_evidence.md` |
| 4 | `adversarial-reviewer` (A) | What contradicts us, what is missing, and is it already done? | `04_adversarial.md` |
| 5 | `research-strategist` | What is the gap and which experiments tell the explanations apart? | `05_strategy.md` |
| 5b | `adversarial-reviewer` (B) | Can this plan fail in an informative way? | `05b_redteam.md` |
| 6a | `experiment-engineer` | Is the experiment built and run exactly as specified? | code, tests, results, `06_manifest.md` |
| 6b | `experiment-evaluator` | Does the evidence meet the decision rule written before the data? | `06_evaluation.md` |

## Why these roles are separate

- **The builder does not grade its work** (6a and 6b). The evaluator has fresh context, recomputes the metrics from the raw outputs, and cannot edit code.
- **The auditor and the adversarial reviewer start without the project history** (`omitClaudeMd`). Their judgment must not depend on our narrative.
- **The adversarial reviewer runs twice**: on the evidence (A) and on the plan (B), before any GPU time is spent.
- **Each stage writes a file and passes the path.** Subagents do not see the conversation, and files make the work reviewable.

## Human gates

1. After the scout: confirm the scope and the queries.
2. After the red-team gate: approve experiments and decision rules.
3. Before any remote GPU or long run.
4. Before any commit or push.

## When to use it

Use the pipeline for a question that needs literature and a new experiment, for example "does X give a gain over Y?".
Do not use it for a question that one file answers, or for a small code change. It starts many subagents and costs many tokens.
Check the cost with `/usage`.

## Lessons built into the pipeline

| Lesson from this project | Where the pipeline handles it |
|---|---|
| An idea that looked new was already published (TaCQ, OWQ) | Stage 4 searches for prior art on purpose; stop condition if it closes the gap |
| Papers cited from memory or from snippets | Evidence levels E0 to E3; a claim needs E2 and an audit |
| A significant gain did not survive a null distribution | Stage 5 needs same-size controls, a noise floor, and at least 5 realizations; gate 5b checks them |
| Letter accuracy hid rounding noise | A continuous primary metric is required; the evaluator reads the per-question outputs |
| Decisions were adjusted after seeing data | The decision rule is written in stage 5 and the evaluator may not move it |

## Naming

`docs/research/<slug>/` with a short lowercase slug. Downloaded papers go in `outputs/literature/<slug>/` (ignored by git).
