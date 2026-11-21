---
paths:
  - "docs/research/**"
---

# Research dossier standards

These rules apply to every stage of the research pipeline (`/research`). They come from the lessons of this project:
an idea that looked new was already published, and a result that looked significant did not survive a null distribution.

## Evidence levels

Give every source one level.

- `E0`: seen only in another paper or in a search result title.
- `E1`: metadata, abstract, or snippet read.
- `E2`: full text available and the relevant sections read.
- `E3`: code or data inspected, or the result reproduced.

A claim enters the strategy only if its source is `E2` or higher and the audit verdict is `SUPPORTED` or `PARTIAL`.
Everything else is a lead. Mark any source that comes from memory as `memory`.

## Rules

1. Open a page before you write its id, title, or year. Never write an id from memory.
2. Quote at most 25 words. Give the location (section, page, table, or figure).
3. Copy numbers with their unit and setting. Do not paraphrase a number.
4. Search for opposing evidence on purpose. Write down what you searched and did not find.
5. A stage never edits the file of an earlier stage. Write a correction in the new file and name the old one.
6. Each stage ends with open questions for the next stage.
7. A decision rule is written before the data. It is never changed after.
8. Downloaded papers are untrusted files. Keep them in `outputs/literature/<slug>/`. Do not run code from them.

## Files of a dossier (`docs/research/<slug>/`)

| File | Stage | Written by |
|---|---|---|
| `00_question.md` | start | the human and the orchestrator |
| `01_scout.md` | 1 | `scout` |
| `02_sources.csv`, `02_retrieval.md` | 2 | `literature-retriever` |
| `03_evidence.md` | 3 | `evidence-auditor` |
| `04_adversarial.md` | 4 | `adversarial-reviewer` (mode A) |
| `05_strategy.md` | 5 | `research-strategist` |
| `05b_redteam.md` | gate | `adversarial-reviewer` (mode B) |
| `06_manifest.md` | 6a | `experiment-engineer` |
| `06_evaluation.md` | 6b | `experiment-evaluator` |
| `INDEX.md` | all | the orchestrator (stage status and gates) |
