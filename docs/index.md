# Documentation index

Find a document by the question you have.

| Question | Where |
|---|---|
| What is the project and how do I install it? | `README.md` |
| What did we test and what did we decide? | `docs/adr/0001-domain-aware-quantization-experiment-record.md` (experiments E1 to E23, decisions D1 to D10) |
| What is the method and what are the results? | `docs/paper/method.pdf` (source: `method.tex`; build with `make paper`) |
| What did the advisor say and what is the plan? | `docs/reviews/advisor-2026-11/` (`01_sintese_didatica.md`, `02_plano_de_acao.md`, `03_proposta_organizacao_repositorio.md`, `referencias.csv`; the original review is in `raw/`) |
| What is the report for the advisor? | `docs/reports/relatorio_orientador.pdf` (it does not yet include the 4B results E15 to E23) |
| How do I run a literature-and-experiment study with agents? | `docs/research/README.md` (pipeline, roles, gates) and the dossiers in `docs/research/<slug>/` |
| How do I work with Claude Code in this project? | `CLAUDE.md`, `docs/guides/claude-code-workflow.md`, `.claude/rules/` |
| Which data does the code read? | `data/` (`persona/`, `legal_facts/` is an unreviewed draft, `controls/`) |
| What was the original task? | `docs/history/` (`REQ.md`, `TASK.md`, `ARCHITECTURE.md`) |

## Rules for this folder

- Keep the ADR, `README.md`, and `docs/paper/method.tex` consistent. Use the same numbers.
- Put inputs that code reads in `data/`, not in `docs/`.
- Do not commit LaTeX build files. `.gitignore` already excludes them.
