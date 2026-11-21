# pretty_duck

Research code for **domain-aware low-bit quantization of LLMs**, with interpretability tests that find which
components a domain's knowledge depends on. Branch: `feature/domain-compression`.
The project started from the Persona method (arXiv:2602.07164). No Persona code is used any more. The code is in `compress/`.

## Current status (read before you claim results)

- The circuit-protection gain (+3.2 OAB points) is **not established**. It comes from one rounding realization and
  letter accuracy, which is a noisy metric (equivalent quantizers differ by up to 15 points).
- Measured and stable: domain-contrast scores do not help allocation (negative), mean ablation is explained by
  perturbation dose, measured per-layer sensitivity beats the diagonal error `D`, and clipping the group range is the
  largest quantizer-side gain.
- The advisor review, the summary, and the work plan are in `docs/reviews/advisor-2026-11/`.
  The repository organization plan is in the same folder (`03_proposta_organizacao_repositorio.md`).
  Follow the plan order: reliable evaluation first (Phase 0), then the ceiling (Phase 1).
- The full record is in `docs/adr/0001-domain-aware-quantization-experiment-record.md` (E1 to E23, decisions D1 to D10).

## Layout

| Path | Content |
|---|---|
| `compress/` | Library: statistics, quantization formats, GPTQ, protection, circuits, patching, evaluation, data loaders. It moves to `src/compress/` in the planned code stage |
| `scripts/` | Command-line drivers for experiments. Keep them thin |
| `tests/` | Offline CPU tests. Tiny model and tokenizer in `tests/conftest.py` |
| `data/` | Small input data kept in git: `persona/`, `legal_facts/` (draft, not reviewed by a jurist), `controls/` |
| `docs/` | `paper/` (`method.tex`, `method.pdf`), `adr/` (decisions and experiment record), `reports/`, `reviews/`, `guides/`, `history/` (original REQ, TASK, ARCHITECTURE) |
| `.claude/` | Project rules, skills, subagent, hooks, and settings for Claude Code |
| `outputs/` | Results and caches. Ignored by git. Never commit or edit it by hand |

Start at `docs/index.md` to find a document.

## Commands

Run all commands from the repository root. Use the project environment: `.venv/bin/python`.

1. Run the tests: `.venv/bin/python -m pytest -q`
2. Run a script: `PYTHONPATH=. .venv/bin/python scripts/<name>.py --help`
3. Build the paper: `make paper`. Other targets: `make test`, `make lint`, `make sync-remote-dry`

## Remote GPU machine

- Host: `ssh maidai`. Project copy: `~/pretty_duck_gate`. Python: `~/pretty_duck/.venv/bin/python`.
- GPU 0 is an RTX 3090 (24 GB). GPU 1 is an RTX 3080 Ti (12 GB). Both are shared with other users.
- Sync code and data with the `sync-remote` skill (`make sync-remote-dry`, then `make sync-remote`).
  Never sync `outputs/`. Do not sync while a run is using the files.
- Start long runs with `nohup`, write a log and a `*_done` marker file, and make them resumable.
- Results come back with `scp`. The remote copy of `third_party/gptq_ref/` (the authors' GPTQ code) is not in git.
- If the host name does not resolve, tell the user. Do not guess another address.

## Tools for this project

- Skills: `/run-experiment` (plan, smoke test, run, record), `/record-experiment` (ADR, README, and paper with the same numbers), `/sync-remote`.
- Subagent `reviewer`: independent read-only review of a change. Use it before you finish a large change.
- Research pipeline `/research <question>` (see `docs/research/README.md`): `scout`, `literature-retriever`, `evidence-auditor`,
  `adversarial-reviewer`, `research-strategist`, `experiment-engineer`, `experiment-evaluator`. Files pass between stages, with human gates.
  Use it for question-level work only. The engineer never grades its own work.
- Hook: edits to `outputs/` and to `*.pt` files are blocked (`.claude/hooks/protect_paths.py`).
- `.claude/hooks/run_tests.py` is a `Stop` hook that is not enabled. See its docstring to enable it.
- Guide for working with Claude Code: `docs/guides/claude-code-workflow.md`.

## Project rules

1. Every number is measured, estimated, or simulated. All quantization here is simulated on dequantized weights.
   Do not claim a speedup.

2. Every intervention that changes weights or hooks uses a context manager and restores the weights exactly.

3. Use the same seeded questions in every condition. Store per-question results.

4. Judge damage with a continuous metric (KL to fp16, NLL on neutral text, held-out log-probability) and several
   realizations. Do not use one letter-accuracy number as proof.

5. Compare protected sets at equal weights and bytes. Use a same-size random control and a null distribution.

6. Do not tune on the evaluation data. Record hypotheses, regimes, and replicate counts before you run.

7. Record every result and decision in the ADR. Keep `README.md`, the ADR, and `docs/method.tex` consistent.

## Verification (do this before you say "done")

1. Run `.venv/bin/python -m pytest -q`. It must pass.
2. After a refactor, reproduce one saved number (a baseline in `outputs/*.json`) on the tiny fixtures or on the remote machine.
3. Report the command you ran and its output. "Tests pass" alone is not evidence.
4. Do not run a long experiment before the smoke test and the identity test pass.

## Experiment workflow (details: `.claude/rules/experiments.md`)

1. Write the ADR entry first: hypothesis, metric, regime, controls, number of realizations, decision rule.
2. Smoke test on the tiny fixtures. Then reproduce a saved number. Then run.
3. Use at least 5 realizations for any claim about a difference. Report the spread.
4. Read the per-question outputs. A result at chance level means check the setup first.
5. Put the analysis in `scripts/analysis/`. Save the result as JSON with a `config` block.
6. Update the ADR, `README.md`, and `docs/method.tex` with the same numbers.

## Workflow

Before you change code:

1. Read the relevant source, tests, and ADRs.
2. Read the matching rule in `.claude/rules/`. Rules for Python, experiments, and docs load when you open matching files. Plan first when the change touches several files.
3. Identify the backend dependencies and message contracts.

While you change code:

1. Make the smallest change that solves the problem.
2. Keep backward compatibility unless the user asks for a break.
3. Add or update tests and documentation.
4. Keep this file short. Add a line only for something you had to correct twice.

Before you finish:

1. Review the diff for unintended changes.
2. Run `python -m pytest -q`.
3. Report the test results and the important files you changed.
4. State what you could not verify.

Do not claim that a test, command, or integration works unless you verified it.
Do not commit or push unless the user asks.

## Writing Rules

Use Simplified Technical English (STE) at about 80% strictness of ASD-STE100.

- Use short, active sentences. Keep one main idea per sentence.
- Use simple, common words. Explain technical terms when they first appear.
- Use consistent names for technical concepts.
- Give instructions in the order of the steps. Use numbered lists for procedures.
- Be explicit. Remove vague or unnecessary wording.
