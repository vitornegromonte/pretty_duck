# pretty_duck

Research code for **domain-aware low-bit quantization of LLMs**, with interpretability tests that find which
components a domain's knowledge depends on. The library is in `src/compress/` (subpackages `quant`, `interp`, `evaluation`, `data`, `experiments`). Experiment drivers are in `scripts/`.
Start at `docs/index.md` to find any document.

## Read before you claim results

- The circuit-protection gain is **not established**. Letter accuracy of one quantized model is a noisy metric:
  the same code gave 40.7% and 49.5% on two GPUs. Judge damage by continuous metrics over several realizations.
- The record of experiments and decisions is the ADR in `docs/adr/`. The advisor review and the work plan are in
  `docs/reviews/advisor-2026-11/`. Follow the plan order: a reliable evaluation first, then the ceiling.
- Take every number from a saved file in `outputs/`. Do not type numbers from memory.

## Commands

Run all commands from the repository root with `.venv/bin/python`.

1. Tests: `make test` (offline, CPU). Style: `make lint`. Paper: `make paper`.
2. A script: `PYTHONPATH=src .venv/bin/python scripts/<name>.py --help`.
3. Remote files: `make sync-remote-dry`, then `make sync-remote`.

## Remote GPU machine

- Host `ssh maidai`, project copy `~/pretty_duck_gate`, Python `~/pretty_duck/.venv/bin/python`.
- GPU 0 is an RTX 3090 (24 GB). GPU 1 is an RTX 3080 Ti (12 GB). Both are shared with other users. Check them before a run.
- Never sync `outputs/`. Do not sync while a run uses the files. Start long runs with `nohup`, a log, and a `*_done` marker.
- The authors' GPTQ code (`third_party/gptq_ref/`) exists only on the remote machine.
- If the host name does not resolve, tell the user. Do not guess another address.

## Gotchas

- `outputs/` is ignored by git and holds the only copy of some results. Never edit it by hand (a hook blocks it).
- `/tmp` is a small tmpfs (1.9 GB). If commands fail with "disk full", ask the user to free it.
- The remote machine still has the old flat `compress/` folder. Remove it before the first sync of `src/`, or the old code can shadow the new package. Do not do this while a run uses it.
- Result files from before the refactor use old keys (`oab_acc`, `law_logp`). New files use `target_acc`, `control_acc`, `fact_logp`. The old code is at the tag `v0-pre-refactor`.

## Tools

- Skills: `/run-experiment`, `/record-experiment`, `/sync-remote`. Subagent `reviewer`: read-only review of a change.
- `/research <question>`: the multi-agent literature and experiment pipeline (`docs/research/README.md`).
  Use it for question-level work only. The engineer never grades its own work.
- Rules in `.claude/rules/` load by path: `python.md`, `experiments.md`, `docs.md`, `research.md`.
- Guide: `docs/guides/claude-code-workflow.md`.

## Workflow

Before you change code:

1. Read the relevant source, tests, and ADRs.
2. Read the matching rule in `.claude/rules/`. Plan first when the change touches several files.

While you change code:

1. Make the smallest change that solves the problem. Do not mix a refactor and a behavior change in one commit.
2. Keep backward compatibility unless the user asks for a break. Old result files must stay readable.
3. Add or update tests and documentation.
4. Keep this file short. Add a line only for something you had to correct twice.

Before you finish:

1. Review the diff for unintended changes.
2. Run `make test`. Report the command, its output, and the important files you changed.
3. After a refactor, reproduce one saved number from `outputs/`.
4. State what you could not verify.

Do not claim that a test, command, or integration works unless you verified it.
Do not commit or push unless the user asks. Write commit messages with a short imperative subject and a body with the reason and the evidence.

## Writing Rules

Use Simplified Technical English (STE) at about 80% strictness of ASD-STE100.

- Use short, active sentences. Keep one main idea per sentence.
- Use simple, common words. Explain technical terms when they first appear.
- Use consistent names for technical concepts.
- Give instructions in the order of the steps. Use numbered lists for procedures.
- Be explicit. Remove vague or unnecessary wording.
