---
name: run-experiment
description: Plan, smoke-test, run, and record one experiment on the remote GPU machine. Use when the user asks to run a new experiment or condition.
disable-model-invocation: true
argument-hint: [experiment description]
---

Run this experiment: $ARGUMENTS

Follow these steps in order. Stop and ask the user if a step fails.

For a full study with literature review, use `/research` instead. This skill covers one experiment that is already specified.
For steps 2 to 6, you can use the `experiment-engineer` subagent. After it finishes, start `experiment-evaluator` with fresh context. The engineer must not grade its own work.

1. **Plan.** Write a `Proposed` entry in `docs/adr/`: hypothesis, metric, regime, controls, number of realizations (at least 5 for a difference claim), decision rule.
2. **Smoke test.** Run the script on the tiny fixtures or on SmolLM-135M on CPU. Run `make test`.
3. **Reference.** Reproduce one saved number from `outputs/`. Run the identity test (zero protected columns equals uniform).
4. **Check the machine.** `ssh maidai 'nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv; pgrep -af python'`. Do not start a run on a GPU that another run uses.
5. **Sync code.** Use the `sync-remote` skill.
6. **Start the run** with `nohup`, a log file, a `*_done` marker file, a seed, and resume support. Write the result as JSON with a `config` block.
7. **Read the outputs.** Look at per-question results and the predicted-letter histogram. A result at chance level means check the setup first.
8. **Record.** Use the `record-experiment` skill.

Do not tune on the evaluation data. Do not claim a result that you did not read from the saved file.
