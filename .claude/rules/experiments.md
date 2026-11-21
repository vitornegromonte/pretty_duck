---
paths:
  - "scripts/**"
  - "experiments/**"
  - "docs/adr/**"
  - "docs/experiments/**"
  - "data/**"
---

# Experiment standards

These rules protect the scientific claims of the project. They come from the project needs and from Anthropic guidance on evals.

## Reproducibility and experiments **(project)**

1. **Seeds.** Every random choice takes a seed from an argument. The default is 42. Record the seed in the output file.
2. **Weights must be restored.** Any intervention that changes model weights or adds hooks uses a context manager.
   The weights must be bit-identical after exit, also after an exception. Add every new manager to `tests/test_restore_and_identity.py`.
3. **Forward-only.** Do not add gradients, fine-tuning, or auxiliary networks to the main method without a decision in an ADR.
4. **Label what is measured.** Mark each number as measured, estimated, or simulated. All quantization here is
   simulated on dequantized weights. Do not claim a speedup.
5. **Equal budgets.** Compare protected sets of the same size in weights and bytes. Count bytes with the exact format
   formulas (`plan_nbytes`), never with nominal bits. When you protect scattered columns with RTN, use
   `MixedQuant(keep_blocks=True)` so the 64-column block boundaries do not move. GPTQ still forms new blocks over the
   remaining columns (known limit). Protecting zero columns must equal uniform (tested).
6. **Controls.** A comparison needs a control of the same size and class, a random set drawn many times, and a
   null distribution over sets. A confidence interval over questions does not measure the uncertainty of a selection rule.
7. **Metrics.** Do not use letter accuracy of one quantized model as the only damage metric. Report a continuous metric
   (KL to the fp16 model, NLL on neutral text, log-probability of held-out facts) and the number of realizations.
   Use several realizations (calibration draws or block shifts) for each condition.
8. **Same data in every condition.** Use the same seeded questions, in the same order, in every condition. Store
   per-question results, not only totals, so you can make paired comparisons.
9. **Held-out data.** Calibration data and evaluation data must not overlap. Choose circuits on one half of the facts
   and validate them on the other half.
10. **Do not tune on the test set.** Fix the hypothesis, the regime, and the number of replicates before you run. Record
    the choice in the ADR.
11. **Negative results count.** Record them in `docs/adr/` with numbers and caveats.
12. **Determinism.** Accumulate Hessians and statistics in float64 or float32 with a fixed order. Do not rely on TF32.
    Store `compress.provenance.provenance()` in each result file (commit, dirty flag, software, GPUs, dtype).

## Data, results, and files

- Save results as JSON with a `config` block (all arguments, seed, model, git commit, GPU, dtype) and the data. Do not
  rename existing keys. Add new keys instead, so old files stay readable.
- Write results after every condition, and make long runs resumable (skip conditions that are already done).
- Do not commit `outputs/`, checkpoints (`*.pt`), caches, or secrets. Large results live in `outputs/` and on the remote machine.
- Do not depend on code that is outside this repository. If you need a small piece of external code, copy it with a comment that names the source and the license, and add a test.
- Do not commit data that has no license. Say in the file when a data set is a draft or not reviewed.
- Do not write secrets, tokens, or personal data into files, logs, or commit messages.
- Do not start long writes to `/tmp`. It is a small tmpfs on the local machine. Use the scratchpad or `outputs/`.

## Designing and running experiments and evals **(project)**

Source: Anthropic Engineering, "Demystifying evals for AI agents" (https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents).
That post gives no number of trials and no variance method. The numbers below are our own and come from the advisor review.

**Design**

1. **Define success before you run.** Write the hypothesis, the metric, the regime, the decision rule, and the number of
   realizations in the ADR with status `Proposed`. Change them only with a note that says why.
2. **Make each task unambiguous.** Two experts must reach the same pass or fail verdict. Every check must follow from the
   written task. Fix an ambiguous task before you spend GPU time.
3. **Prove that the task can be solved.** Keep a reference run with a known answer (for example the fp16 baseline, the
   identity test "zero protected columns equals uniform", and the authors' GPTQ code as a cross-check).
4. **Start small, then scale.** Run a smoke test on a tiny model on CPU. Then run a small real subset. A large effect needs
   few samples. A small effect needs many samples and realizations.
5. **Build balanced sets.** Include cases where the behavior must happen and cases where it must not (law and non-law,
   protected set and same-size control).
6. **Start from real failures.** Turn each bug or surprising result into a test or a new condition.

**Running**

7. **Run several trials.** Results vary between runs. Report the mean and the spread over R realizations (calibration draws,
   block shifts, random sets). Our rule: R is at least 5 for any claim about a difference *(ours)*. Use "pass at k" when one success is
   enough. Use "pass to the power k" when every trial must succeed.
8. **Keep each trial clean and isolated.** Restore the weights, draw the calibration data again, and share no state between
   trials. Failures that come from one shared cause (out-of-memory, a stale cache) are not independent trials.
9. **Grade the outcome, not the path.** Use metrics that need a real solution and that a shortcut cannot satisfy. Prefer
   continuous scores and partial credit to one 0 or 1 number.
10. **Prefer code graders.** They are fast and reproducible. If you use a model as a judge, calibrate it against human labels,
    give it an "unknown" option, and score each dimension separately.
11. **Read the outputs.** Look at per-question predictions, the histogram of predicted letters, and failed items. A result at
    chance level or at zero usually means a broken setup. Check the setup before you conclude anything.
12. **Watch the headroom.** If the baseline is near its ceiling, or the damage is near zero, an effect cannot be seen. Report the headroom next to each result.

**Maintenance**

13. **Keep two kinds of evals.** A *capability* eval starts where the system fails and gives a hill to climb. A *regression* eval
    should stay near 100% and must not break: saved baseline numbers, identity tests, and reproduction of a stored result.
    Run the regression eval before and after every refactor.
14. **Treat evals as living code.** Version them, review them, and fix unfair or ambiguous grading. A change to an eval gets a
    commit message that states what changed and why.
15. **Layer the checks.** No single check catches every failure. Combine automated tests, manual reading of outputs, and an
    outside review (the advisor).

## Experiment lifecycle checklist **(ours)**

1. **Plan.** Write the ADR entry (`Proposed`): hypothesis, metric, regime, controls, realizations, decision rule.
2. **Smoke.** Run on the tiny fixtures or on SmolLM-135M. The test suite stays green.
3. **Reference.** Reproduce one saved number. Run the identity test.
4. **Run.** Use a seed, a `config` block, per-question outputs, resume support, a log, and a `*_done` marker.
5. **Analyze in code.** Put analysis in `scripts/analysis/` and read the saved JSON. Do not keep analysis only in scratch files or in a chat.
6. **Record.** Update the ADR (`Accepted` or `Rejected`, with numbers and caveats), `README.md`, and `docs/method.tex`.
7. **Commit.** One commit for the code and one for the record. Do not push unless asked.
