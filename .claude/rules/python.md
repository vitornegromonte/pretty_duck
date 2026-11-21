---
paths:
  - "compress/**/*.py"
  - "src/**/*.py"
  - "scripts/**/*.py"
  - "experiments/**/*.py"
  - "tests/**/*.py"
---

# Python rules

Style, imports, and common mistakes (mutable defaults, bare `except`, wildcard imports) are enforced by `ruff`
(see `[tool.ruff]` in `pyproject.toml`). Run `make lint`. Do not rewrite an old file only to satisfy the linter.
Fix it when you change it for another reason. The rules below are what the linter cannot check.

## Design

- Prefer functions and small dataclasses. Use a class when an object must hold or restore state, for example a context
  manager that changes model weights and puts them back (`MixedQuant`, `GPTQQuant`). Do not build class hierarchies for experiments.
- Make dependencies explicit: pass them as arguments. Isolate side effects.
- Put logic in `compress/`. Keep `scripts/` thin: parse arguments, build the experiment, run it, save the result.
  Write results to files, not to stdout. Scripts may write progress to stderr with one `log` helper.
- Use `logging`, not `print()`, in library code under `compress/`.
- Use a dataclass or a typed structure for a data contract such as a result row.

## Interfaces

- Add type hints to public functions. Write units and shapes in names or docstrings (`out_features`, `n_questions`, `bits`).
- Write a docstring for every public function: what it returns, the shapes of tensors, and what it changes.
- Use specific exception types. Name the bad value in the message.

## Testing

- Run `make test` before you finish. Tests run offline and on CPU. Use the tiny fixtures in `tests/conftest.py`. Do not download models.
- Add a test for every new behavior and every bug you fix. Write the failing test first when you can.
- Test properties, not only examples: weights are restored bit for bit (`tests/test_restore_and_identity.py`),
  protecting zero columns equals uniform, a quantized value lies on the grid, patching all units reproduces the clean run.
- Compare with an independent reference when one exists (for example the authors' GPTQ code).
- Do not weaken a test to make it pass. If an expected value was wrong, say so in the commit message.
- Mark tests that need a GPU with `pytest.mark.skipif` and state why.

## Resources

- Assume the GPU is shared and small. A 4B model in fp16 uses about 8.4 GB. Keep saved weights on the host (`offload=True`) when memory is short.
- Do not keep activations or full-vocabulary logits you do not need. Use last-position logits.
- Measure time and peak memory before you optimize. Report them when you claim a cost.
- Batch work only when the result does not change. Prove it with a test.
