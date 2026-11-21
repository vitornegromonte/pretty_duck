---
paths:
  - "compress/**/*.py"
  - "src/**/*.py"
  - "scripts/**/*.py"
  - "experiments/**/*.py"
  - "tests/**/*.py"
---

# Python code standards

All new code and all changes to existing code must follow these standards.
Priorities, in order: correctness, reproducibility, readability, maintainability, simplicity.

Rules marked **(project)** come from the needs of this repository (low-bit quantization experiments on LLMs).
Where an old file breaks a rule, do not rewrite it only to comply. Fix it when you change it for another reason.

## General design principles

Follow these principles:

- SOLID: apply object-oriented principles where they improve maintainability.
- DRY: avoid unnecessary duplication. Move code to a shared module when a third copy appears.
- KISS: prefer simple solutions over complex ones.
- YAGNI: do not implement features that are not required now.
- Composition over inheritance: combine small, focused components.
- Separation of concerns: keep unrelated responsibilities apart.
- Dependency inversion: depend on stable abstractions, not on concrete implementations.

Additional requirements:

- Prefer readable code over clever code.
- Make dependencies explicit. Pass them as arguments.
- Keep side effects isolated.
- Avoid hidden behavior and implicit dependencies.
- Avoid premature optimization. Measure first.
- Avoid unnecessary abstractions and frameworks.

**(project)** This code base prefers functions and small dataclasses. Use a class when an object must hold state or
restore state, for example a context manager that changes model weights and puts them back (`MixedQuant`, `GPTQQuant`).
Do not build class hierarchies for experiments.

## Python standards

- Follow PEP 8. Use `ruff` for formatting and lint checks when it is configured.
- Add type hints to public functions, methods, and important internal interfaces.
- Use descriptive names. Write units and shapes in names or docstrings (`out_features`, `n_questions`, `bits`).
- Prefer explicit keyword arguments to long positional argument lists.
- Use `pathlib` for filesystem paths.
- Use context managers for resources that need cleanup.
- Use structured logging (`logging`) instead of `print()` in library code under `compress/`.
- Use specific exception types. Write error messages that name the bad value.
- Do not use bare `except` clauses. Do not swallow exceptions.
- Do not use mutable default arguments.
- Do not use wildcard imports.
- Avoid global state. Keep module-level state to constants.
- Use constants for shared configuration values (prompts, seeds, default sizes).
- Order imports: standard library, third party, local. Separate the groups with a blank line.
- Prefer the standard library when it is enough.
- Use dataclasses or typed structures for important data contracts, for example a result row.
- Keep each function focused on one responsibility. Split a function that needs a long comment to explain its sections.
- Keep each class cohesive.
- Write a docstring for every public function. State what it returns, the shapes of tensors, and what it changes.

**(project)** Command-line scripts in `scripts/` may write progress lines to stderr with one shared `log` helper.
They must write results to files, not to stdout. New scripts must keep `main()` short:
parse arguments, build the experiment, run it, save the result. Put the logic in `compress/`.

## Testing

- Run `python -m pytest -q` from the repository root before you finish. Tests must run offline and on CPU.
- Add a test for every new behavior and for every bug you fix. Write the failing test first when you can.
- Use the tiny model and tokenizer fixtures in `tests/conftest.py`. Do not download models in tests.
- Test properties, not only examples: a restored weight is bit-identical, a protected set of zero columns equals the
  uniform result, a quantized value lies on the grid, a patch of all units reproduces the clean run.
- Compare to an independent reference when one exists (for example the authors' GPTQ code).
- Do not weaken a test to make it pass. If an expected value was wrong, say so in the commit message.
- Mark tests that need a GPU with `pytest.mark.skipif` and say why.
- Do not claim that a test, command, or experiment works unless you ran it.

## Performance and resources **(project)**

- Assume the GPU is shared and small. A 4B model in fp16 uses about 8.4 GB. Keep copies of weights on the host
  (`offload=True`) when memory is short.
- Do not keep activations or full-vocabulary logits you do not need. Use last-position logits.
- Batch work when the result does not change. Prove that with a test.
- Measure time and peak memory before you optimize, and report them when you claim a cost.
