# Workflow standards

## Version control

- Make the smallest change that solves the problem. Do not mix a refactor and a behavior change in one commit.
- Write a commit message with a short subject in the imperative mood and a body that gives the reason and the evidence.
- Do not push, force-push, or rewrite history unless the user asks.
- Do not change files on the remote machine while a run uses them. Sync `compress/` and `scripts/` only, and never `outputs/`.

## Review checklist

Before you finish a change, check each item:

1. The change is small and does what the task asks. No unrelated edits.
2. Names, types, and docstrings are clear. No dead code and no commented-out code.
3. No duplicated logic that already exists in `compress/`.
4. Weights are restored, seeds are set, and results record their configuration.
5. Tests exist for the new behavior. `python -m pytest -q` passes. You report the real result.
6. Documentation and the ADR match the code and the numbers.
7. You state what you could not verify.
