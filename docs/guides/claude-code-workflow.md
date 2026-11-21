# Working with Claude Code

Source: Anthropic, "Best practices for Claude Code" (https://code.claude.com/docs/en/best-practices).
Items marked *(ours)* are our own adaptation. Review this section when the workflow changes.

1. **Give every task a check that returns pass or fail.** Without a check, "looks done" is the only evidence.
   Checks can be a test, a script that compares a result with a saved one, a linter, or a build exit code.
   Put the command that runs the check in `CLAUDE.md`.
2. **Ask for evidence, not claims.** Report the command you ran and its output. "All tests pass" alone is weak evidence.
3. **Explore, plan, implement, commit.** First read the code and the ADR without changing anything. Make a plan when
   the change touches several files or the code is new to you. Skip the plan when you can describe the diff in one sentence.
4. **Write specific requests.** Name the files, describe the symptom, say what "done" looks like, and state which
   approach you prefer. For a large task, write a short `SPEC.md` first (goal, inputs, outputs, checks). Build from it in a fresh session.
5. **Keep context clean.** Clear the context between unrelated tasks. After two failed corrections, stop, clear, and
   write one new request that includes what you learned. Use a subagent for broad searches, and for an independent
   review with fresh context. Give a review subagent read-only tools.
6. **Commit small steps as checkpoints.** Each step is easy to revert. A rewind of the conversation does not replace git.
7. **Keep `CLAUDE.md` short.** Include only what the code cannot tell: commands, places where the code misleads,
   decisions that are not in the repository, and anything you corrected twice. Remove a line when Claude would do the
   right thing without it. Stay under about 200 lines. Emphasis ("IMPORTANT") does not fix an ignored rule: shorten
   the file, or move a rule that has no exceptions into a hook. Put long procedures in a skill or a separate guide in
   `agents/Rules/`.
8. **Enforce what must never fail with the harness, not with prose.** Use hooks and deny rules for rules without
   exceptions. Candidates for this project *(ours)*: a `Stop` hook that runs `python -m pytest -q`, and a rule that
   blocks edits in `outputs/`. Propose a hook to the user. Do not change settings files without a request.
9. **Unattended runs need five things:** a definition of done, a way for the run to verify itself, limited permissions
   (a branch or a worktree), a limit on turns or cost, and a human approval for every irreversible step (push, delete,
   sync to the remote machine, overwrite results).
10. **Use a separate session or worktree for parallel work.** Use a second session as an independent reviewer.
