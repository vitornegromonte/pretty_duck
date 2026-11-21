"""PreToolUse hook: block Edit and Write on experiment outputs and checkpoints.

Claude Code sends the tool call as JSON on stdin. Exit code 2 blocks the call and sends stderr to Claude.
Rule of the project: results in ``outputs/`` and ``*.pt`` files are never edited by hand.
"""

import json
import sys
from pathlib import Path


def blocked(file_path: str, project_dir: Path) -> str | None:
    path = Path(file_path)
    try:
        relative = path.resolve().relative_to(project_dir.resolve())
    except ValueError:
        return None
    if relative.parts and relative.parts[0] == "outputs":
        return "outputs/ holds experiment results. Do not edit it by hand. Write a script that produces the file."
    if path.suffix == ".pt":
        return "Do not edit checkpoints or tensors (*.pt) by hand."
    return None


def main() -> int:
    event = json.load(sys.stdin)
    file_path = (event.get("tool_input") or {}).get("file_path")
    if not file_path:
        return 0
    project_dir = Path(event.get("cwd") or ".")
    reason = blocked(file_path, project_dir)
    if reason:
        print(reason, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
