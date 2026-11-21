"""Stop hook (not enabled by default): run the tests before Claude ends its turn.

Exit code 2 sends the failure back to Claude so it keeps working. To enable it, add this to ``.claude/settings.json``:

    "Stop": [{"hooks": [{"type": "command", "command": "python3 \\"$CLAUDE_PROJECT_DIR\\"/.claude/hooks/run_tests.py"}]}]

It adds about 10 to 30 seconds to every turn. Claude Code stops the loop after 8 blocks in a row.
"""

import json
import subprocess
import sys


def main() -> int:
    event = json.load(sys.stdin)
    if event.get("stop_hook_active"):
        return 0
    result = subprocess.run([".venv/bin/python", "-m", "pytest", "-q", "-x"], capture_output=True, text=True)
    if result.returncode != 0:
        print("The tests fail. Fix them before you finish.\n" + result.stdout[-1500:], file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
