"""Provenance of a result file: which code, which software, and which hardware produced it.

Every experiment script stores ``provenance()`` next to its ``config``. Without it, two runs of the same command
cannot be compared (an 8.8-point difference between two "identical" runs had no recorded cause). No personal data
is stored: no user name, no host name, no paths.
"""

from __future__ import annotations

import datetime
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def provenance(argv: list[str] | None = None) -> dict:
    """Return a JSON-serializable record of the code, software, and hardware of this run."""
    status = _git("status", "--porcelain")
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:  # pragma: no cover
        transformers_version = None
    cuda = torch.cuda.is_available()
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv if argv is None else argv),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda": torch.version.cuda if cuda else None,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if cuda else [],
        "default_dtype": str(torch.get_default_dtype()),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }
