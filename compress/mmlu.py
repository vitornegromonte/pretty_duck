"""Canonical MMLU CSVs, cached from the ``cais/mmlu`` data tarball.

The Hugging Face mirror only ships parquet files; the canonical tarball has the
original headerless CSVs.  Using stdlib ``tarfile``/``csv`` keeps ``datasets``
and ``pyarrow`` out of the dependency list.  The 159 MB tarball is downloaded
once and only the requested subject/split CSV is extracted.
"""

from __future__ import annotations

import csv
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

MMLU_URL = "https://huggingface.co/datasets/cais/mmlu/resolve/main/data.tar"
LETTERS = "ABCD"
SPLIT_ALIASES = {"validation": "val", "val": "val", "test": "test", "dev": "dev"}


def default_cache_dir() -> Path:
    """``.cache/mmlu`` by default; override with ``PRETTY_DUCK_CACHE``."""
    return Path(os.environ.get("PRETTY_DUCK_CACHE", ".cache")) / "mmlu"


def _safe_subject(subject: str) -> str:
    if not subject or any(c not in "abcdefghijklmnopqrstuvwxyz_" for c in subject):
        raise ValueError(f"invalid MMLU subject {subject!r}")
    return subject


def _download_tar(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"downloading MMLU tarball (~160 MB) to {path}", file=sys.stderr)
    urllib.request.urlretrieve(MMLU_URL, tmp)
    tmp.replace(path)


def load_mmlu(
    subject: str, split: str = "test", cache_dir: str | Path | None = None
) -> list[dict]:
    """Return rows ``{question, choices, answer}`` for one MMLU subject/split.

    ``split`` accepts ``validation`` as an alias for the tarball's ``val``.
    ``answer`` is the index into the four ``choices``.
    """
    subject = _safe_subject(subject)
    split = SPLIT_ALIASES.get(split, split)
    if split not in ("test", "val", "dev"):
        raise ValueError(f"unknown MMLU split {split!r}; use test/validation/dev")
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    csv_path = cache / "data" / split / f"{subject}_{split}.csv"

    if not csv_path.exists():
        tar_path = cache / "data.tar"
        if not tar_path.exists():
            _download_tar(tar_path)
        member = f"data/{split}/{subject}_{split}.csv"
        with tarfile.open(tar_path, "r:*") as tf:
            src = tf.extractfile(member)
            if src is None:
                raise KeyError(
                    f"{subject!r}/{split} not found in {tar_path}; "
                    "check the MMLU subject name"
                )
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
            tmp.write_bytes(src.read())
            tmp.replace(csv_path)

    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) != 6:
                raise ValueError(
                    f"{csv_path}: expected 6 columns (question, A-D, answer), "
                    f"got {len(row)}"
                )
            answer = row[5].strip().upper()
            if answer not in LETTERS:
                raise ValueError(f"{csv_path}: invalid answer key {row[5]!r}")
            rows.append(
                {"question": row[0], "choices": row[1:5], "answer": LETTERS.index(answer)}
            )
    return rows
