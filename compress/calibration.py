"""Text for calibration and for perplexity checks.

Calibration text is a list of lines (one text per line). Batches are tokenizer outputs with ``input_ids`` and
``attention_mask``. Calibration lines must never overlap the evaluation rows (see ``mmlu.load_mmlu_groups``).
"""

from __future__ import annotations

import random
from pathlib import Path


def read_lines(path: str | Path, *, min_chars: int = 1) -> list[str]:
    """Non-empty lines of a text file that have at least ``min_chars`` characters."""
    lines = (ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines())
    return [ln for ln in lines if len(ln) >= min_chars]


def text_batches(tokenizer, lines: list[str], n: int, *, max_len: int = 512, batch_size: int = 4) -> list[dict]:
    """Tokenize the first ``n`` lines into padded batches."""
    lines = lines[:n]
    return [tokenizer(lines[i:i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            for i in range(0, len(lines), batch_size)]


def mixed_domain_batches(tokenizer, directory: str | Path, domains: tuple[str, ...], n: int, *, seed: int,
                         max_len: int = 512, batch_size: int = 4) -> list[dict]:
    """Batches that mix ``n`` lines evenly over ``<directory>/<domain>.txt`` files (one text file per domain)."""
    rng = random.Random(seed)
    lines: list[str] = []
    for domain in domains:
        domain_lines = read_lines(Path(directory) / f"{domain}.txt")
        rng.shuffle(domain_lines)
        lines += domain_lines[: n // len(domains)]
    rng.shuffle(lines)
    return text_batches(tokenizer, lines, len(lines), max_len=max_len, batch_size=batch_size)
