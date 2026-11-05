"""Build seek/reject calibration texts from a Persona ``*_train.csv``.

Each CSV row (question, matching, not_matching) becomes one single-line sample
per side: ``"<question> <answer>"``.  Parsing with ``csv`` keeps multi-line
fields intact, unlike reading the raw file line by line.

    python scripts/make_persona_calib.py Persona/data/wealth_train.csv --out-dir outputs/persona_calib
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def one_line(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("train_csv")
    p.add_argument("--out-dir", default="outputs/persona_calib")
    args = p.parse_args()
    with open(args.train_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, col in (("seek", "matching"), ("reject", "not_matching")):
        lines = [one_line(f"{r['question']} {r[col]}") for r in rows if r[col].strip()]
        (out / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{out / name}.txt: {len(lines)} samples")


if __name__ == "__main__":
    main()
