"""Build one-line-per-sample calibration files from MMLU validation splits.

Calibration uses the ``val`` split (questions and choices, no answer keys) so
it never overlaps the ``test`` split used by ``evaluate_compression --mmlu``.

    python scripts/make_mmlu_calib.py --out-dir outputs/calib \
        math=college_mathematics,elementary_mathematics,high_school_mathematics,abstract_algebra \
        law=professional_law,international_law,jurisprudence
"""

from __future__ import annotations

import argparse
from pathlib import Path

from compress.mmlu import LETTERS, load_mmlu


def sample_text(row: dict) -> str:
    """Question plus lettered choices on a single line (answer key omitted)."""
    choices = " ".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
    return " ".join(f"{row['question']} {choices}".split())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("domains", nargs="+", metavar="DOMAIN=SUBJECT[,SUBJECT...]")
    p.add_argument("--out-dir", default="outputs/calib")
    p.add_argument("--split", default="val")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for spec in args.domains:
        domain, _, subjects = spec.partition("=")
        if not domain or not subjects:
            raise SystemExit(f"expected DOMAIN=SUBJECT[,SUBJECT...], got {spec!r}")
        lines = [
            sample_text(row)
            for subject in subjects.split(",")
            for row in load_mmlu(subject, args.split)
        ]
        path = out / f"{domain}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{path}: {len(lines)} samples from {subjects}")


if __name__ == "__main__":
    main()
