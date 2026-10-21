#!/usr/bin/env python3
"""Summarize evaluate_compression reports and compare paired runs.

Usage:
    python scripts/check_results.py
    python scripts/check_results.py eval_error.json eval_contrast.json
    python scripts/check_results.py --baseline eval_error.json \
        --candidate eval_contrast.json

The paired section needs the per-question ``predictions`` written by MMLU
evaluation.  Delta is candidate - baseline, so negative dNLL is better.
Accuracy uses the exact McNemar test on discordant questions; NLL uses a
paired bootstrap over questions.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    report = json.loads(path.read_text())
    if "results" not in report:
        raise SystemExit(f"{path} is not an evaluate_compression report")
    return report


def _variant(report: dict) -> str:
    parts = []
    if report.get("compressed_dir"):
        parts.append(Path(report["compressed_dir"]).name)
    mask = report.get("mask")
    if mask:
        bits = [str(mask.get("kind", "mask"))]
        if mask.get("sparsity") is not None:
            bits.append(f"sp={mask['sparsity']}")
        if mask.get("only_mlp"):
            bits.append("mlp")
        if mask.get("z_clip") is not None:
            bits.append(f"zclip={mask['z_clip']}")
        parts.append(",".join(bits))
    return "+".join(parts) or "full"


def _short(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def _summary(rows: list[tuple[Path, dict]]) -> None:
    print(
        f"{'report':<26} {'variant':<22} {'domain':<10} "
        f"{'n':>5} {'acc%':>7} {'NLL':>8} {'x fp16':>7}"
    )
    for path, report in rows:
        variant = _short(_variant(report), 22)
        ratio = report.get("compression_ratio_vs_fp16_linears")
        ratio_text = f"{ratio:.2f}" if ratio else "-"
        for domain, result in sorted(report["results"].items()):
            print(
                f"{_short(path.name, 26):<26} {variant:<22} {domain:<10} "
                f"{result['n']:>5} {result['accuracy']:>7.2f} "
                f"{result['nll_per_token']:>8.4f} {ratio_text:>7}"
            )


def _mcnemar(base_only: int, candidate_only: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts."""
    n = base_only + candidate_only
    if n == 0:
        return 1.0
    k = min(base_only, candidate_only)
    log_terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        for i in range(k + 1)
    ]
    top = max(log_terms)
    tail = math.exp(top - n * math.log(2.0)) * sum(
        math.exp(term - top) for term in log_terms
    )
    return min(1.0, 2.0 * tail)


def _bootstrap(
    diffs: list[float], reps: int = 2000, seed: int = 0
) -> tuple[float, float]:
    if len(diffs) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(diffs)
    means = []
    for _ in range(reps):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def _paired_domain(
    base: dict, candidate: dict, domain: str, reps: int, seed: int
) -> None:
    ra, rb = base["results"][domain], candidate["results"][domain]
    d_acc = 100.0 * (rb["correct"] - ra["correct"]) / ra["n"] if ra["n"] else 0.0
    d_nll = rb["nll_per_token"] - ra["nll_per_token"]
    print(
        f"  {domain}: n={ra['n']}->{rb['n']}  "
        f"acc {ra['accuracy']:.2f}->{rb['accuracy']:.2f}% ({d_acc:+.2f} pp)  "
        f"NLL {ra['nll_per_token']:.4f}->{rb['nll_per_token']:.4f} "
        f"({d_nll:+.4f})"
    )

    pa = ra.get("predictions") or []
    pb = rb.get("predictions") or []
    if not pa or not pb:
        print("    (no per-question predictions; aggregate delta only)")
        return
    if len(pa) != len(pb):
        print("    ! prediction counts differ; skipping paired tests")
        return
    if any(x["expected"] != y["expected"] for x, y in zip(pa, pb)):
        print("    ! expected-answer sequence differs; skipping paired tests")
        return
    if ra.get("subject") and rb.get("subject") and ra["subject"] != rb["subject"]:
        print("    ! subjects differ; skipping paired tests")
        return

    base_ok = [p["predicted"] == p["expected"] for p in pa]
    cand_ok = [p["predicted"] == p["expected"] for p in pb]
    base_only = sum(1 for a, b in zip(base_ok, cand_ok) if a and not b)
    cand_only = sum(1 for a, b in zip(base_ok, cand_ok) if not a and b)
    p_value = _mcnemar(base_only, cand_only)

    diffs = [y["nll"] - x["nll"] for x, y in zip(pa, pb)]
    low, high = _bootstrap(diffs, reps=reps, seed=seed)
    mean = sum(diffs) / len(diffs)
    print(
        f"    paired acc: base-only {base_only}, candidate-only {cand_only}, "
        f"McNemar p={p_value:.3g}"
    )
    print(
        f"    paired NLL: {mean:+.4f} 95% CI [{low:+.4f}, {high:+.4f}]"
    )


def _compare(
    base_path: Path, base: dict, cand_path: Path, candidate: dict,
    reps: int, seed: int,
) -> None:
    print(f"\n{_variant(candidate)} - {_variant(base)} ({cand_path.name} vs {base_path.name})")
    if base.get("model") != candidate.get("model"):
        print("  ! models differ; comparison is not apples-to-apples")
    if base.get("prompt_style") != candidate.get("prompt_style"):
        print("  ! prompt_style differs; comparison is not apples-to-apples")
    common = sorted(set(base["results"]) & set(candidate["results"]))
    if not common:
        print("  no shared domains")
    for domain in common:
        _paired_domain(base, candidate, domain, reps, seed)


def _pick_baseline(paths: list[Path]) -> Path | None:
    for key in ("error", "nomask", "no_mask", "none", "baseline", "plain", "full"):
        for path in paths:
            stem = path.stem.lower()
            if key in stem and "contrast" not in stem:
                return path
    return None


def _self_test() -> None:
    assert abs(_mcnemar(0, 0) - 1.0) < 1e-12
    assert abs(_mcnemar(10, 0) - 2 * 0.5**10) < 1e-12
    low, high = _bootstrap([1.0, 1.0, 1.0])
    assert low == high == 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files", nargs="*", help="report JSONs (default: eval*.json mmlu*.json *report*.json)"
    )
    parser.add_argument("--baseline", help="baseline report (default: auto-detect)")
    parser.add_argument(
        "--candidate", action="append", default=[], help="candidate report (repeatable)"
    )
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        print("self-test passed")
        return

    names = args.files
    if not names:
        names = sorted(
            {p for pattern in ("eval*.json", "mmlu*.json", "*report*.json")
             for p in glob.glob(pattern)}
        )
    if not names:
        raise SystemExit("no report JSONs found; pass paths explicitly")

    paths, seen = [], set()
    for name in names:
        path = Path(name)
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.exists():
            raise SystemExit(f"missing report: {path}")
        paths.append(path)

    loaded = [(path, _load(path)) for path in paths]
    _summary(loaded)
    print("\nDelta = candidate - baseline; negative dNLL is better. "
          "McNemar p is exact and two-sided.")

    baseline_path = Path(args.baseline) if args.baseline else _pick_baseline(paths)
    if baseline_path is None:
        print("\nNo baseline auto-detected; use --baseline/--candidate to compare runs.")
        return
    if baseline_path not in paths:
        paths.append(baseline_path)
        loaded.append((baseline_path, _load(baseline_path)))
    baseline = dict(loaded)[baseline_path]

    candidate_paths = [Path(p) for p in args.candidate]
    if candidate_paths:
        for path in candidate_paths:
            if path not in paths:
                loaded.append((path, _load(path)))
    else:
        candidate_paths = [p for p in paths if p != baseline_path]

    reports = dict(loaded)
    for path in candidate_paths:
        _compare(baseline_path, baseline, path, reports[path], args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
