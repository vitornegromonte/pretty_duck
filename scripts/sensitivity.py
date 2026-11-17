"""Measured per-layer sensitivity table and additivity test.

Step 1: for every Linear in a stats file and every format, quantize that layer
alone and measure the NLL increase per domain on a *measure* split of held-out
text.  Step 2: quantize random sets of layers together (uniform int4 and mixed
formats) and measure the joint increase on the measure split and on a disjoint
*held-out* split, so the additive prediction sum_g Delta L_g can be checked.

    PYTHONPATH=. python scripts/sensitivity.py --model-path Qwen/Qwen3.5-0.8B \
        --stats stats_dom.pt --eval math=outputs/eval/math.txt --eval law=outputs/eval/law.txt \
        --out outputs/sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from compress import stats as stats_mod
from compress.cli import _load_model, _parse_pairs
from compress.sensitivity import measure_plans, measure_sensitivity


def read_lines(path: str, seed: int) -> list[str]:
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    random.Random(seed).shuffle(lines)
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--stats", required=True, help="stats file; its layers are the units")
    p.add_argument("--eval", action="append", default=[], metavar="DOMAIN=FILE")
    p.add_argument("--formats", default="int8,int4,nf4")
    p.add_argument("--n-measure", type=int, default=64)
    p.add_argument("--n-heldout", type=int, default=64)
    p.add_argument("--plan-sizes", default="4,12,40")
    p.add_argument("--plans-per-size", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    files = _parse_pairs(args.eval, "eval")
    if not files:
        raise SystemExit("provide at least one --eval domain=file.txt")
    formats = args.formats.split(",")
    layers = sorted(next(iter(stats_mod.load_stats(args.stats)[0].values())))
    measure, heldout = {}, {}
    for d, path in files.items():
        lines = read_lines(path, args.seed)
        if len(lines) < args.n_measure + args.n_heldout:
            raise SystemExit(f"{path}: {len(lines)} lines, need {args.n_measure + args.n_heldout}")
        measure[d] = lines[: args.n_measure]
        heldout[d] = lines[args.n_measure: args.n_measure + args.n_heldout]

    model, tokenizer = _load_model(args.model_path)
    log = lambda m: print(m, file=sys.stderr, flush=True)
    kw = dict(batch_size=args.batch_size, max_len=args.max_len, log=log)

    print(f"{len(layers)} layers x {len(formats)} formats x {len(files)} domains", flush=True)
    table = measure_sensitivity(model, tokenizer, measure, layers, formats, **kw)

    rng = random.Random(args.seed)
    plans: dict[str, dict[str, str]] = {}
    for size in [int(s) for s in args.plan_sizes.split(",")]:
        for i in range(args.plans_per_size):
            chosen = rng.sample(layers, size)
            plans[f"int4_m{size}_{i}"] = {layer: "int4" for layer in chosen}
            chosen = rng.sample(layers, size)
            plans[f"mixed_m{size}_{i}"] = {layer: rng.choice(formats) for layer in chosen}
    plan_measure = measure_plans(model, tokenizer, measure, plans, **kw)
    plan_heldout = measure_plans(model, tokenizer, heldout, plans, **kw)

    Path(args.out).write_text(json.dumps({
        "config": {"model": args.model_path, "stats": args.stats, "formats": formats,
                   "n_measure": args.n_measure, "n_heldout": args.n_heldout, "seed": args.seed,
                   "eval_files": files},
        "table": table, "plans": plans, "plan_measure": plan_measure, "plan_heldout": plan_heldout,
    }))


if __name__ == "__main__":
    main()
