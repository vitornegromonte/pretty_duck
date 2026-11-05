"""Rung 3: quantize only persona-selected weight columns; does the persona shift survive?

For each channel-set kind (zdiff, tarc, importance, V-matched random) and each
``bits``, fake-quantize those weight columns and measure how much of the
seek-minus-reject margin shift is retained.  Random sets are matched to the
zdiff set on size and V rank, and every condition reports its diagonal dose
sum_ij E_ij^2 V_j (V from the seek stats), so equal-dose comparisons are
visible.  ``bits`` below 4 are intervention strengths only (nothing is stored).

    PYTHONPATH=. python scripts/persona_quant.py --model-path Qwen/Qwen3.5-0.8B \
        --stats outputs/stats_persona.pt --test Persona/data/wealth_test_infer.csv \
        --out outputs/persona_quant.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from persona_gate import PROMPTS  # noqa: E402

from compress import stats as stats_mod  # noqa: E402
from compress.cli import _load_model  # noqa: E402
from compress.patching import select_channels  # noqa: E402
from compress.quant_probe import quant_persona_probe  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target", default="seek")
    p.add_argument("--others", action="append", default=None)
    p.add_argument("--k-frac", type=float, action="append", default=None)
    p.add_argument("--bits", type=int, action="append", default=None)
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--prompt-style", default="auto")
    args = p.parse_args()
    k_fracs, bits_list, others = args.k_frac or [0.05, 0.2], args.bits or [4, 3, 2], args.others or ["reject"]

    stats, _ = stats_mod.load_stats(args.stats)
    V = {n: stats[args.target][n]["V"] for n in stats[args.target]}
    conds: dict[str, tuple[dict, int]] = {}
    for bits in bits_list:
        conds[f"int{bits}_all"] = ({n: torch.arange(v.numel()) for n, v in V.items()}, bits)
        for k in k_fracs:
            sets = {kind: select_channels(stats, args.target, others, k, kind)
                    for kind in ("zdiff", "tarc", "importance")}
            for kind, s in sets.items():
                conds[f"int{bits}_k{k}_{kind}"] = (s, bits)
            for rep in range(args.n_random):
                conds[f"int{bits}_k{k}_random{rep}"] = (select_channels(
                    stats, args.target, others, k, "random", ref_sets=sets["zdiff"], seed=rep), bits)

    model, tokenizer = _load_model(args.model_path)
    with open(args.test, encoding="utf-8") as f:
        rows = [{k: r[k].strip() for k in ("question", "A", "B")} | {"matching": r["matching"].strip().upper()}
                for r in csv.DictReader(f)][: args.max_questions]
    prompts = {k: PROMPTS[k] for k in ("neutral", "seek", "reject")}
    print(f"{len(conds)} conditions, {len(rows)} questions", flush=True)
    report = quant_persona_probe(model, tokenizer, rows, prompts=prompts, conditions=conds, V=V,
                                 style=args.prompt_style, n_boot=args.n_boot, keep_per_question=True,
                                 log=lambda m: print(m, flush=True))
    report["config"] = {"model": args.model_path, "stats": args.stats, "test": args.test,
                        "k_fracs": k_fracs, "bits": bits_list, "n_random": args.n_random}
    Path(args.out).write_text(json.dumps(report))


if __name__ == "__main__":
    main()
