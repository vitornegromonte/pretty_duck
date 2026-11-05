"""Minimal-pair patching of the persona margin, with lexical-priming controls.

For each prompt pair (real seek/reject, word-shuffled, unrelated) we patch
clean-run Linear inputs over the aligned suffix into the corrupt run and report
how much of the clean-minus-corrupt margin each channel set restores.  A set
is persona-related only if it restores more of the *real* pair than of the
shuffled one.

    PYTHONPATH=. python scripts/persona_patch.py --model-path Qwen/Qwen3.5-0.8B \
        --stats outputs/stats_persona.pt --test Persona/data/wealth_test_infer.csv \
        --out outputs/persona_patch.json
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
from compress.pair_patching import layer_of, pair_patch_probe  # noqa: E402
from compress.patching import select_channels  # noqa: E402

PAIRS = {
    "real": ("seek", "reject"),
    "shuffled": ("shuffled_seek", "shuffled_reject"),
    "irrelevant": ("irrelevant_a", "irrelevant_b"),
}


def build_conditions(stats, target, others, k_fracs, n_random, layer_sweep=True):
    names = sorted(stats[target])
    modules_in = {n: stats[target][n]["A"].numel() for n in names}
    conds = {"ceiling_all": {n: torch.arange(d) for n, d in modules_in.items()}}
    for layer in sorted({layer_of(n) for n in names if layer_of(n) is not None} if layer_sweep else ()):
        conds[f"layer{layer:02d}_all"] = {
            n: torch.arange(d) for n, d in modules_in.items() if layer_of(n) == layer}
    for k in k_fracs:
        sets = {kind: select_channels(stats, target, others, k, kind)
                for kind in ("tarc", "zdiff", "importance")}
        for kind, s in sets.items():
            conds[f"k{k}_{kind}"] = s
        for rep in range(n_random):
            conds[f"k{k}_random{rep}"] = select_channels(
                stats, target, others, k, "random", ref_sets=sets["tarc"], seed=rep)
    return conds


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target", default="seek")
    p.add_argument("--others", action="append", default=None)
    p.add_argument("--pairs", default="real,shuffled,irrelevant")
    p.add_argument("--k-frac", type=float, action="append", default=None)
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--prompt-style", default="auto")
    p.add_argument("--reverse", action="store_true",
                   help="swap clean/corrupt: patch the corrupt prompt's activations into the clean "
                        "run (necessity: fraction of the shift destroyed)")
    p.add_argument("--no-layer-sweep", action="store_true")
    p.add_argument("--per-question", action="store_true")
    args = p.parse_args()

    stats, _ = stats_mod.load_stats(args.stats)
    conds = build_conditions(stats, args.target, args.others or ["reject"],
                             args.k_frac or [0.05, 0.2], args.n_random,
                             layer_sweep=not args.no_layer_sweep)
    model, tokenizer = _load_model(args.model_path)
    with open(args.test, encoding="utf-8") as f:
        rows = [{k: r[k].strip() for k in ("question", "A", "B")} | {"matching": r["matching"].strip().upper()}
                for r in csv.DictReader(f)][: args.max_questions]
    out = {"config": {"model": args.model_path, "stats": args.stats, "test": args.test,
                      "n_conditions": len(conds), "prompt_style": args.prompt_style,
                      "direction": "necessity" if args.reverse else "sufficiency"},
           "prompts": PROMPTS, "pairs": {}}
    for name in args.pairs.split(","):
        clean, corrupt = PAIRS[name][::-1] if args.reverse else PAIRS[name]
        print(f"== pair {name}: clean={clean!r} corrupt={corrupt!r}, {len(conds)} conditions", flush=True)
        out["pairs"][name] = pair_patch_probe(
            model, tokenizer, rows, clean_sys=PROMPTS[clean], corrupt_sys=PROMPTS[corrupt],
            conditions=conds, style=args.prompt_style, n_boot=args.n_boot,
            keep_per_question=args.per_question,
            log=lambda m: print(f"  [{name}] {m}", flush=True))
        r = out["pairs"][name]
        print(f"  margins clean {r['mean_margin']['clean']:+.3f} corrupt {r['mean_margin']['corrupt']:+.3f}; "
              f"ceiling restoration {r['conditions']['ceiling_all']['restoration']:.3f}", flush=True)
        Path(args.out).write_text(json.dumps(out, indent=1))  # keep partial results


if __name__ == "__main__":
    main()
