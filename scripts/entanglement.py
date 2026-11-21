"""Is there a domain-specific remainder once the shared core is removed?

Reuses the single-unit scores of ``circuit_probe.py`` (discovery half) and
re-evaluates circuits on each domain's held-out pairs.  For every domain A and
size k three circuits are built from A's scores:

  full      the top-k units;
  no_core   the top-k after removing the shared core (units in the top
            ``--core-k`` of at least ``--core-min`` domains);
  specific  the top-k after removing every unit in the top ``--excl-k`` of any
            *other* domain.

Each is scored (sufficiency / necessity of the held-out logit difference) on
every domain B.  A domain-specific remainder shows as diagonal >> off-diagonal.

    PYTHONPATH=. python scripts/entanglement.py --model-path Qwen/Qwen3.5-0.8B \
        --probe outputs/circuit_probe.json --out outputs/entanglement.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

from compress.circuit import (Scanner, Unit, build_facts, core_units, enumerate_units, excluding,
                              make_pairs, match_random, split_facts, top_units, unit_sets)
from compress.models import load_model
from compress.facts import FACTS, TEMPLATES

log = lambda m: print(m, file=sys.stderr, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--probe", required=True, help="circuit_probe.json with per-unit scores")
    p.add_argument("--ks", default="4,8,16,32,64")
    p.add_argument("--core-k", type=int, default=32)
    p.add_argument("--core-min", type=int, default=2)
    p.add_argument("--excl-k", type=int, default=64)
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    probe = json.loads(Path(args.probe).read_text())
    domains = list(probe["domains"])
    seed = probe["config"]["seed"]
    mlp_group = probe["config"]["mlp_group"]
    model, tokenizer = load_model(args.model_path)
    units = enumerate_units(model, mlp_group)
    kind = {(u.layer, u.start, u.stop): u for u in units}
    layers = sorted(unit_sets(units))

    scores, val = {}, {}
    for d in domains:
        scores[d] = [{"unit": kind[(l, s, e)], "suff": suff, "nec": nec}
                     for l, s, e, suff, nec in probe["domains"][d]["units"]]
        facts = build_facts(model, tokenizer, TEMPLATES[d], FACTS[d])
        fa, fb = split_facts(facts, seed)
        assert len(make_pairs(fa, seed).tok_clean) == probe["domains"][d]["n_scan_pairs"], "split changed"
        val[d] = Scanner(model, make_pairs(fb, seed + 1), layers).offload()  # keep GPU memory for the model
        log(f"[{d}] validation pairs {len(fb)}")

    core = core_units(scores, args.core_k, args.core_min)
    core_all = core_units(scores, args.core_k, len(domains))
    log(f"core (top-{args.core_k} in >= {args.core_min} domains): {len(core)} units; in all {len(domains)}: {len(core_all)}")

    def circuit(d: str, k: int, variant: str) -> list[Unit]:
        if variant == "full":
            return top_units(scores[d], k)
        if variant == "no_core":
            return top_units(excluding(scores[d], core), k)
        others = set().union(*[set(top_units(scores[o], args.excl_k)) for o in domains if o != d])
        return top_units(excluding(scores[d], others), k)

    rng = random.Random(args.seed)
    out = {"config": vars(args) | {"core_size": len(core), "core_all_size": len(core_all)},
           "core": [[u.layer, u.start, u.stop] for u in sorted(core, key=lambda u: (u.layer, u.start))],
           "cells": {}}
    for label, cset in (("core", sorted(core, key=lambda u: (u.layer, u.start))),):
        out["cells"][f"{label}->"] = {b: {"size": len(cset), "suff": val[b].restoration(unit_sets(cset)),
                                          "nec": val[b].necessity(unit_sets(cset))} for b in domains}
    for variant in ("full", "no_core", "specific"):
        for k in ks:
            for a in domains:
                circ = circuit(a, k, variant)
                rand = [match_random(units, circ, rng) for _ in range(args.n_random)]
                row = {}
                for b in domains:
                    row[b] = {"size": len(circ), "suff": val[b].restoration(unit_sets(circ)),
                              "nec": val[b].necessity(unit_sets(circ)),
                              "rand_suff": st.mean(val[b].restoration(unit_sets(r)) for r in rand),
                              "rand_nec": st.mean(val[b].necessity(unit_sets(r)) for r in rand)}
                out["cells"][f"{variant}|{k}|{a}"] = row
        log(f"done variant {variant}")
    Path(args.out).write_text(json.dumps(out))

    print(f"core units ({len(core)}): " + ", ".join(f"L{u.layer.split('layers.')[1].split('.')[0]}.{u.layer.split('.')[-1]}[{u.start}]" for u in sorted(core, key=lambda u: (u.layer, u.start))))
    print("core alone, sufficiency / necessity on each domain:",
          {b: (round(v['suff'], 2), round(v['nec'], 2)) for b, v in out["cells"]["core->"].items()})
    for variant in ("full", "no_core", "specific"):
        print(f"\n== {variant}: necessity on own domain (diag) vs mean of the other domains (off); random in brackets")
        for k in ks:
            cells = []
            for a in domains:
                row = out["cells"][f"{variant}|{k}|{a}"]
                diag = row[a]["nec"]; off = st.mean(row[b]["nec"] for b in domains if b != a)
                cells.append(f"{a[:5]} {diag:+.2f}/{off:+.2f}({row[a]['rand_nec']:+.2f})")
            print(f"   k={k:3d}: " + "   ".join(cells))


if __name__ == "__main__":
    main()
