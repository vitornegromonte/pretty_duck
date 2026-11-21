"""Neuron-level re-test of the entanglement result.

The group-level scan (``circuit_probe.py``) scored attention heads and groups of
64 MLP neurons.  Here every MLP neuron (input channel of ``down_proj``) of the
layers that matter is scored by batched single-neuron patching, heads stay as
units, and the entanglement test of ``entanglement.py`` is repeated on the
mixed pool.  If neurons mostly have one dominant function, circuits built from
them should be sparser and more domain-specific than the group-level ones.

    PYTHONPATH=. python scripts/neuron_scan.py --model-path Qwen/Qwen3.5-0.8B \
        --probe outputs/circuit_probe.json --out outputs/neuron_scan.json
    # timing only:  ... --bench-layer 16
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
import time
from pathlib import Path

import torch

from compress.circuit import (Scanner, Unit, build_facts, core_units, enumerate_units, excluding,
                              make_pairs, match_random, scan_neurons, split_facts, top_units, unit_sets)
from compress.models import load_model
from compress.facts import FACTS, TEMPLATES

log = lambda m: print(m, file=sys.stderr, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--layers", default="auto", help="'auto' or comma-separated layer indices to scan")
    p.add_argument("--layer-top", type=int, default=128, help="auto: layers of each domain's top-N MLP groups")
    p.add_argument("--chunk", type=int, default=16)
    p.add_argument("--ks", default="32,64,128,256,512,1024,2048")
    p.add_argument("--core-k", type=int, default=128)
    p.add_argument("--core-min", type=int, default=2)
    p.add_argument("--excl-k", type=int, default=512)
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--bench-layer", type=int, default=None, help="time one layer on one domain and exit")
    p.add_argument("--reuse", action="store_true", help="load neuron scores from --out instead of scanning")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    probe = json.loads(Path(args.probe).read_text())
    domains, seed, mlp_group = list(probe["domains"]), probe["config"]["seed"], probe["config"]["mlp_group"]
    model, tokenizer = load_model(args.model_path)
    groups = enumerate_units(model, mlp_group)
    by_key = {(u.layer, u.start, u.stop): u for u in groups}
    layers = sorted(unit_sets(groups))
    heads = [u for u in groups if u.kind == "head"]
    layer_idx = lambda name: int(name.split(".layers.")[1].split(".")[0])
    down = lambda i: next(l for l in layers if l.endswith(f".layers.{i}.mlp.down_proj"))

    if args.layers == "auto":
        wanted = set()
        for d in domains:
            mlp = sorted((r for r in probe["domains"][d]["units"] if "mlp" in r[0]), key=lambda r: -r[3])
            wanted |= {layer_idx(r[0]) for r in mlp[: args.layer_top]}
        scan_layers = sorted(wanted)
    else:
        scan_layers = [int(x) for x in args.layers.split(",")]
    log(f"scanning neurons of layers {scan_layers} ({len(scan_layers)} x 3584 per domain)")

    scan, val = {}, {}
    for d in domains:
        facts = build_facts(model, tokenizer, TEMPLATES[d], FACTS[d])
        fa, fb = split_facts(facts, seed)
        assert len(make_pairs(fa, seed).tok_clean) == probe["domains"][d]["n_scan_pairs"], "split changed"
        scan[d] = Scanner(model, make_pairs(fa, seed), layers)
        val[d] = Scanner(model, make_pairs(fb, seed + 1), layers)
        log(f"[{d}] scan pairs {len(fa)}, validation pairs {len(fb)}")

    if args.bench_layer is not None:
        d, name = domains[0], down(args.bench_layer)
        n = scan[d].clean_in[name].shape[-1]
        t0 = time.time()
        s = scan_neurons(scan[d], name, list(range(min(n, 512))), chunk=args.chunk)
        dt = time.time() - t0
        print(f"bench: {len(s)} neurons in {dt:.1f}s ({len(s) / dt:.1f}/s); full scan of {len(scan_layers)} layers x {n} "
              f"x {len(domains)} domains ~ {len(scan_layers) * n * len(domains) / (len(s) / dt) / 60:.0f} min; "
              f"max suff {max(s):.3f}")
        return

    neuron_scores: dict[str, dict[str, list[float]]] = {d: {} for d in domains}
    t0 = time.time()
    if args.reuse:
        saved = json.loads(Path(args.out).read_text())
        scan_layers, neuron_scores = saved["scan_layers"], saved["neuron_scores"]
        log(f"reusing neuron scores for layers {scan_layers}")
    else:
        for d in domains:
            for i in scan_layers:
                name = down(i)
                neuron_scores[d][name] = scan_neurons(scan[d], name, list(range(scan[d].clean_in[name].shape[-1])), args.chunk)
                log(f"[{d}] layer {i}: top neuron {max(neuron_scores[d][name]):.3f}  ({(time.time() - t0) / 60:.1f} min elapsed)")
            Path(args.out).write_text(json.dumps({"scan_layers": scan_layers, "neuron_scores": neuron_scores}))

    # unit pool: heads (group-level scan) + neurons of the scanned layers
    neurons = {name: [Unit(name, j, j + 1, "mlp") for j in range(len(sc))] for name, sc in neuron_scores[domains[0]].items()}
    pool = heads + [u for us in neurons.values() for u in us]
    scores = {}
    for d in domains:
        head_scores = [{"unit": by_key[(l, s, e)], "suff": su, "nec": ne}
                       for l, s, e, su, ne in probe["domains"][d]["units"] if by_key[(l, s, e)].kind == "head"]
        neuron = [{"unit": u, "suff": v, "nec": 0.0} for name, sc in neuron_scores[d].items()
                  for u, v in zip(neurons[name], sc)]
        scores[d] = head_scores + neuron

    families = {"mixed": scores,
                "mlp_only": {d: [x for x in scores[d] if x["unit"].kind == "mlp"] for d in domains}}

    def safe_random(pool_units, circ, rng):
        try:
            return match_random(pool_units, circ, rng)
        except ValueError:  # e.g. every full-attention head is already in the circuit
            return None

    rng = random.Random(args.seed)
    cores, cells = {}, {}
    for fam, fscores in families.items():
        core = core_units(fscores, args.core_k, args.core_min)
        cores[fam] = len(core)
        log(f"[{fam}] core (top-{args.core_k} in >= {args.core_min} domains): {len(core)} units")

        def circuit(d, k, variant, fscores=fscores, core=core):
            if variant == "full":
                return top_units(fscores[d], k)
            if variant == "no_core":
                return top_units(excluding(fscores[d], core), k)
            others = set().union(*[set(top_units(fscores[o], args.excl_k)) for o in domains if o != d])
            return top_units(excluding(fscores[d], others), k)

        fpool = pool if fam == "mixed" else [u for u in pool if u.kind == "mlp"]
        for variant in ("full", "no_core", "specific"):
            for k in ks:
                for a in domains:
                    circ = circuit(a, k, variant)
                    rand = [r for r in (safe_random(fpool, circ, rng) for _ in range(args.n_random)) if r]
                    row = {"channels": sum(u.size for u in circ), "n_heads": sum(u.kind == "head" for u in circ)}
                    for b in domains:
                        row[b] = {"suff": val[b].restoration(unit_sets(circ)), "nec": val[b].necessity(unit_sets(circ)),
                                  "rand_nec": st.mean(val[b].necessity(unit_sets(r)) for r in rand) if rand else None}
                    cells[f"{fam}|{variant}|{k}|{a}"] = row
            log(f"[{fam}] done variant {variant}")
        Path(args.out).write_text(json.dumps({"scan_layers": scan_layers, "neuron_scores": neuron_scores, "cells": cells}))
    core = cores

    jac = {}
    for k in ks:
        tops = {d: set(top_units(scores[d], k)) for d in domains}
        for i, a in enumerate(domains):
            for b in domains[i + 1:]:
                jac[f"{k}|{a}|{b}"] = len(tops[a] & tops[b]) / len(tops[a] | tops[b])
    # single-neuron agreement across domains (over scanned neurons)
    def rank(x):
        return torch.tensor(x).argsort().argsort().double()
    def sp(a, b):
        a, b = rank(a), rank(b); a, b = a - a.mean(), b - b.mean()
        return float((a * b).sum() / (a.norm() * b.norm()))
    flat = {d: [v for name in neuron_scores[d] for v in neuron_scores[d][name]] for d in domains}
    agree = {}
    for i, a in enumerate(domains):
        for b in domains[i + 1:]:
            both = [j for j in range(len(flat[a])) if flat[a][j] > 0.002 or flat[b][j] > 0.002]
            agree[f"{a}|{b}"] = {"all": sp(flat[a], flat[b]), "active": sp([flat[a][j] for j in both], [flat[b][j] for j in both]),
                                 "n_active": len(both)}
    out = {"scan_layers": scan_layers, "neuron_scores": neuron_scores, "core_size": core,
           "cells": cells, "jaccard": jac, "neuron_agreement": agree,
           "config": vars(args) | {"n_pool": len(pool), "n_heads": len(heads)}}
    Path(args.out).write_text(json.dumps(out))

    print(f"scanned layers {scan_layers}; pool {len(pool)} units ({len(heads)} heads); cores {core}")
    print("single-neuron agreement across domains (Spearman all / active, n active):",
          {k: (round(v['all'], 3), round(v['active'], 3), v['n_active']) for k, v in agree.items()})
    fmt = lambda x: "  n/a " if x is None else f"{x:+.2f}"
    for fam in families:
        for variant in ("full", "no_core", "specific"):
            print(f"\n== {fam} / {variant}: necessity own (diag) / mean of others (off) [random]; channels; heads")
            for k in ks:
                cs = []
                for a in domains:
                    row = cells[f"{fam}|{variant}|{k}|{a}"]
                    off = st.mean(row[b]["nec"] for b in domains if b != a)
                    cs.append(f"{a[:5]} {row[a]['nec']:+.2f}/{off:+.2f} [{fmt(row[a]['rand_nec'])}] ch{row['channels']} h{row['n_heads']}")
                print(f"   k={k:5d}: " + "   ".join(cs))
    print("\nJaccard of top-k circuits (mixed):", {k: round(v, 3) for k, v in jac.items() if k.split('|')[0] in (str(ks[1]), str(ks[-2]))})


if __name__ == "__main__":
    main()
