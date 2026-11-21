"""Circuit discovery by patching, joint quantization, additivity and entanglement.

Per domain (templated factual recall with entity-swap minimal pairs) the facts
are split in two halves: units (attention heads, MLP neuron groups) are scored
by single-unit patching on half A and circuits are validated on half B.  Then:

1. sufficiency / necessity of the top-k circuit vs class-matched random sets;
2. quantizing the circuit's weight columns jointly vs the same random sets
   (answer log-prob, accuracy, and generic NLL on neutral text);
3. additivity: joint damage vs the sum of single-unit damages;
4. entanglement: circuit of domain A evaluated on domain B (patching and
   quantization), with Jaccard overlaps, against matched random baselines.

    PYTHONPATH=. python scripts/circuit_probe.py --model-path Qwen/Qwen3.5-0.8B \
        --neutral outputs/eval/law.txt --out outputs/circuit_probe.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

import torch

from compress import stats as stats_mod
from compress.circuit import (KnowledgeProbe, Scanner, additivity, build_facts, dose_matched_random,
                              enumerate_units, make_pairs, match_random, scan_units, split_facts,
                              top_units, unit_sets)
from compress.models import load_model
from compress.facts import FACTS, TEMPLATES
from compress.oab import register_csv_domains

log = lambda m: print(m, file=sys.stderr, flush=True)


def summarize(xs: list[float]) -> dict:
    return {"mean": st.mean(xs), "min": min(xs), "max": max(xs), "n": len(xs)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--domains", default="capitals,elements,french")
    p.add_argument("--neutral", default=None, help="text file for the generic-NLL damage check")
    p.add_argument("--stats", default=None, help="stats file; V of the write-projection inputs gives the dose")
    p.add_argument("--no-quant", action="store_true", help="skip quantization, additivity and their entanglement cells")
    p.add_argument("--facts-csv", default=None, help="review sheet with extra cloze-fact domains")
    p.add_argument("--csv-domain", action="append", default=[], metavar="NAME=RELATION",
                   help="register RELATION of the sheet as domain NAME (then list NAME in --domains)")
    p.add_argument("--n-neutral", type=int, default=32)
    p.add_argument("--mlp-group", type=int, default=64)
    p.add_argument("--ks", default="16,32,64,128")
    p.add_argument("--bits", default="4,3,2")
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--k-ent", type=int, default=64, help="circuit size for the entanglement matrix")
    p.add_argument("--k-add", type=int, default=16, help="circuit size for the additivity test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    ks, bits_list, domains = [int(k) for k in args.ks.split(",")], [int(b) for b in args.bits.split(",")], args.domains.split(",")
    if not args.no_quant and not (args.neutral and args.stats):
        raise SystemExit("--neutral and --stats are required unless --no-quant is given")
    if args.csv_domain:
        register_csv_domains(args.facts_csv, args.csv_domain, FACTS, TEMPLATES)

    model, tokenizer = load_model(args.model_path)
    units = enumerate_units(model, args.mlp_group)
    layers = sorted(unit_sets(units))
    V, neutral = None, None
    if not args.no_quant:
        domain_stats, _ = stats_mod.load_stats(args.stats)
        V = {layer: torch.stack([dom[layer]["V"].double() for dom in domain_stats.values()]).mean(0)
             for layer in layers}
        neutral = [ln.strip() for ln in Path(args.neutral).read_text(encoding="utf-8").splitlines() if ln.strip()]
        random.Random(args.seed).shuffle(neutral)
        neutral = neutral[: args.n_neutral]
    log(f"{len(units)} units in {len(layers)} write projections; classes "
        f"{ {c: sum(u.cls == c for u in units) for c in sorted({u.cls for u in units})} }")

    out: dict = {"config": vars(args) | {"n_units": len(units)}, "domains": {}, "entanglement": {}}
    D: dict[str, dict] = {}
    for d in domains:
        facts = build_facts(model, tokenizer, TEMPLATES[d], FACTS[d])
        fa, fb = split_facts(facts, args.seed)
        log(f"[{d}] {len(facts)} known facts of equal prompt length -> scan {len(fa)}, validate {len(fb)}")
        if len(fa) < 6 or len(fb) < 6:
            raise SystemExit(f"{d}: too few known facts ({len(facts)}); the model may not know this domain")
        ba, bb = make_pairs(fa, args.seed), make_pairs(fb, args.seed + 1)
        sa, sb = Scanner(model, ba, layers), Scanner(model, bb, layers)
        scores = scan_units(sa, units, log=lambda m, d=d: log(f"[{d}] {m}"))
        probe = None if args.no_quant else KnowledgeProbe(model, tokenizer, bb.ids_clean, bb.tok_clean, neutral)
        D[d] = {"scores": scores, "sb": sb, "probe": probe}
        out["domains"][d] = {
            "n_facts": len(facts), "n_scan_pairs": len(ba.tok_clean), "n_val_pairs": len(bb.tok_clean),
            "mean_ld": {"scan": {"clean": float(sa.ld_clean.mean()), "corrupt": float(sa.ld_corrupt.mean())},
                        "val": {"clean": float(sb.ld_clean.mean()), "corrupt": float(sb.ld_corrupt.mean())}},
            "baseline_knowledge": probe.base if probe else None,
            "units": [[s["unit"].layer, s["unit"].start, s["unit"].stop, s["suff"], s["nec"]] for s in scores],
            "all_units": {"suff_val": sb.restoration(unit_sets(units)), "nec_val": sb.necessity(unit_sets(units))},
            "circuits": [], "quantization": [], "additivity": [],
        }
        rng = random.Random(args.seed)
        for key in ("suff", "nec"):
            for k in ks:
                circ = top_units(scores, k, key)
                rand = [match_random(units, circ, rng) for _ in range(args.n_random)]
                entry = {"ranked_by": key, "k": k,
                         "channels": sum(u.size for u in circ),
                         "layers_spanned": len({u.layer for u in circ}),
                         "kinds": {kind: sum(u.kind == kind for u in circ) for kind in ("head", "mlp")}}
                for split, sc in (("scan", sa), ("val", sb)):
                    s, slo, shi = sc.restoration(unit_sets(circ), args.n_boot)
                    n_, nlo, nhi = sc.necessity(unit_sets(circ), args.n_boot)
                    entry[split] = {"suff": s, "suff_ci": [slo, shi], "nec": n_, "nec_ci": [nlo, nhi],
                                    "random_suff": summarize([sc.restoration(unit_sets(r)) for r in rand]),
                                    "random_nec": summarize([sc.necessity(unit_sets(r)) for r in rand])}
                out["domains"][d]["circuits"].append(entry)
                log(f"[{d}] {key} k={k}: val suff {entry['val']['suff']:.2f} (random {entry['val']['random_suff']['mean']:.2f}), "
                    f"nec {entry['val']['nec']:.2f} (random {entry['val']['random_nec']['mean']:.2f})")
                if key == "suff" and probe is not None:  # quantization of the sufficiency-ranked circuit
                    for bits in bits_list:
                        q = {"k": k, "bits": bits, "circuit": probe.damage(unit_sets(circ), bits, V)}
                        rd = [probe.damage(unit_sets(r), bits, V) for r in rand]
                        q["random"] = {m: summarize([x[m] for x in rd]) for m in rd[0]}
                        dm = dose_matched_random(model, units, circ, V, bits, rng)
                        rdm = [probe.damage(unit_sets(r), bits, V) for r in dm]
                        q["dose_matched"] = {m: summarize([x[m] for x in rdm]) for m in rdm[0]}
                        out["domains"][d]["quantization"].append(q)
        circ = top_units(scores, args.k_add, "suff")
        for bits in ([] if probe is None else bits_list):
            ad = {"bits": bits, "circuit": additivity(probe, circ, bits),
                  "random": [additivity(probe, match_random(units, circ, rng), bits) for _ in range(3)]}
            out["domains"][d]["additivity"].append(ad)
            log(f"[{d}] additivity int{bits}: circuit joint {ad['circuit']['joint']:.3f} vs sum {ad['circuit']['sum_single']:.3f}")
        Path(args.out).write_text(json.dumps(out))
        sb.offload()  # keep only validation activations, on the CPU, for the entanglement step
        del sa, ba, bb
        torch.cuda.empty_cache()

    # entanglement: circuit of A evaluated on the validation pairs / facts of B
    rng = random.Random(args.seed + 7)
    circuits = {d: top_units(D[d]["scores"], args.k_ent, "suff") for d in domains}
    rands = {d: [match_random(units, circuits[d], rng) for _ in range(args.n_random)] for d in domains}
    ent = out["entanglement"] = {"k": args.k_ent, "domains": domains, "cells": {}, "jaccard": {}}
    for a in domains:
        for b in domains:
            sb, probe = D[b]["sb"], D[b]["probe"]
            cell = {"suff": sb.restoration(unit_sets(circuits[a])), "nec": sb.necessity(unit_sets(circuits[a])),
                    "random_nec": summarize([sb.necessity(unit_sets(r)) for r in rands[a]]),
                    "random_suff": summarize([sb.restoration(unit_sets(r)) for r in rands[a]])}
            for bits in ([] if probe is None else bits_list):
                cell[f"int{bits}"] = probe.damage(unit_sets(circuits[a]), bits, V)
                rd = [probe.damage(unit_sets(r), bits, V) for r in rands[a]]
                cell[f"random_int{bits}"] = {m: summarize([x[m] for x in rd]) for m in rd[0]}
            ent["cells"][f"{a}->{b}"] = cell
            log(f"entanglement {a}->{b}: nec {cell['nec']:.2f} (random {cell['random_nec']['mean']:.2f})")
        for b in domains:
            sa_, sb_ = set(circuits[a]), set(circuits[b])
            ent["jaccard"][f"{a}|{b}"] = len(sa_ & sb_) / len(sa_ | sb_)
    Path(args.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
