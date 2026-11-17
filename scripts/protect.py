"""Does protecting the circuit core rescue a low-bit model?

Quantize every Linear of the text tower to ``bits`` (simulated, per-row symmetric)
and keep a chosen set of write-projection unit columns at fp16:

  none      nothing protected (uniform quantization);
  core      units in the top ``--core-k`` of at least ``--core-min`` domains;
  union     union of every domain's top ``--union-k`` circuit;
  *_random  class-matched random units (same unit classes, same weight count);
  *_dose    class-matched units with the highest diagonal dose E^2 V.

Scored on every domain's held-out clean prompts (answer log-prob, top-1
accuracy) and on neutral text (NLL).  Circuit units come from the discovery
half; evaluation uses the held-out half.

    PYTHONPATH=. python scripts/protect.py --model-path Qwen/Qwen3.5-0.8B \
        --probe outputs/circuit_probe.json --stats outputs/stats_dom.pt \
        --neutral outputs/eval/law.txt --out outputs/protect.json
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
from compress.circuit import (KnowledgeProbe, build_facts, core_units, dose_of, enumerate_units,
                              make_pairs, match_random, split_facts, top_units, unit_sets)
from compress.cli import _load_model
from compress.facts import FACTS, TEMPLATES
from compress.patching import line_nll
from compress.protect import MixedQuant

log = lambda m: print(m, file=sys.stderr, flush=True)


def class_matched_top(units, ranked_units, like):
    """Highest-ranked units per size class, with the same class counts as ``like``."""
    need: dict[str, int] = {}
    for u in like:
        need[u.cls] = need.get(u.cls, 0) + 1
    chosen = []
    for u in ranked_units:
        if need.get(u.cls, 0) > 0:
            chosen.append(u)
            need[u.cls] -= 1
    return chosen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--neutral", required=True)
    p.add_argument("--n-neutral", type=int, default=32)
    p.add_argument("--bits", default="4,3,2")
    p.add_argument("--core-k", type=int, default=32)
    p.add_argument("--core-min", type=int, default=2)
    p.add_argument("--union-k", type=int, default=32)
    p.add_argument("--n-random", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    probe_json = json.loads(Path(args.probe).read_text())
    domains, seed = list(probe_json["domains"]), probe_json["config"]["seed"]
    model, tokenizer = _load_model(args.model_path)
    units = enumerate_units(model, probe_json["config"]["mlp_group"])
    by_key = {(u.layer, u.start, u.stop): u for u in units}
    domain_stats, _ = stats_mod.load_stats(args.stats)
    qlayers = sorted(next(iter(domain_stats.values())))
    V = {l: torch.stack([d[l]["V"].double() for d in domain_stats.values()]).mean(0) for l in qlayers}

    scores = {d: [{"unit": by_key[(l, s, e)], "suff": su, "nec": ne}
                  for l, s, e, su, ne in probe_json["domains"][d]["units"]] for d in domains}
    core = sorted(core_units(scores, args.core_k, args.core_min), key=lambda u: (u.layer, u.start))
    union = sorted({u for d in domains for u in top_units(scores[d], args.union_k)}, key=lambda u: (u.layer, u.start))
    sets = {"core": core, "union": union}
    log(f"core {len(core)} units, union {len(union)} units, quantizing {len(qlayers)} layers")

    neutral = [ln.strip() for ln in Path(args.neutral).read_text(encoding="utf-8").splitlines() if ln.strip()]
    random.Random(seed).shuffle(neutral)
    neutral = neutral[: args.n_neutral]
    probes = {}
    for d in domains:
        facts = build_facts(model, tokenizer, TEMPLATES[d], FACTS[d])
        _, fb = split_facts(facts, seed)
        batch = make_pairs(fb, seed + 1)
        probes[d] = KnowledgeProbe(model, tokenizer, batch.ids_clean, batch.tok_clean)
    base = {d: probes[d].stats() for d in domains}
    base_nll = sum(line_nll(model, tokenizer, neutral)) / len(neutral)
    out = {"config": vars(args) | {"core_size": len(core), "union_size": len(union), "n_layers": len(qlayers)},
           "baseline": {"domains": base, "nll": base_nll}, "conditions": []}
    log(f"baseline {({d: (round(base[d]['logp'], 2), base[d]['acc']) for d in domains})}, nll {base_nll:.3f}")

    def evaluate(name, bits, protected_units, group):
        protect = unit_sets(protected_units) if protected_units else None
        with MixedQuant(model, qlayers, bits, protect) as mq:
            res = {d: probes[d].stats() for d in domains}
            nll = sum(line_nll(model, tokenizer, neutral)) / len(neutral)
        entry = {"name": name, "group": group, "bits": bits, "n_units": len(protected_units),
                 "protected_fraction": mq.protected_weights / mq.total_weights,
                 "dlogp": {d: base[d]["logp"] - res[d]["logp"] for d in domains},
                 "dacc": {d: base[d]["acc"] - res[d]["acc"] for d in domains},
                 "dnll": nll - base_nll}
        out["conditions"].append(entry)
        return entry

    rng = random.Random(args.seed)
    for bits in [int(b) for b in args.bits.split(",")]:
        none = evaluate("none", bits, [], "none")
        log(f"int{bits} none: dacc {({d: round(v, 2) for d, v in none['dacc'].items()})} dnll {none['dnll']:+.3f}")
        dose = {u: dose_of(model, [u], bits, V) for u in units}
        ranked = sorted(units, key=lambda u: -dose[u])
        for label, chosen in sets.items():
            e = evaluate(label, bits, chosen, label)
            rs = [evaluate(f"{label}_random{i}", bits, match_random(units, chosen, rng), f"{label}_random")
                  for i in range(args.n_random)]
            dd = evaluate(f"{label}_dose", bits, class_matched_top(units, ranked, chosen), f"{label}_dose")
            log(f"int{bits} {label}({len(chosen)}): dacc {({d: round(v, 2) for d, v in e['dacc'].items()})} "
                f"dnll {e['dnll']:+.3f} | random dacc {st.mean(st.mean(r['dacc'].values()) for r in rs):+.2f} "
                f"dnll {st.mean(r['dnll'] for r in rs):+.3f} | dose-top dacc {st.mean(dd['dacc'].values()):+.2f} dnll {dd['dnll']:+.3f}")
        Path(args.out).write_text(json.dumps(out))
    Path(args.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
