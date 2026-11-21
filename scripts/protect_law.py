"""Does protecting the Brazilian-law circuit preserve OAB accuracy under low-bit quantization?

Every Linear of the text tower is fake-quantized group-wise (one scale per ``--group`` columns)
at ``--bits``; a set of write-projection unit columns stays at full precision.  Protected sets
(all with the same number of weights as the circuit, except the per-layer AWQ-like set):

  circuit   top-k units by single-unit sufficiency on the law facts (scan half);
  dose      class-matched units with the highest quantization dose E^2 V;
  awq       the columns of every write projection with the highest mean |activation|
            (fraction matched to the circuit's share of weights);
  random    class-matched random units (``--n-random`` draws).

``--controls capitals,elements,french`` instead protects the circuits of those domains (same sizes) to test
whether the gain is specific to the law circuit.

Metrics: OAB accuracy and NLL of the correct letter on the same ``--n-questions`` questions in
every condition (per-question values are stored for paired comparisons), and the log-probability
of held-out sigla facts.  Bits are chosen automatically where uniform quantization hurts without
collapsing (accuracy >= 30%, NLL up by >= 0.03).

    PYTHONPATH=. python scripts/protect_law.py --model-path Qwen/Qwen3.5-4B \
        --probe outputs/circuit_probe_4b_all.json --stats outputs/stats_4b.pt \
        --oab outputs/data/oab_exams/oab_exams.jsonl --out outputs/protect_law_4b.json
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
from compress.circuit import (KnowledgeProbe, build_facts, enumerate_units, make_pairs, match_random,
                              split_facts, top_units, unit_sets)
from compress.models import load_model
from compress.evaluate import evaluate_mmlu_rows
from compress.facts import FACTS, TEMPLATES
from compress.oab import load_mmmlu, load_oab, register_csv_domains
from compress.protect import MixedQuant
from compress.provenance import provenance
from compress.quant_probe import fake_quant_group

log = lambda m: print(m, file=sys.stderr, flush=True)


def class_matched_top(ranked_units, like):
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
    p.add_argument("--oab", default=None, help="OAB jsonl (benchmark oab)")
    p.add_argument("--benchmark", choices=["oab", "mmmlu"], default="oab")
    p.add_argument("--mmmlu-csv", default=None, help="openai/MMMLU mmlu_PT-BR.csv (benchmark mmmlu)")
    p.add_argument("--n-law", type=int, default=500)
    p.add_argument("--n-other", type=int, default=500)
    p.add_argument("--with-standard", action="store_true", help="with --controls, also run the standard comparison")
    p.add_argument("--domain", default="direito_siglas")
    p.add_argument("--n-questions", type=int, default=600)
    p.add_argument("--bits", default="4,3,2")
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--ks", default="32,128")
    p.add_argument("--n-random", type=int, default=3)
    p.add_argument("--controls", default="", help="comma-separated other domains whose circuits are protected "
                   "instead of the law circuit (specificity control); skips the standard comparison")
    p.add_argument("--sys-prompt", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    bits_list, ks = [int(b) for b in args.bits.split(",")], [int(k) for k in args.ks.split(",")]

    probe = json.loads(Path(args.probe).read_text())
    if probe["config"].get("csv_domain"):
        register_csv_domains(probe["config"]["facts_csv"], probe["config"]["csv_domain"], FACTS, TEMPLATES)
    seed = probe["config"]["seed"]
    model, tokenizer = load_model(args.model_path)
    modules = dict(model.named_modules())
    units = enumerate_units(model, probe["config"]["mlp_group"])
    by_key = {(u.layer, u.start, u.stop): u for u in units}
    scores = [{"unit": by_key[(l, s, e)], "suff": su, "nec": ne} for l, s, e, su, ne in probe["domains"][args.domain]["units"]]
    domain_stats, _ = stats_mod.load_stats(args.stats)
    qlayers = sorted(next(iter(domain_stats.values())))
    mean = lambda key: {l: torch.stack([d[l][key].double() for d in domain_stats.values()]).mean(0) for l in qlayers}
    V, A = mean("V"), mean("A")
    write_layers = sorted(unit_sets(units))
    write_weights = sum(modules[l].weight.numel() for l in write_layers)

    facts = build_facts(model, tokenizer, TEMPLATES[args.domain], FACTS[args.domain])
    fa, fb = split_facts(facts, seed)
    assert len(make_pairs(fa, seed).tok_clean) == probe["domains"][args.domain]["n_scan_pairs"], "fact split changed"
    batch = make_pairs(fb, seed + 1)
    kprobe = KnowledgeProbe(model, tokenizer, batch.ids_clean, batch.tok_clean)
    if args.benchmark == "mmmlu":
        rows = load_mmmlu(args.mmmlu_csv, n_law=args.n_law, n_other=args.n_other, seed=seed)
        sys_prompt = args.sys_prompt or "As questões a seguir são de múltipla escolha. Responda com a letra da alternativa correta."
    else:
        rows = load_oab(args.oab)
        random.Random(seed).shuffle(rows)
        rows = rows[: args.n_questions]
        sys_prompt = args.sys_prompt or "As questões a seguir são de provas da OAB. Responda com a letra da alternativa correta."
    log(f"{len(units)} units, {len(qlayers)} quantized layers, {len(rows)} OAB questions, {len(batch.tok_clean)} validation facts")

    def measure() -> dict:
        res = evaluate_mmlu_rows(model, tokenizer, rows, sys_prompt=sys_prompt, prompt_style="plain")
        pr = res["predictions"]
        k = kprobe.stats()
        return {"oab_acc": res["accuracy"], "oab_nll": res["nll_per_token"],
                "oab_correct": [int(x["predicted"] == x["expected"]) for x in pr], "oab_nll_q": [x["nll"] for x in pr],
                "law_logp": k["logp"], "law_acc": k["acc"]}

    out = {"config": vars(args) | {"n_val_facts": len(batch.tok_clean), "write_weights": write_weights}, "baseline": measure(), "conditions": [], "provenance": provenance(),
           "rows_area": [r.get("area", "oab") for r in rows]}
    log(f"fp16 baseline: OAB {out['baseline']['oab_acc']:.1f}% nll {out['baseline']['oab_nll']:.3f}, law logp {out['baseline']['law_logp']:.2f}")

    def evaluate(name, group, bits, units_protected=None, columns=None):
        protect = columns if columns is not None else (unit_sets(units_protected) if units_protected else None)
        with MixedQuant(model, qlayers, bits, protect, args.group) as mq:
            m = measure()
        entry = {"name": name, "group": group, "bits": bits, "n_units": len(units_protected or []),
                 "protected_weights": mq.protected_weights, "protected_fraction": mq.protected_weights / mq.total_weights, **m}
        out["conditions"].append(entry)
        log(f"int{bits} {name}: OAB {m['oab_acc']:.1f}% nll {m['oab_nll']:.3f} law logp {m['law_logp']:.2f} (protected {100 * entry['protected_fraction']:.2f}% of weights)")
        Path(args.out).write_text(json.dumps(out))
        return entry

    def run_controls() -> None:
        for bits in bits_list:
            for d in args.controls.split(","):
                sc = [{"unit": by_key[(l, s_, e)], "suff": su, "nec": ne} for l, s_, e, su, ne in probe["domains"][d]["units"]]
                for k in ks:
                    evaluate(f"ctrl_{d}_k{k}", f"ctrl_{d}", bits, top_units(sc, k))

    if args.controls and not args.with_standard:
        run_controls()
        Path(args.out).write_text(json.dumps(out))
        return

    uniform = {b: evaluate(f"uniform_int{b}", "uniform", b) for b in bits_list}
    base_nll = out["baseline"]["oab_nll"]
    chosen = [b for b in bits_list if uniform[b]["oab_acc"] >= 30 and uniform[b]["oab_nll"] - base_nll >= 0.03]
    log(f"bits with informative damage: {chosen or 'none (running all)'}")
    rng = random.Random(args.seed)
    for bits in chosen or bits_list:
        def unit_dose(u):
            w = modules[u.layer].weight.data[:, u.start:u.stop].float()
            err = (w - fake_quant_group(w, bits, args.group)).double().square()
            return float((err * V[u.layer][u.start:u.stop].to(err.device)).sum())
        dose = {u: unit_dose(u) for u in units}
        ranked = sorted(units, key=lambda u: -dose[u])
        for k in ks:
            circ = top_units(scores, k)
            frac = sum(modules[u.layer].weight.shape[0] * u.size for u in circ) / write_weights
            evaluate(f"circuit_k{k}", "circuit", bits, circ)
            evaluate(f"dose_k{k}", "dose", bits, class_matched_top(ranked, circ))
            cols = {l: torch.sort(A[l].topk(max(1, round(frac * A[l].numel()))).indices).values for l in write_layers}
            evaluate(f"awq_k{k}", "awq", bits, columns=cols)
            for i in range(args.n_random):
                evaluate(f"random{i}_k{k}", "random", bits, match_random(units, circ, rng))
    if args.controls:
        run_controls()
    Path(args.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
