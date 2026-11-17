"""End-to-end: where should extra precision bytes go, at equal actual bytes?

Baseline is uniform ``nf4`` over the text tower's Linears.  Each method spends
the same extra byte budget (a fraction of the uniform-nf4 size) on upgrades,
chosen by a different signal:

  layers_D          whole-layer nf4 -> int8/fp16 upgrades, greedy on the diagonal
                    error reduction per byte (mean D over the calibration domains);
  layers_measured   same greedy on the *measured* single-layer NLL damage table;
  units_circuit     fp16-protected unit columns, ranked by mean single-unit
                    sufficiency over the knowledge domains;
  units_dose        fp16-protected unit columns, ranked by nf4 dose per extra byte;
  units_random      fp16-protected unit columns in random order (5 draws).

Actual bytes come from the exact formulas (``plan_nbytes``).  Scored on every
knowledge domain's held-out clean prompts and on held-out text NLL (lines that
neither the measured table nor the calibration used).

    PYTHONPATH=. python scripts/alloc_e2e.py --model-path Qwen/Qwen3.5-0.8B \
        --probe outputs/circuit_probe.json --errors outputs/errors_dom.json \
        --sens outputs/sensitivity.json --stats outputs/stats_dom.pt --out outputs/alloc_e2e.json
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
from compress.circuit import KnowledgeProbe, build_facts, enumerate_units, make_pairs, split_facts
from compress.cli import _load_model
from compress.facts import FACTS, TEMPLATES
from compress.patching import line_nll
from compress.protect import FormatPlan, plan_nbytes
from compress.quant import quantize_weight, storage_nbytes

log = lambda m: print(m, file=sys.stderr, flush=True)
RANK = {"nf4": 0, "int8": 1, "fp16": 2}


def greedy_layers(layers, shapes, benefit, budget):
    """Upgrade layers nf4 -> int8/fp16 by best benefit per extra byte until the budget is spent."""
    fmt = {l: "nf4" for l in layers}
    nbytes = lambda l, f: storage_nbytes(shapes[l], f)
    remaining = budget
    while True:
        best = None
        for l in layers:
            for new in ("int8", "fp16"):
                if RANK[new] <= RANK[fmt[l]]:
                    continue
                cost = nbytes(l, new) - nbytes(l, fmt[l])
                gain = benefit(l, fmt[l], new)
                if cost <= remaining and gain > 0 and (best is None or gain / cost > best[0]):
                    best = (gain / cost, l, new, cost)
        if best is None:
            return {l: f for l, f in fmt.items() if f != "nf4"}
        _, l, new, cost = best
        fmt[l] = new
        remaining -= cost


def fill_units(order, extra, budget):
    chosen, used = [], 0
    for u in order:
        if used + extra[u] <= budget:
            chosen.append(u)
            used += extra[u]
    return chosen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--errors", required=True)
    p.add_argument("--sens", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--eval-dir", default="outputs/eval")
    p.add_argument("--text-domains", default="math,law,bio,hum")
    p.add_argument("--budgets", default="0.01,0.02,0.05,0.10")
    p.add_argument("--n-random", type=int, default=5)
    p.add_argument("--n-text", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    probe_json = json.loads(Path(args.probe).read_text())
    kdoms, seed = list(probe_json["domains"]), probe_json["config"]["seed"]
    errors = json.loads(Path(args.errors).read_text())["layers"]
    sens = json.loads(Path(args.sens).read_text())["table"]["layers"]
    domain_stats, _ = stats_mod.load_stats(args.stats)
    qlayers = sorted(next(iter(domain_stats.values())))
    V = {l: torch.stack([d[l]["V"].double() for d in domain_stats.values()]).mean(0) for l in qlayers}

    model, tokenizer = _load_model(args.model_path)
    modules = dict(model.named_modules())
    shapes = {l: tuple(modules[l].weight.shape) for l in qlayers}
    units = enumerate_units(model, probe_json["config"]["mlp_group"])
    by_key = {(u.layer, u.start, u.stop): u for u in units}
    b0 = plan_nbytes(shapes)

    # signals ---------------------------------------------------------------
    D = {l: {f: st.mean(errors[l]["error"][d][f] for d in errors[l]["error"]) for f in ("nf4", "int8", "fp16")}
         for l in qlayers}
    meas = {l: {"nf4": st.mean(sens[l]["nf4"][d]["delta"] for d in sens[l]["nf4"]),
                "int8": st.mean(sens[l]["int8"][d]["delta"] for d in sens[l]["int8"]), "fp16": 0.0}
            for l in qlayers}
    mean_suff: dict = {}
    for d in kdoms:
        for l, s, e, suff, _ in probe_json["domains"][d]["units"]:
            mean_suff.setdefault(by_key[(l, s, e)], []).append(suff)
    mean_suff = {u: st.mean(v) for u, v in mean_suff.items()}
    extra = {u: shapes[u.layer][0] * u.size * 2 - storage_nbytes((shapes[u.layer][0], u.size), "nf4") for u in units}

    @torch.no_grad()
    def nf4_dose(u):
        w = modules[u.layer].weight.data[:, u.start:u.stop].float()
        err = (w - quantize_weight(w, "nf4").dequant).double().square()
        return float((err * V[u.layer][u.start:u.stop].to(err.device)).sum())

    dose = {u: nf4_dose(u) for u in units}
    order_circuit = sorted(units, key=lambda u: -mean_suff[u])
    order_dose = sorted(units, key=lambda u: -dose[u] / extra[u])

    # evaluation ------------------------------------------------------------
    probes = {}
    for d in kdoms:
        facts = build_facts(model, tokenizer, TEMPLATES[d], FACTS[d])
        _, fb = split_facts(facts, seed)
        b = make_pairs(fb, seed + 1)
        probes[d] = KnowledgeProbe(model, tokenizer, b.ids_clean, b.tok_clean)
    text = {}
    for d in args.text_domains.split(","):
        lines = [ln.strip() for ln in (Path(args.eval_dir) / f"{d}.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
        random.Random(seed).shuffle(lines)
        text[d] = lines[64:64 + args.n_text]  # disjoint from the measured table's lines [0:64)
    nll = lambda: {d: sum(line_nll(model, tokenizer, ls)) / len(ls) for d, ls in text.items()}
    base = {d: probes[d].stats() for d in kdoms}
    base_nll = nll()
    out = {"config": vars(args) | {"uniform_nf4_bytes": b0, "n_layers": len(qlayers)},
           "baseline": {"domains": base, "nll": base_nll}, "conditions": []}

    def evaluate(name, group, budget_frac, formats=None, protect=None):
        nbytes = plan_nbytes(shapes, formats, protect)
        with FormatPlan(model, qlayers, formats, protect):
            res = {d: probes[d].stats() for d in kdoms}
            tn = nll()
        e = {"name": name, "group": group, "budget": budget_frac, "bytes": nbytes, "extra_pct": 100 * (nbytes / b0 - 1),
             "dlogp": {d: base[d]["logp"] - res[d]["logp"] for d in kdoms},
             "dacc": {d: base[d]["acc"] - res[d]["acc"] for d in kdoms},
             "dnll": {d: tn[d] - base_nll[d] for d in tn}}
        out["conditions"].append(e)
        return e

    def line(e):
        return (f"{e['name']:22s} +{e['extra_pct']:5.2f}%  dlogp {st.mean(e['dlogp'].values()):+.3f}  "
                f"dacc {st.mean(e['dacc'].values()):.2f}  dnll {st.mean(e['dnll'].values()):+.4f}")

    log(line(evaluate("uniform_nf4", "uniform", 0.0)))
    log(line(evaluate("uniform_int8", "uniform", None, {l: "int8" for l in qlayers})))
    for frac in [float(b) for b in args.budgets.split(",")]:
        budget = frac * b0
        fd = greedy_layers(qlayers, shapes, lambda l, a, b: D[l][a] - D[l][b], budget)
        fm = greedy_layers(qlayers, shapes, lambda l, a, b: meas[l][a] - meas[l][b], budget)
        log(line(evaluate(f"layers_D@{frac}", "layers_D", frac, fd)) + f"  ({len(fd)} layers upgraded)")
        log(line(evaluate(f"layers_measured@{frac}", "layers_measured", frac, fm)) + f"  ({len(fm)} layers upgraded)")
        for group, order in (("units_circuit", order_circuit), ("units_dose", order_dose)):
            chosen = fill_units(order, extra, budget)
            prot = {}
            for u in chosen:
                prot.setdefault(u.layer, []).extend(range(u.start, u.stop))
            prot = {l: torch.tensor(sorted(c)) for l, c in prot.items()}
            log(line(evaluate(f"{group}@{frac}", group, frac, None, prot)) + f"  ({len(chosen)} units)")
        for i in range(args.n_random):
            order = units[:]
            random.Random(args.seed + i).shuffle(order)
            prot = {}
            for u in fill_units(order, extra, budget):
                prot.setdefault(u.layer, []).extend(range(u.start, u.stop))
            evaluate(f"units_random{i}@{frac}", "units_random", frac, None, {l: torch.tensor(sorted(c)) for l, c in prot.items()})
        Path(args.out).write_text(json.dumps(out))
    Path(args.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
