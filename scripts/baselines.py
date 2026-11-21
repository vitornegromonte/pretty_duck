"""Real baselines for circuit-guided protection: GPTQ and global outlier-column protection.

Everything is int``--bits`` group-wise (``--group``), evaluated on the same seeded OAB questions as
``protect_law.py`` (per-question results are stored for paired comparisons).  Weight budgets are equal:
every protected set has the circuit's number of fp16 weights.

Rounding:  ``rtn`` (round to nearest) and ``gptq`` (sequential GPTQ, Hessians from a mixed-domain
calibration set).  Protected sets, each combined with either rounding:

  circuit      top-k write-projection units by sufficiency on the law facts;
  dose         class-matched units with the highest quantization dose E^2 V;
  awq          write-projection columns with the highest mean |activation|;
  outlier_all  columns of *any* layer with the highest dose per weight (knapsack by density: a stronger,
               domain-blind outlier-protection baseline that is free to pick any layer);
  random       class-matched random units.

    PYTHONPATH=. python scripts/baselines.py --model-path Qwen/Qwen3.5-4B \
        --probe outputs/circuit_probe_4b_all2.json --stats outputs/stats_4b.pt \
        --oab outputs/data/oab_exams/oab_exams.jsonl --calib-dir outputs/calib2 --out outputs/baselines_4b.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from compress import stats as stats_mod
from compress.circuit import (KnowledgeProbe, build_facts, enumerate_units, make_pairs, match_random,
                              split_facts, top_units, unit_sets)
from compress.cli import _load_model
from compress.evaluate import evaluate_mmlu_rows
from compress.facts import FACTS, TEMPLATES
from compress.gptq import GPTQQuant, gptq_quantize
from compress.oab import load_oab, register_csv_domains
from compress.protect import MixedQuant
from compress.provenance import provenance
from compress.quant_probe import fake_quant_group

log = lambda m: print(m, file=sys.stderr, flush=True)
DOMAINS = ("bio", "cs", "hum", "law", "math")


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


def calib_batches(tokenizer, directory: str, n: int, max_len: int, batch: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    lines: list[str] = []
    for d in DOMAINS:
        ls = [ln.strip() for ln in (Path(directory) / f"{d}.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
        rng.shuffle(ls)
        lines += ls[: n // len(DOMAINS)]
    rng.shuffle(lines)
    return [tokenizer(lines[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            for i in range(0, len(lines), batch)]


def outlier_columns(modules, qlayers, V, bits, group, budget_weights) -> dict[str, torch.Tensor]:
    """Columns of any layer with the highest dose per weight, until ``budget_weights`` are protected."""
    dens, owner, col, rows = [], [], [], []
    for li, name in enumerate(qlayers):
        w = modules[name].weight.data.float()
        err = (w - fake_quant_group(w, bits, group)).double().square().sum(0)  # per input column
        d = err * V[name].to(err.device) / w.shape[0]
        dens.append(d.cpu())
        owner.append(torch.full((d.numel(),), li))
        col.append(torch.arange(d.numel()))
        rows.append(torch.full((d.numel(),), w.shape[0]))
    dens, owner, col, rows = map(torch.cat, (dens, owner, col, rows))
    order = torch.argsort(dens, descending=True)
    take = order[: int(torch.searchsorted(torch.cumsum(rows[order], 0), budget_weights, right=True)) + 1]
    out: dict[str, torch.Tensor] = {}
    for li, name in enumerate(qlayers):
        sel = col[take][owner[take] == li]
        if sel.numel():
            out[name] = torch.sort(sel).values
    return out


def reference_quantize(W, H, bits, group, protect):
    """The authors' GPTQ code (third_party/gptq_ref) on the same grid and Hessian as ``gptq_quantize``."""
    if protect is not None and protect.numel():
        raise ValueError("the reference implementation has no protected columns")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "gptq_ref"))
    from gptq import GPTQ

    class SymGrid:  # round(w/s) in [-2^(b-1), 2^(b-1)-1] with s = absmax / (2^(b-1)-1), as fake_quant_group
        maxq = torch.tensor(2 ** bits - 1)
        zero = torch.tensor(float(2 ** (bits - 1)))

        def ready(self):
            return False

        def find_params(self, x, weight=True):
            self.scale = (x.abs().amax(dim=1, keepdim=True) / (2 ** (bits - 1) - 1)).clamp_min(1e-12)
            self.zero = torch.full_like(self.scale, float(2 ** (bits - 1)))
            self.maxq = self.maxq.to(x.device)

    layer = torch.nn.Linear(W.shape[1], W.shape[0], bias=False, device=W.device)
    layer.weight.data = W.float().clone()
    g = GPTQ(layer)
    g.quantizer = SymGrid()
    g.H = 2 * H.to(device=W.device, dtype=torch.float32)  # the reference scales H by 2/n
    g.fasterquant(blocksize=128, percdamp=0.01, groupsize=group)
    return layer.weight.data.float()


def neutral_nll(model, tokenizer, path, n=48, max_len=256, batch=8) -> float:
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if len(ln.strip()) > 80][:n]
    dev, tot, cnt = next(model.parameters()).device, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(lines), batch):
            b = tokenizer(lines[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(dev)
            logits = model(**b).logits[:, :-1].float()
            tgt, m = b["input_ids"][:, 1:], b["attention_mask"][:, 1:].bool()
            nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
            tot += float(nll[m].sum())
            cnt += int(m.sum())
    return tot / cnt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--oab", required=True)
    p.add_argument("--calib-dir", required=True)
    p.add_argument("--n-calib", type=int, default=160)
    p.add_argument("--calib-len", type=int, default=512)
    p.add_argument("--domain", default="direito_siglas")
    p.add_argument("--n-questions", type=int, default=600)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--ks", default="32,128")
    p.add_argument("--n-random", type=int, default=3)
    p.add_argument("--roundings", default="gptq,rtn")
    p.add_argument("--sys-prompt", default="As questões a seguir são de provas da OAB. Responda com a letra da alternativa correta.")
    p.add_argument("--keep-blocks", action="store_true", help="RTN with protected columns keeps the original 64-column blocks")
    p.add_argument("--only-uniform", action="store_true", help="only the unprotected conditions (rounding cross-checks)")
    p.add_argument("--neutral", default=None, help="text file; adds the mean token NLL on 48 of its lines to every row")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    ks, roundings = [int(k) for k in args.ks.split(",")], args.roundings.split(",")

    probe = json.loads(Path(args.probe).read_text())
    if probe["config"].get("csv_domain"):
        register_csv_domains(probe["config"]["facts_csv"], probe["config"]["csv_domain"], FACTS, TEMPLATES)
    seed = probe["config"]["seed"]
    model, tokenizer = _load_model(args.model_path)
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
    batch = make_pairs(fb, seed + 1)
    kprobe = KnowledgeProbe(model, tokenizer, batch.ids_clean, batch.tok_clean)
    rows = load_oab(args.oab)
    random.Random(seed).shuffle(rows)
    rows = rows[: args.n_questions]
    calib = calib_batches(tokenizer, args.calib_dir, args.n_calib, args.calib_len, 4, seed)
    log(f"{len(units)} units, {len(qlayers)} layers, {len(rows)} questions, {len(calib)} calibration batches")

    def measure() -> dict:
        res = evaluate_mmlu_rows(model, tokenizer, rows, sys_prompt=args.sys_prompt, prompt_style="plain")
        pr = res["predictions"]
        k = kprobe.stats()
        return {"oab_acc": res["accuracy"], "oab_nll": res["nll_per_token"],
                "oab_correct": [int(x["predicted"] == x["expected"]) for x in pr], "oab_nll_q": [x["nll"] for x in pr],
                "law_logp": k["logp"], "law_acc": k["acc"],
                **({"neutral_nll": neutral_nll(model, tokenizer, args.neutral)} if args.neutral else {})}

    out = {"config": vars(args) | {"write_weights": write_weights}, "baseline": measure(), "conditions": [],
           "provenance": provenance()}
    log(f"fp16: OAB {out['baseline']['oab_acc']:.1f}% nll {out['baseline']['oab_nll']:.3f}")

    def evaluate(name, group, rounding, columns=None, n_units=0):
        if rounding in ("gptq", "gptq_ref"):
            ctx = GPTQQuant(model, calib, qlayers, args.bits, columns, args.group,
                            reference_quantize if rounding == "gptq_ref" else gptq_quantize)
        else:
            ctx = MixedQuant(model, qlayers, args.bits, columns, args.group, offload=True, keep_blocks=args.keep_blocks)
        with ctx as q:
            m = measure()
        entry = {"name": f"{rounding}_{name}", "group": group, "rounding": rounding, "bits": args.bits,
                 "n_units": n_units, "protected_weights": q.protected_weights,
                 "protected_fraction": q.protected_weights / q.total_weights, **m}
        out["conditions"].append(entry)
        log(f"{entry['name']}: OAB {m['oab_acc']:.1f}% nll {m['oab_nll']:.3f} law logp {m['law_logp']:.2f} "
            f"(protected {100 * entry['protected_fraction']:.2f}%)")
        Path(args.out).write_text(json.dumps(out))

    def unit_dose(u):
        w = modules[u.layer].weight.data[:, u.start:u.stop].float()
        err = (w - fake_quant_group(w, args.bits, args.group)).double().square()
        return float((err * V[u.layer][u.start:u.stop].to(err.device)).sum())

    dose = {u: unit_dose(u) for u in units}
    ranked = sorted(units, key=lambda u: -dose[u])
    rng = random.Random(args.seed)
    plans: list[tuple[str, str, dict | None, int]] = [("uniform", "uniform", None, 0)]
    out["overlap"] = {}
    for k in ([] if args.only_uniform else ks):
        circ = top_units(scores, k)
        circ_cols = unit_sets(circ)
        budget = sum(modules[l].weight.shape[0] * c.numel() for l, c in circ_cols.items())
        frac = budget / write_weights
        cols_awq = {l: torch.sort(A[l].topk(max(1, round(frac * A[l].numel()))).indices).values for l in write_layers}
        cols_out = outlier_columns(modules, qlayers, V, args.bits, args.group, budget)
        in_out = sum(int(torch.isin(c, cols_out.get(l, torch.tensor([], dtype=torch.long))).sum()) for l, c in circ_cols.items())
        in_awq = sum(int(torch.isin(c, cols_awq[l]).sum()) for l, c in circ_cols.items())
        n_cols = sum(c.numel() for c in circ_cols.values())
        out["overlap"][f"k{k}"] = {"circuit_columns": n_cols, "in_outlier_all": in_out, "in_awq": in_awq,
                                   "budget_weights": budget}
        log(f"k={k}: {n_cols} circuit columns, {in_out} also outlier_all, {in_awq} also awq")
        plans += [(f"circuit_k{k}", "circuit", circ_cols, len(circ)),
                  (f"outlier_all_k{k}", "outlier_all", cols_out, 0),
                  (f"dose_k{k}", "dose", unit_sets(class_matched_top(ranked, circ)), len(circ)),
                  (f"awq_k{k}", "awq", cols_awq, 0)]
        plans += [(f"random{i}_k{k}", "random", unit_sets(match_random(units, circ, rng)), len(circ))
                  for i in range(args.n_random)]
    for rounding in roundings:
        for name, group, columns, n_units in plans:
            evaluate(name, group, rounding, columns, n_units)
    Path(args.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
