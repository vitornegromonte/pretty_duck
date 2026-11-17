"""Changes to the quantization algorithm itself, on Qwen3.5-4B: grid, low-bit regime, calibration, soft weights, allocation, recovery.

Every condition is a (rounding, bits, options) spec evaluated on the seeded OAB questions (600) and, for
the domain-specificity suites, on Portuguese MMMLU (500 law + 500 other); per-question results are stored.
Suites (``--suites``), all at int3 group-wise unless stated (int4 headroom disappears under GPTQ):

  grid     rtn / gptq / +clip / +asym / +act-order / group 32           (how good can the plain quantizer get)
  cal      GPTQ Hessian from (1-a) generic + a domain calibration text   (domain-aware calibration; a=0.5, 1;
           domains: OAB questions held out of the evaluation, Portuguese law and non-law MMMLU outside the
           evaluation sample = the controls)
  protect  fp16 columns under GPTQ: circuit / dose / outlier_all / random (equal weights)
  soft     circuit columns get weight 1+beta in the GPTQ objective instead of fp16 (zero extra bytes)
  mixed    {3,4}-bit per layer at 3.5 bits average: greedy by measured layer damage (law lines / generic
           lines) vs random layer subsets
  lowrank  fp16 rank-r correction of the rounding error whitened by the generic / OAB / non-law Hessian

Results are appended to ``--out`` after every condition and skipped on restart.

    PYTHONPATH=. python scripts/quant_suite.py --model-path Qwen/Qwen3.5-4B --probe outputs/circuit_probe_4b_all2.json \
        --stats outputs/stats_4b.pt --oab outputs/data/oab_exams/oab_exams.jsonl --mmmlu-csv outputs/data/mmmlu/mmlu_PT-BR.csv \
        --calib-dir outputs/calib2 --neutral outputs/eval/hum.txt --out outputs/quant_suite_4b.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines as B  # noqa: E402

from compress import stats as stats_mod  # noqa: E402
from compress.circuit import (KnowledgeProbe, build_facts, enumerate_units, make_pairs, match_random,  # noqa: E402
                              split_facts, top_units, unit_sets)
from compress.cli import _load_model  # noqa: E402
from compress.evaluate import evaluate_mmlu_rows  # noqa: E402
from compress.facts import FACTS, TEMPLATES  # noqa: E402
from compress.gptq import GPTQQuant  # noqa: E402
from compress.oab import load_mmmlu, load_oab, register_csv_domains  # noqa: E402
from compress.protect import MixedQuant  # noqa: E402
from compress.quant_probe import fake_quant_group  # noqa: E402

log = lambda m: print(m, file=sys.stderr, flush=True)
OAB_PROMPT = "As questões a seguir são de provas da OAB. Responda com a letra da alternativa correta."
MM_PROMPT = "As questões a seguir são de múltipla escolha. Responda com a letra da alternativa correta."


def fmt_row(r: dict) -> str:
    return r["question"].strip() + "\n" + "\n".join(f"{l}) {c}" for l, c in zip("ABCD", r["choices"]))


def tok_batches(tokenizer, lines, n, max_len=512, batch=4):
    lines = lines[:n]
    return [tokenizer(lines[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            for i in range(0, len(lines), batch)]


def lines_nll(model, tokenizer, lines, max_len=256, batch=8) -> float:
    dev, tot, cnt = next(model.parameters()).device, 0.0, 0
    with torch.no_grad():
        for i in range(0, len(lines), batch):
            b = tokenizer(lines[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(dev)
            logits = model(**b).logits[:, :-1].float()
            nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), b["input_ids"][:, 1:], reduction="none")
            m = b["attention_mask"][:, 1:].bool()
            tot += float(nll[m].sum())
            cnt += int(m.sum())
    return tot / cnt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--oab", required=True)
    p.add_argument("--mmmlu-csv", required=True)
    p.add_argument("--calib-dir", required=True)
    p.add_argument("--neutral", required=True)
    p.add_argument("--suites", default="grid,cal,protect,soft,mixed,lowrank")
    p.add_argument("--only", default="", help="comma-separated condition names (default all of the suites)")
    p.add_argument("--domain", default="direito_siglas")
    p.add_argument("--n-questions", type=int, default=600)
    p.add_argument("--n-mmmlu", type=int, default=500)
    p.add_argument("--n-calib", type=int, default=160)
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--k", type=int, default=128, help="circuit size for protect / soft")
    p.add_argument("--n-random", type=int, default=2)
    p.add_argument("--reps", type=int, default=3, help="calibration-sample replicates for the protect suite (rounding noise)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

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
    layer_weights = {l: modules[l].weight.numel() for l in qlayers}

    facts = build_facts(model, tokenizer, TEMPLATES[args.domain], FACTS[args.domain])
    fa, fb = split_facts(facts, seed)
    batch = make_pairs(fb, seed + 1)
    kprobe = KnowledgeProbe(model, tokenizer, batch.ids_clean, batch.tok_clean)

    # evaluation rows and disjoint calibration pools
    oab_all = load_oab(args.oab)
    random.Random(seed).shuffle(oab_all)
    oab_rows, oab_pool = oab_all[: args.n_questions], oab_all[args.n_questions:]
    mm_rows = load_mmmlu(args.mmmlu_csv, n_law=args.n_mmmlu, n_other=args.n_mmmlu, seed=seed)
    mm_all = load_mmmlu(args.mmmlu_csv, n_law=10 ** 6, n_other=10 ** 6, seed=seed)
    mm_law_pool = [r for r in mm_all if r["area"] == "law"][args.n_mmmlu:]
    mm_other_pool = [r for r in mm_all if r["area"] == "other"][args.n_mmmlu:]
    gen_cache: dict[int, list] = {}

    def gen_batches_for(rep: int):  # replicate r draws a different calibration sample
        if rep not in gen_cache:
            gen_cache[rep] = B.calib_batches(tokenizer, args.calib_dir, args.n_calib, 512, 4, seed + rep)
        return gen_cache[rep]

    cal_lines = {"oab": [fmt_row(r) for r in oab_pool], "mmlaw": [fmt_row(r) for r in mm_law_pool],
                 "mmother": [fmt_row(r) for r in mm_other_pool]}
    cal = {k: tok_batches(tokenizer, v, args.n_calib) for k, v in cal_lines.items()}
    measure_lines = {"law": cal_lines["oab"][args.n_calib: args.n_calib + 48],
                     "gen": cal_lines["mmother"][args.n_calib: args.n_calib + 48]}
    log(f"{len(oab_rows)} OAB + {len(mm_rows)} MMMLU rows; calibration pools oab {len(oab_pool)}, mm law "
        f"{len(mm_law_pool)}, mm other {len(mm_other_pool)}")

    def measure(with_mm: bool) -> dict:
        res = evaluate_mmlu_rows(model, tokenizer, oab_rows, sys_prompt=OAB_PROMPT, prompt_style="plain")
        pr, k = res["predictions"], kprobe.stats()
        m = {"oab_acc": res["accuracy"], "oab_nll": res["nll_per_token"],
             "oab_correct": [int(x["predicted"] == x["expected"]) for x in pr],
             "law_logp": k["logp"], "neutral_nll": B.neutral_nll(model, tokenizer, args.neutral)}
        if with_mm:
            r2 = evaluate_mmlu_rows(model, tokenizer, mm_rows, sys_prompt=MM_PROMPT, prompt_style="plain")
            m["mm_correct"] = [int(x["predicted"] == x["expected"]) for x in r2["predictions"]]
            for area in ("law", "other"):
                idx = [i for i, r in enumerate(mm_rows) if r["area"] == area]
                m[f"mm_{area}_acc"] = 100 * sum(m["mm_correct"][i] for i in idx) / len(idx)
        return m

    out = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else None
    if out is None:
        out = {"config": vars(args), "baseline": measure(True), "conditions": [],
               "mm_area": [r["area"] for r in mm_rows]}
        Path(args.out).write_text(json.dumps(out))
    done = {c["name"] for c in out["conditions"]}
    b0 = out["baseline"]
    log(f"fp16: OAB {b0['oab_acc']:.1f}%  MMMLU law {b0['mm_law_acc']:.1f}% other {b0['mm_other_acc']:.1f}%")

    # --- column sets (computed per bit width) -------------------------------------------------------
    sets_cache: dict = {}

    def column_sets(bits: int, k: int) -> dict:
        if (bits, k) in sets_cache:
            return sets_cache[(bits, k)]
        circ = top_units(scores, k)
        cols = unit_sets(circ)
        budget = sum(modules[l].weight.shape[0] * c.numel() for l, c in cols.items())
        dose = {}
        for u in units:
            w = modules[u.layer].weight.data[:, u.start:u.stop].float()
            err = (w - fake_quant_group(w, bits, args.group)).double().square()
            dose[u] = float((err * V[u.layer][u.start:u.stop].to(err.device)).sum())
        rng = random.Random(seed * 1000 + k)
        res = {"circuit": cols, "budget": budget,
               "dose": unit_sets(B.class_matched_top(sorted(units, key=lambda u: -dose[u]), circ)),
               "outlier_all": B.outlier_columns(modules, qlayers, V, bits, args.group, budget),
               "random": [unit_sets(match_random(units, circ, rng)) for _ in range(args.n_random)]}
        sets_cache[(bits, k)] = res
        return res

    def soft_weights(cols: dict, beta: float) -> dict:
        w = {}
        for l, c in cols.items():
            v = torch.ones(modules[l].in_features)
            v[c.cpu()] = 1 + beta
            w[l] = v
        return w

    # --- measured per-layer damage for the allocation suite -------------------------------------------
    table: dict = {}

    def layer_table() -> dict:
        if table:
            return table
        base = {d: lines_nll(model, tokenizer, ls) for d, ls in measure_lines.items()}
        for i, name in enumerate(qlayers):
            for bits in (3, 4):
                with MixedQuant(model, [name], bits, None, args.group, offload=True):
                    table[(name, bits)] = {d: lines_nll(model, tokenizer, ls) - base[d] for d, ls in measure_lines.items()}
            if i % 20 == 0:
                log(f"layer table {i}/{len(qlayers)}")
        out["layer_table"] = {f"{n}|{b}": v for (n, b), v in table.items()}
        Path(args.out).write_text(json.dumps(out))
        return table

    def allocation(domain: str | None, rng: random.Random | None = None) -> dict[str, int]:
        t = layer_table()
        if rng is None:
            gain = {l: (t[(l, 3)][domain] - t[(l, 4)][domain]) / layer_weights[l] for l in qlayers}
            order = sorted(qlayers, key=lambda l: -gain[l])
        else:
            order = list(qlayers)
            rng.shuffle(order)
        total, bits, at4 = sum(layer_weights.values()), {l: 3 for l in qlayers}, 0
        for l in order:
            if at4 + layer_weights[l] <= 0.5 * total:
                bits[l], at4 = 4, at4 + layer_weights[l]
        return bits

    # --- the conditions ----------------------------------------------------------------------------
    S: list[dict] = []

    def add(name, suite, bits=3, rounding="gptq", mm=True, **kw):
        S.append({"name": name, "suite": suite, "bits": bits, "rounding": rounding, "mm": mm, **kw})

    suites = args.suites.split(",")
    if "grid" in suites:
        add("grid_rtn3", "grid", rounding="rtn", mm=False)
        add("grid_gptq3", "grid", mm=False)
        add("grid_gptq3_clip", "grid", mm=False, gk={"clip": True})
        add("grid_gptq3_asym", "grid", mm=False, gk={"asym": True})
        add("grid_gptq3_clip_asym", "grid", mm=False, gk={"clip": True, "asym": True})
        add("grid_gptq3_actorder", "grid", mm=False, gk={"act_order": True})
        add("grid_gptq3_clip_asym_actorder", "grid", mm=False, gk={"clip": True, "asym": True, "act_order": True})
        add("grid_gptq3_g32", "grid", mm=False, group=32)
        add("grid_gptq4", "grid", bits=4, mm=False)
        add("grid_gptq4_clip_asym", "grid", bits=4, mm=False, gk={"clip": True, "asym": True})
    if "cal" in suites:
        for d in ("oab", "mmlaw", "mmother"):
            for a in (0.5, 1.0):
                add(f"cal_{d}_a{a:g}", "cal", cal=(d, a))
        for d in ("oab", "mmother"):
            add(f"cal_{d}_a1_int4", "cal", bits=4, cal=(d, 1.0))
    if "protect" in suites:
        for r in range(args.reps):
            add(f"protect_uniform_r{r}", "protect", rep=r)
            add(f"protect_circuit_r{r}", "protect", rep=r, prot=("circuit", None))
            add(f"protect_outlier_all_r{r}", "protect", rep=r, prot=("outlier_all", None))
            add(f"protect_dose_r{r}", "protect", rep=r, prot=("dose", None))
            for i in range(args.n_random):
                add(f"protect_random{i}_r{r}", "protect", rep=r, prot=("random", i))
    if "soft" in suites:
        for beta in (3.0, 15.0):
            add(f"soft_circuit_b{beta:g}", "soft", soft=("circuit", None, beta))
            for i in range(args.n_random):
                add(f"soft_random{i}_b{beta:g}", "soft", soft=("random", i, beta))
    if "mixed" in suites:
        add("mixed_law", "mixed", bits="mixed", alloc=("law", None))
        add("mixed_gen", "mixed", bits="mixed", alloc=("gen", None))
        for i in range(3):
            add(f"mixed_random{i}", "mixed", bits="mixed", alloc=(None, i))
    if "lowrank" in suites:
        for src in ("gen", "oab", "mmother"):
            for r in (8, 32):
                add(f"lowrank_{src}_r{r}", "lowrank", rank=(src, r))
    only = set(filter(None, args.only.split(",")))

    def run(spec: dict) -> None:
        bits, group = spec["bits"], spec.get("group", args.group)
        kw: dict = {}
        protect = None
        if spec.get("prot"):
            kind, i = spec["prot"]
            cs = column_sets(bits, args.k)
            protect = cs["random"][i] if kind == "random" else cs[kind]
        if spec.get("soft"):
            kind, i, beta = spec["soft"]
            cs = column_sets(bits, args.k)
            kw["col_weight"] = soft_weights(cs["random"][i] if kind == "random" else cs[kind], beta)
        if spec.get("cal"):
            d, a = spec["cal"]
            kw.update(domain_batches=cal[d], alpha=a)
        if spec.get("rank"):
            src, r = spec["rank"]
            kw.update(rank=r, lr_source="gen" if src == "gen" else "dom")
            if src != "gen":
                kw["domain_batches"] = cal[src]
        if bits == "mixed":
            dom, ri = spec["alloc"]
            bits = allocation(dom, random.Random(seed + 100 + ri) if ri is not None else None)
        if spec["rounding"] == "rtn":
            ctx = MixedQuant(model, qlayers, bits, protect, group, offload=True)
        else:
            ctx = GPTQQuant(model, gen_batches_for(spec.get("rep", 0)), qlayers, bits, protect, group, gptq_kwargs=spec.get("gk"), **kw)
        with ctx as q:
            m = measure(spec["mm"])
        total = q.total_weights
        avg = (sum(bits[l] * layer_weights[l] for l in qlayers) / sum(layer_weights.values())) if isinstance(bits, dict) else bits
        entry = {k: v for k, v in spec.items() if k not in ("mm",)} | {
            "bits": "mixed" if isinstance(bits, dict) else bits, "avg_bits": avg,
            "protected_fraction": q.protected_weights / total,
            "lowrank_fraction": getattr(q, "lowrank_params", 0) / total, **m}
        out["conditions"].append(entry)
        msg = f"{spec['name']}: OAB {m['oab_acc']:.1f}% nll_neutral {m['neutral_nll']:.3f}"
        if spec["mm"]:
            msg += f" | MMMLU law {m['mm_law_acc']:.1f}% other {m['mm_other_acc']:.1f}%"
        log(msg)
        Path(args.out).write_text(json.dumps(out))

    for spec in S:
        if spec["name"] in done or (only and spec["name"] not in only):
            continue
        run(spec)
    Path(args.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
