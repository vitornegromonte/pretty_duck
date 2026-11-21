"""A suite of quantization conditions evaluated on one model, with a circuit as the protected set.

A *condition* is one way to quantize the model: a rounding (RTN or GPTQ), a bit width, and options (grid, protected
columns, soft weights, calibration, bit allocation, low-rank correction). ``build_conditions`` lists them by suite and
is a pure function. ``QuantSuite`` runs them one by one on the same questions, stores the per-question results, and
skips the conditions that are already in the output file, so a run can be resumed.

Suites (all at ``bits`` group-wise unless the name says otherwise):

  grid       rtn / gptq / +clip / +asym / +act-order / group 32, and int4 (how good can the plain quantizer get)
  cal        GPTQ Hessian from (1-a) generic text plus a: calibration text of the target or the control subjects
  protect    columns kept at full precision: circuit / dose / awq / outlier_all / random, at equal weights
  soft       circuit columns get weight 1+beta in the GPTQ objective instead of full precision (no extra bytes)
  mixed      3- and 4-bit layers at 3.5 bits on average: by measured damage on target or control text, or random
  lowrank    full-precision rank-r correction of the rounding error, whitened by a generic or domain Hessian
  reference  uniform RTN, GPTQ, and the authors' GPTQ (cross-check; needs third_party/gptq_ref)

Evaluation rows come from two groups of MMLU subjects, ``target`` and ``control``. Calibration text comes from the
rows of the same groups that are not evaluation rows.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .calibration import mixed_domain_batches, read_lines, text_batches
from .circuit import class_matched_top, match_random, top_units, unit_sets
from .evaluate import token_nll
from .experiment import CircuitContext, accuracy_by_area, score_rows
from .gptq import GPTQQuant
from .gptq_reference import reference_quantize
from .mmlu import load_mmlu_groups, row_to_text
from .protect import MixedQuant, outlier_columns
from .provenance import provenance
from .quant_probe import fake_quant_group

log = logging.getLogger(__name__)

SYS_PROMPT = "The following are multiple choice questions. Answer with the letter of the correct choice."
CALIBRATION_DOMAINS = ("bio", "cs", "hum", "law", "math")  # <calib_dir>/<domain>.txt, mixed evenly
SUITES = ("grid", "cal", "protect", "soft", "mixed", "lowrank", "reference")
PROTECT_KINDS = ("circuit", "dose", "awq", "outlier_all")


@dataclass(frozen=True)
class Condition:
    """One way to quantize the model. Unused options keep their defaults."""

    name: str
    suite: str
    bits: int | str = 3                         # "mixed": a per-layer allocation (see ``alloc``)
    rounding: str = "gptq"                      # rtn | gptq | gptq_ref
    group: int | None = None                    # None: the suite's group size
    full_eval: bool = True                      # False: target questions only
    rep: int = 0                                # calibration replicate (a different calibration sample)
    gptq: dict = field(default_factory=dict)    # grid options: clip, asym, act_order
    protect: tuple | None = None                # (kind, random index or None, k)
    soft: tuple | None = None                   # (kind, random index or None, beta, k)
    cal: tuple | None = None                    # (group name, alpha)
    rank: tuple | None = None                   # (Hessian source, rank)
    alloc: tuple | None = None                  # (group name or None, random seed offset or None)


@dataclass
class SuiteConfig:
    out: str
    calib_dir: str
    neutral: str                                # text file for the perplexity check
    target_subjects: list[str]
    control_subjects: list[str]
    suites: tuple[str, ...] = SUITES[:-1]
    only: frozenset[str] = frozenset()
    bits: int = 3
    group: int = 64
    ks: tuple[int, ...] = (128,)
    n_random: int = 2
    reps: int = 3
    roundings: tuple[str, ...] = ("gptq",)
    keep_blocks: bool = False
    n_eval: int = 300
    n_calib: int = 160
    n_measure: int = 48                         # lines per group for the per-layer damage table
    sys_prompt: str = SYS_PROMPT


def build_conditions(cfg: SuiteConfig) -> list[Condition]:
    """The conditions of the selected suites, in running order."""
    bits, out = cfg.bits, []

    def add(**kw) -> None:
        out.append(Condition(**kw))

    if "grid" in cfg.suites:
        add(name=f"grid_rtn{bits}", suite="grid", bits=bits, rounding="rtn", full_eval=False)
        add(name=f"grid_gptq{bits}", suite="grid", bits=bits, full_eval=False)
        for tag, options in (("clip", {"clip": True}), ("asym", {"asym": True}),
                             ("clip_asym", {"clip": True, "asym": True}), ("actorder", {"act_order": True}),
                             ("clip_asym_actorder", {"clip": True, "asym": True, "act_order": True})):
            add(name=f"grid_gptq{bits}_{tag}", suite="grid", bits=bits, full_eval=False, gptq=options)
        add(name=f"grid_gptq{bits}_g32", suite="grid", bits=bits, full_eval=False, group=32)
        add(name="grid_gptq4", suite="grid", bits=4, full_eval=False)
        add(name="grid_gptq4_clip_asym", suite="grid", bits=4, full_eval=False, gptq={"clip": True, "asym": True})
    if "cal" in cfg.suites:
        for domain in ("target", "control"):
            for alpha in (0.5, 1.0):
                add(name=f"cal_{domain}_a{alpha:g}", suite="cal", bits=bits, cal=(domain, alpha))
            add(name=f"cal_{domain}_a1_int4", suite="cal", bits=4, cal=(domain, 1.0))
    if "protect" in cfg.suites:
        for rounding in cfg.roundings:
            reps = range(cfg.reps) if rounding != "rtn" else range(1)  # RTN is deterministic
            for rep in reps:
                tag = f"{rounding}" + (f"_r{rep}" if rounding != "rtn" else "")
                add(name=f"protect_{tag}_uniform", suite="protect", bits=bits, rounding=rounding, rep=rep)
                for k in cfg.ks:
                    for kind in PROTECT_KINDS:
                        add(name=f"protect_{tag}_{kind}_k{k}", suite="protect", bits=bits, rounding=rounding,
                            rep=rep, protect=(kind, None, k))
                    for i in range(cfg.n_random):
                        add(name=f"protect_{tag}_random{i}_k{k}", suite="protect", bits=bits, rounding=rounding,
                            rep=rep, protect=("random", i, k))
    if "soft" in cfg.suites:
        for k in cfg.ks:
            for beta in (3.0, 15.0):
                add(name=f"soft_circuit_b{beta:g}_k{k}", suite="soft", bits=bits, soft=("circuit", None, beta, k))
                for i in range(cfg.n_random):
                    add(name=f"soft_random{i}_b{beta:g}_k{k}", suite="soft", bits=bits, soft=("random", i, beta, k))
    if "mixed" in cfg.suites:
        add(name="mixed_target", suite="mixed", bits="mixed", alloc=("target", None))
        add(name="mixed_control", suite="mixed", bits="mixed", alloc=("control", None))
        for i in range(3):
            add(name=f"mixed_random{i}", suite="mixed", bits="mixed", alloc=(None, i))
    if "lowrank" in cfg.suites:
        for source in ("gen", "target", "control"):
            for rank in (8, 32):
                add(name=f"lowrank_{source}_r{rank}", suite="lowrank", bits=bits, rank=(source, rank))
    if "reference" in cfg.suites:
        for rounding in ("rtn", "gptq", "gptq_ref"):
            add(name=f"reference_{rounding}_uniform", suite="reference", bits=bits, rounding=rounding)
    names = [c.name for c in out]
    if len(names) != len(set(names)):
        raise ValueError("condition names must be unique")
    return out


class QuantSuite:
    """Runs the conditions of a ``SuiteConfig`` on one ``CircuitContext`` and writes the results as JSON."""

    def __init__(self, ctx: CircuitContext, cfg: SuiteConfig, rows_and_pools: tuple | None = None):
        """``rows_and_pools`` is ``(eval_rows, pools)`` as ``load_mmlu_groups`` returns; the default loads MMLU."""
        self.ctx, self.cfg = ctx, cfg
        self.eval_rows, pools = rows_and_pools or load_mmlu_groups(
            {"target": cfg.target_subjects, "control": cfg.control_subjects}, cfg.n_eval, seed=ctx.seed)
        self.target_rows = [r for r in self.eval_rows if r["area"] == "target"]
        lines = {name: [row_to_text(r) for r in rows] for name, rows in pools.items()}
        needed = cfg.n_calib + cfg.n_measure  # calibration lines, then lines for the per-layer damage table
        for name, group_lines in lines.items():
            if len(group_lines) < needed:
                raise ValueError(f"the {name} pool has {len(group_lines)} lines, fewer than {needed}")
        self.domain_batches = {name: text_batches(ctx.tokenizer, group_lines, cfg.n_calib)
                               for name, group_lines in lines.items()}
        self.measure_lines = {name: group_lines[cfg.n_calib:needed] for name, group_lines in lines.items()}
        self.neutral_lines = read_lines(cfg.neutral, min_chars=80)[:48]
        self._general: dict[int, list[dict]] = {}
        self._sets: dict[tuple[int, int], dict] = {}
        self._table: dict[tuple[str, int], dict[str, float]] = {}
        self.result: dict = {}

    # ----- measurement -------------------------------------------------------------------------------------
    def measure(self, full: bool) -> dict:
        ctx = self.ctx
        rows = self.eval_rows if full else self.target_rows
        scored = score_rows(ctx.model, ctx.tokenizer, rows, self.cfg.sys_prompt)
        by_area = accuracy_by_area(scored["correct"], rows)
        facts = ctx.fact_probe.stats()
        return {"target_acc": by_area["target"], "control_acc": by_area.get("control"),
                "letter_nll": scored["nll"], "correct": scored["correct"],
                "fact_logp": facts["logp"], "fact_acc": facts["acc"],
                "neutral_nll": token_nll(ctx.model, ctx.tokenizer, self.neutral_lines)}

    # ----- inputs of the conditions ------------------------------------------------------------------------
    def general_batches(self, rep: int) -> list[dict]:
        """Generic calibration text; replicate ``rep`` draws another sample."""
        if rep not in self._general:
            self._general[rep] = mixed_domain_batches(
                self.ctx.tokenizer, self.cfg.calib_dir, CALIBRATION_DOMAINS, self.cfg.n_calib, seed=self.ctx.seed + rep)
        return self._general[rep]

    def column_sets(self, bits: int, k: int) -> dict:
        """Protected column sets of the same size in weights: circuit, dose, awq, outlier_all, and random sets."""
        if (bits, k) in self._sets:
            return self._sets[(bits, k)]
        ctx = self.ctx
        circuit = top_units(ctx.scores, k)
        circuit_columns = unit_sets(circuit)
        budget = sum(ctx.modules[layer].weight.shape[0] * cols.numel() for layer, cols in circuit_columns.items())
        dose = {}
        for unit in ctx.units:
            w = ctx.modules[unit.layer].weight.data[:, unit.start:unit.stop].float()
            error = (w - fake_quant_group(w, bits, self.cfg.group)).double().square()
            dose[unit] = float((error * ctx.V[unit.layer][unit.start:unit.stop].to(error.device)).sum())
        share = budget / ctx.write_weights
        awq = {layer: torch.sort(ctx.A[layer].topk(max(1, round(share * ctx.A[layer].numel()))).indices).values
               for layer in ctx.write_layers}
        rng = random.Random(ctx.seed * 1000 + k)
        sets = {"circuit": circuit_columns,
                "dose": unit_sets(class_matched_top(sorted(ctx.units, key=lambda u: -dose[u]), circuit)),
                "awq": awq,
                "outlier_all": outlier_columns(ctx.modules, ctx.qlayers, ctx.V, bits, self.cfg.group, budget),
                "random": [unit_sets(match_random(ctx.units, circuit, rng)) for _ in range(self.cfg.n_random)]}
        self._sets[(bits, k)] = sets
        return sets

    def _pick(self, bits: int, kind: str, index: int | None, k: int) -> dict:
        sets = self.column_sets(bits, k)
        return sets["random"][index] if kind == "random" else sets[kind]

    def soft_weights(self, columns: dict, beta: float) -> dict[str, torch.Tensor]:
        """Per-column weight 1+beta on ``columns`` and 1 elsewhere, for the write projections."""
        weights = {}
        for layer, cols in columns.items():
            w = torch.ones(self.ctx.modules[layer].in_features)
            w[cols.cpu()] = 1 + beta
            weights[layer] = w
        return weights

    def layer_table(self) -> dict[tuple[str, int], dict[str, float]]:
        """Damage (NLL increase on target and control text) of quantizing each layer alone at 3 and 4 bits."""
        if self._table:
            return self._table
        ctx = self.ctx
        base = {name: token_nll(ctx.model, ctx.tokenizer, lines) for name, lines in self.measure_lines.items()}
        for i, layer in enumerate(ctx.qlayers):
            for bits in (3, 4):
                with MixedQuant(ctx.model, [layer], bits, None, self.cfg.group, offload=True):
                    self._table[(layer, bits)] = {
                        name: token_nll(ctx.model, ctx.tokenizer, lines) - base[name]
                        for name, lines in self.measure_lines.items()}
            if i % 20 == 0:
                log.info("layer table %d/%d", i, len(ctx.qlayers))
        self.result["layer_table"] = {f"{layer}|{bits}": v for (layer, bits), v in self._table.items()}
        return self._table

    def allocation(self, group: str | None, rng: random.Random | None) -> dict[str, int]:
        """Half of the weights at 4 bits and the rest at 3: the layers where 4 bits saves the most damage per
        weight (``group``), or random layers (``rng``)."""
        ctx, table = self.ctx, self.layer_table()
        if rng is None:
            gain = {layer: (table[(layer, 3)][group] - table[(layer, 4)][group]) / ctx.layer_weights[layer]
                    for layer in ctx.qlayers}
            order = sorted(ctx.qlayers, key=lambda layer: -gain[layer])
        else:
            order = list(ctx.qlayers)
            rng.shuffle(order)
        total, bits, at4 = sum(ctx.layer_weights.values()), {layer: 3 for layer in ctx.qlayers}, 0
        for layer in order:
            if at4 + ctx.layer_weights[layer] <= 0.5 * total:
                bits[layer], at4 = 4, at4 + ctx.layer_weights[layer]
        return bits

    # ----- running -----------------------------------------------------------------------------------------
    def _quantizer(self, cond: Condition, bits):
        """The context manager that quantizes the model for ``cond`` and restores it on exit."""
        ctx, cfg = self.ctx, self.cfg
        group = cond.group or cfg.group
        protect = self._pick(bits, cond.protect[0], cond.protect[1], cond.protect[2]) if cond.protect else None
        if cond.rounding == "rtn":
            return MixedQuant(ctx.model, ctx.qlayers, bits, protect, group, offload=True, keep_blocks=cfg.keep_blocks)
        extra: dict = {}
        if cond.soft:
            kind, index, beta, k = cond.soft
            extra["col_weight"] = self.soft_weights(self._pick(bits, kind, index, k), beta)
        if cond.cal:
            name, alpha = cond.cal
            extra.update(domain_batches=self.domain_batches[name], alpha=alpha)
        if cond.rank:
            source, rank = cond.rank
            extra.update(rank=rank, lr_source="gen" if source == "gen" else "dom")
            if source != "gen":
                extra["domain_batches"] = self.domain_batches[source]
        quantizer = reference_quantize if cond.rounding == "gptq_ref" else None
        return GPTQQuant(ctx.model, self.general_batches(cond.rep), ctx.qlayers, bits, protect, group,
                         quantizer=quantizer, gptq_kwargs=cond.gptq or None, **extra)

    def run_condition(self, cond: Condition) -> dict:
        """Quantize, measure, restore, and return the result row."""
        bits = cond.bits
        if bits == "mixed":
            name, offset = cond.alloc
            bits = self.allocation(name, random.Random(self.ctx.seed + 100 + offset) if offset is not None else None)
        with self._quantizer(cond, bits) as quantized:
            measured = self.measure(cond.full_eval)
        total = quantized.total_weights
        layer_weights = self.ctx.layer_weights
        average_bits = (sum(bits[layer] * layer_weights[layer] for layer in layer_weights) / sum(layer_weights.values())
                        if isinstance(bits, dict) else bits)
        return {**asdict(cond), "bits": "mixed" if isinstance(bits, dict) else bits, "avg_bits": average_bits,
                "protected_fraction": quantized.protected_weights / total,
                "lowrank_fraction": getattr(quantized, "lowrank_params", 0) / total, **measured}

    def _save(self) -> None:
        Path(self.cfg.out).write_text(json.dumps(self.result))

    def run(self) -> dict:
        """Run every selected condition that is not already in the output file."""
        out = Path(self.cfg.out)
        self.result = json.loads(out.read_text()) if out.exists() else {}
        if not self.result:
            self.result = {"config": asdict(self.cfg) | {"only": sorted(self.cfg.only)}, "provenance": provenance(),
                           "baseline": self.measure(True), "conditions": [],
                           "eval_areas": [r["area"] for r in self.eval_rows]}
            self._save()
        self.result.setdefault("provenance_resumed", []).append(provenance())
        done = {c["name"] for c in self.result["conditions"]}
        for cond in build_conditions(self.cfg):
            if cond.name in done or (self.cfg.only and cond.name not in self.cfg.only):
                continue
            row = self.run_condition(cond)
            self.result["conditions"].append(row)
            log.info("%s: target %.1f%% neutral NLL %.3f", cond.name, row["target_acc"], row["neutral_nll"])
            self._save()
        self._save()
        return self.result
