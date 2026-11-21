"""Shared setup of the protection experiments: a model, one circuit from a probe file, and the scoring of a benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from compress.data.facts import FACTS, TEMPLATES
from compress.interp.damage import KnowledgeProbe
from compress.interp.minimal_pairs import build_facts, make_pairs, split_facts
from compress.interp.units import Unit, enumerate_units, unit_sets
from compress.models import load_model
from compress.quant import stats as stats_mod


@dataclass
class CircuitContext:
    """Everything a protection experiment needs from one model, one fact domain, and one circuit."""

    model: torch.nn.Module
    tokenizer: object
    modules: dict[str, torch.nn.Module]
    units: list[Unit]
    scores: list[dict]              # per-unit {"unit", "suff", "nec"} of the domain, from the probe file
    qlayers: list[str]              # every Linear that is quantized
    V: dict[str, torch.Tensor]      # mean E[h^2] per input channel (dose), averaged over the stats domains
    A: dict[str, torch.Tensor]      # mean E|h| per input channel (activation magnitude)
    write_layers: list[str]         # projections that write to the residual stream (units live here)
    write_weights: int
    layer_weights: dict[str, int]
    seed: int
    fact_probe: KnowledgeProbe      # log-probability of held-out facts of the domain


def build_context(model, tokenizer, probe: dict, stats_path: str | Path, domain: str,
                  *, require_correct: bool = True) -> CircuitContext:
    """Assemble the context from a loaded model, a probe dict (``circuit_probe.py`` output), and a stats file."""
    seed = probe["config"]["seed"]
    modules = dict(model.named_modules())
    units = enumerate_units(model, probe["config"]["mlp_group"])
    by_key = {(u.layer, u.start, u.stop): u for u in units}
    scores = [{"unit": by_key[(layer, start, stop)], "suff": suff, "nec": nec}
              for layer, start, stop, suff, nec in probe["domains"][domain]["units"]]
    domain_stats, _ = stats_mod.load_stats(stats_path)
    qlayers = sorted(next(iter(domain_stats.values())))

    def mean(key: str) -> dict[str, torch.Tensor]:
        return {layer: torch.stack([d[layer][key].double() for d in domain_stats.values()]).mean(0)
                for layer in qlayers}

    write_layers = sorted(unit_sets(units))
    facts = build_facts(model, tokenizer, TEMPLATES[domain], FACTS[domain], require_correct=require_correct)
    _, validation = split_facts(facts, seed)
    pairs = make_pairs(validation, seed + 1)
    return CircuitContext(
        model=model, tokenizer=tokenizer, modules=modules, units=units, scores=scores, qlayers=qlayers,
        V=mean("V"), A=mean("A"), write_layers=write_layers,
        write_weights=sum(modules[layer].weight.numel() for layer in write_layers),
        layer_weights={layer: modules[layer].weight.numel() for layer in qlayers}, seed=seed,
        fact_probe=KnowledgeProbe(model, tokenizer, pairs.ids_clean, pairs.tok_clean),
    )


def load_context(model_path: str, probe_path: str | Path, stats_path: str | Path, domain: str) -> CircuitContext:
    """Load the model and build the context (see ``build_context``)."""
    model, tokenizer = load_model(model_path)
    probe = json.loads(Path(probe_path).read_text())
    return build_context(model, tokenizer, probe, stats_path, domain)
