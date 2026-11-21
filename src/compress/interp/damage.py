"""Quantization damage of a unit set: how much a behavior and generic text suffer when the weight columns
of the units are quantized, and controls of the same size and dose.
"""

from __future__ import annotations

import random

import torch

from compress.evaluation.text import line_nll
from compress.interp.units import Unit, match_random, unit_sets
from compress.quant.fake import QuantizeColumns


class KnowledgeProbe:
    """Baseline answer statistics of clean prompts, and damage from column quantization."""

    def __init__(self, model, tokenizer, ids: torch.Tensor, tokens: torch.Tensor,
                 neutral_lines: list[str] | None = None, batch_size: int = 16, max_len: int = 128):
        self.model, self.tokenizer = model, tokenizer
        self.device = next(model.parameters()).device
        self.ids, self.tokens = ids.to(self.device), tokens.to(self.device)
        self.neutral, self.kw = neutral_lines, dict(batch_size=batch_size, max_len=max_len)
        self.base = self._stats()

    def stats(self) -> dict:
        """Mean answer log-prob, top-1 accuracy (and neutral NLL) of the model as it is now."""
        return self._stats()

    @torch.no_grad()
    def _stats(self) -> dict:
        logits = self.model(input_ids=self.ids).logits[:, -1].float()
        logp = logits.log_softmax(-1).gather(1, self.tokens[:, None]).squeeze(1)
        logp = torch.nan_to_num(logp, nan=-30.0, neginf=-30.0).clamp_min(-30.0)  # broken models score -30
        out = {"logp": float(logp.mean()), "acc": float((logits.argmax(-1) == self.tokens).float().mean())}
        if self.neutral:
            nll = line_nll(self.model, self.tokenizer, self.neutral, **self.kw)
            out["nll"] = sum(nll) / len(nll)
        return out

    def damage(self, sets: dict[str, torch.Tensor], bits: int,
               V: dict[str, torch.Tensor] | None = None) -> dict:
        """Loss of mean answer log-prob / accuracy (and neutral NLL increase) under quantization.

        With ``V`` (layer -> E[h^2] per input channel) the diagonal dose
        ``sum_ij E_ij^2 V_j`` of the touched columns is reported as ``dose``.
        """
        with QuantizeColumns(self.model, sets, bits, V) as qc:
            q = self._stats()
        out = {"dlogp": self.base["logp"] - q["logp"], "dacc": self.base["acc"] - q["acc"]}
        if "nll" in q:
            out["dnll"] = q["nll"] - self.base["nll"]
        if V is not None:
            out["dose"] = qc.dose
        return out


@torch.no_grad()
def dose_of(model: torch.nn.Module, units: list[Unit], bits: int, V: dict[str, torch.Tensor]) -> float:
    """Diagonal dose of quantizing the columns of ``units`` (no forward pass)."""
    with QuantizeColumns(model, unit_sets(units), bits, V) as qc:
        return qc.dose


def dose_matched_random(model: torch.nn.Module, all_units: list[Unit], circuit: list[Unit],
                        V: dict[str, torch.Tensor], bits: int, rng: random.Random,
                        n_keep: int = 10, n_candidates: int = 200) -> list[list[Unit]]:
    """Class-matched random sets whose dose is closest to the circuit's (out of many draws)."""
    target = dose_of(model, circuit, bits, V)
    cands = [match_random(all_units, circuit, rng) for _ in range(n_candidates)]
    doses = [dose_of(model, c, bits, V) for c in cands]
    order = sorted(range(len(cands)), key=lambda i: abs(doses[i] - target))
    return [cands[i] for i in order[:n_keep]]


def additivity(probe: KnowledgeProbe, units: list[Unit], bits: int) -> dict:
    """Joint damage of a unit set against the sum of its members' separate damages."""
    joint = probe.damage(unit_sets(units), bits)["dlogp"]
    singles = [probe.damage(unit_sets([u]), bits)["dlogp"] for u in units]
    return {"joint": joint, "sum_single": sum(singles), "n": len(units)}
