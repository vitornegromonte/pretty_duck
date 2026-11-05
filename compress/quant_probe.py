"""Quantization as an intervention: fake-quantize only chosen weight columns.

Input channel ``j`` of a Linear is column ``j`` of its weight.  For a channel
set we replace those columns by their quantize/dequantize round trip (per-row
scales computed over the selected columns only) and leave everything else at
full precision.  Because the error is confined to the set, comparing sets of
equal size and equal diagonal dose ``sum_ij E_ij^2 V_j`` asks whether some
channels matter more *for a behavior* than others of the same size and cost.

This is a *simulation* of damage, not a storage format: ``bits`` below 4 are
allowed purely as stronger interventions (nothing is packed or stored).  The
weights are restored exactly on exit, also on errors.
"""

from __future__ import annotations

import torch

from .evaluate import batch_option_nll
from .quant import _quantize_int


def fake_quant(W: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-row quantize/dequantize of ``W`` at ``bits`` (float32 result)."""
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be in [2, 8], got {bits}")
    return _quantize_int(W.float(), bits).dequant


class QuantizeColumns:
    """Context manager: fake-quantize the selected columns of the named Linears.

    ``V`` (optional, layer -> E[h^2] over input channels) lets the manager
    accumulate ``dose`` = sum_ij E_ij^2 V_j over the touched columns.
    """

    def __init__(self, model: torch.nn.Module, sets: dict[str, torch.Tensor], bits: int,
                 V: dict[str, torch.Tensor] | None = None):
        self.model, self.sets, self.bits, self.V = model, sets, bits, V
        self.dose = 0.0
        self._saved: list[tuple[torch.nn.Linear, torch.Tensor, torch.Tensor]] = []

    @torch.no_grad()
    def __enter__(self) -> "QuantizeColumns":
        modules = dict(self.model.named_modules())
        try:
            for name, idx in self.sets.items():
                if idx.numel() == 0:
                    continue
                module = modules.get(name)
                if not isinstance(module, torch.nn.Linear):
                    raise KeyError(f"quantization target {name!r} is not an nn.Linear in the model")
                weight = module.weight.data
                cols = idx.to(weight.device)
                original = weight[:, cols].clone()
                deq = fake_quant(original, self.bits)
                self._saved.append((module, cols, original))
                weight[:, cols] = deq.to(weight.dtype)
                if self.V is not None:
                    err = (original.float() - deq).double().square()
                    self.dose += float((err * self.V[name].to(err.device)[cols].double()).sum())
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    @torch.no_grad()
    def __exit__(self, *exc) -> None:
        for module, cols, original in reversed(self._saved):
            module.weight.data[:, cols] = original
        self._saved.clear()


def _margins(model, tokenizer, rows, system_prompt, style) -> list[float]:
    """Per-question ``logp(matching) - logp(other)`` (summed over answer tokens)."""
    device = next(model.parameters()).device
    out = []
    for row in rows:
        (nll_a, _), (nll_b, _) = batch_option_nll(
            model, tokenizer, system_prompt, row["question"], [row["A"], row["B"]], device, style)
        out.append((nll_b - nll_a) if row["matching"] == "A" else (nll_a - nll_b))
    return out


def _ratio_ci(num: torch.Tensor, den: torch.Tensor, n_boot: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    n = len(num)
    boots = torch.empty(n_boot, dtype=torch.float64)
    for b in range(n_boot):
        i = torch.randint(n, (n,), generator=g)
        boots[b] = num[i].sum() / den[i].sum()
    lo, hi = torch.quantile(boots, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return float(num.sum() / den.sum()), lo, hi


@torch.no_grad()
def quant_persona_probe(
    model: torch.nn.Module,
    tokenizer,
    rows: list[dict],
    *,
    prompts: dict[str, str],
    conditions: dict[str, tuple[dict[str, torch.Tensor], int]],
    V: dict[str, torch.Tensor] | None = None,
    style: str = "auto",
    n_boot: int = 1000,
    seed: int = 0,
    keep_per_question: bool = False,
    log=lambda msg: None,
) -> dict:
    """How much of the seek-minus-reject margin shift survives each column quantization.

    ``prompts`` needs ``seek`` and ``reject``; an optional ``neutral`` prompt
    gives a persona-free measure of generic damage (mean |change| of its
    margin).  ``conditions`` maps a name to ``(channel sets, bits)``.
    ``retention`` = sum(shift_q) / sum(shift_fp) over questions.
    """
    model.eval()
    base = {p: torch.tensor(_margins(model, tokenizer, rows, s, style), dtype=torch.float64)
            for p, s in prompts.items()}
    shift0 = base["seek"] - base["reject"]
    report = {"n_questions": len(rows), "baseline_shift": float(shift0.mean()),
              "baseline_margin": {p: float(m.mean()) for p, m in base.items()}, "conditions": {}}
    if keep_per_question:
        report["per_question"] = {"baseline": {p: m.tolist() for p, m in base.items()}, "conditions": {}}
    for name, (sets, bits) in conditions.items():
        with QuantizeColumns(model, sets, bits, V) as q:
            m = {p: torch.tensor(_margins(model, tokenizer, rows, s, style), dtype=torch.float64)
                 for p, s in prompts.items()}
            dose = q.dose
        shift = m["seek"] - m["reject"]
        retention, lo, hi = _ratio_ci(shift, shift0, n_boot, seed)
        entry = {"bits": bits, "dose": dose, "retention": retention, "ci95": [lo, hi],
                 "shift": float(shift.mean()),
                 "margin": {p: float(x.mean()) for p, x in m.items()},
                 "channels": sum(int(i.numel()) for i in sets.values())}
        if "neutral" in m:
            entry["generic_damage"] = float((m["neutral"] - base["neutral"]).abs().mean())
        report["conditions"][name] = entry
        if keep_per_question:
            report["per_question"]["conditions"][name] = {p: x.tolist() for p, x in m.items()}
        log(f"{name}: retention {retention:+.3f} [{lo:+.3f}, {hi:+.3f}] dose {dose:.4g}")
    return report
