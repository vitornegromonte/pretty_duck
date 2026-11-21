"""Measured quantization sensitivity: Delta L_{g,d}(q) = L_d(W with layer g in format q) - L_d(W).

The planner's diagonal estimate ``D`` is a local approximation; here the loss
increase is *measured* by quantizing one Linear at a time (real formats, via
``quantize_weight``/dequantize, i.e. simulated inference on dequantized
weights) and scoring held-out text per domain.  Per-line paired deltas are kept
so intervals and held-out comparisons can be computed later.

``measure_plans`` evaluates several layers quantized *together*, to test
whether single-layer effects add up.  Weights are restored exactly on exit,
also on errors.
"""

from __future__ import annotations

import torch

from compress.evaluation.text import line_nll
from compress.quant.formats import quantize_weight


class QuantizeLayers:
    """Context manager: replace whole Linear weights by their quantize/dequantize round trip."""

    def __init__(self, model: torch.nn.Module, plan: dict[str, str]):
        self.model, self.plan = model, plan
        self._saved: list[tuple[torch.nn.Linear, torch.Tensor]] = []

    @torch.no_grad()
    def __enter__(self) -> "QuantizeLayers":
        modules = dict(self.model.named_modules())
        try:
            for name, fmt in self.plan.items():
                module = modules.get(name)
                if not isinstance(module, torch.nn.Linear):
                    raise KeyError(f"plan layer {name!r} is not an nn.Linear in the model")
                weight = module.weight.data
                self._saved.append((module, weight.clone()))
                weight.copy_(quantize_weight(weight, fmt).dequant.to(weight.dtype))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    @torch.no_grad()
    def __exit__(self, *exc) -> None:
        for module, original in reversed(self._saved):
            module.weight.data.copy_(original)
        self._saved.clear()


def _score(model, tokenizer, eval_sets, batch_size, max_len) -> dict[str, list[float]]:
    return {d: line_nll(model, tokenizer, lines, batch_size=batch_size, max_len=max_len)
            for d, lines in eval_sets.items()}


def _paired(delta: list[float]) -> dict:
    t = torch.tensor(delta, dtype=torch.float64)
    se = float(t.std(unbiased=True) / len(t) ** 0.5) if len(t) > 1 else float("nan")
    return {"delta": float(t.mean()), "se": se}


def measure_sensitivity(
    model: torch.nn.Module,
    tokenizer,
    eval_sets: dict[str, list[str]],
    layers: list[str],
    formats: list[str],
    *,
    batch_size: int = 16,
    max_len: int = 128,
    keep_lines: bool = True,
    log=lambda msg: None,
) -> dict:
    """One-layer-at-a-time ``Delta L`` for every (layer, format, domain), paired per line."""
    base = _score(model, tokenizer, eval_sets, batch_size, max_len)
    out = {"baseline_nll": {d: sum(v) / len(v) for d, v in base.items()},
           "n_lines": {d: len(v) for d, v in eval_sets.items()}, "layers": {}}
    for i, layer in enumerate(layers):
        out["layers"][layer] = {}
        for fmt in formats:
            with QuantizeLayers(model, {layer: fmt}):
                scored = _score(model, tokenizer, eval_sets, batch_size, max_len)
            entry = {}
            for d in eval_sets:
                delta = [s - b for s, b in zip(scored[d], base[d])]
                entry[d] = _paired(delta) | ({"lines": delta} if keep_lines else {})
            out["layers"][layer][fmt] = entry
        if (i + 1) % 10 == 0 or i + 1 == len(layers):
            log(f"{i + 1}/{len(layers)} layers")
    return out


def measure_plans(
    model: torch.nn.Module,
    tokenizer,
    eval_sets: dict[str, list[str]],
    plans: dict[str, dict[str, str]],
    *,
    batch_size: int = 16,
    max_len: int = 128,
    log=lambda msg: None,
) -> dict:
    """``Delta L`` per domain for each multi-layer plan ``{name: {layer: format}}``."""
    base = _score(model, tokenizer, eval_sets, batch_size, max_len)
    out = {}
    for name, plan in plans.items():
        with QuantizeLayers(model, plan):
            scored = _score(model, tokenizer, eval_sets, batch_size, max_len)
        out[name] = {d: _paired([s - b for s, b in zip(scored[d], base[d])]) for d in eval_sets}
        log(f"{name}: " + ", ".join(f"{d} {v['delta']:+.4f}" for d, v in out[name].items()))
    return out
