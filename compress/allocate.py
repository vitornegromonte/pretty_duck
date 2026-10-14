"""Greedy heterogeneous bit allocation per Linear block.

Given per-block, per-domain quantization-error estimates (from
:func:`compress.quant.diagonal_error`) and per-format byte costs, build a
plan that minimizes stored bytes subject to a target-domain error budget,
or minimizes estimated target error subject to a size budget.

The search is the textbook greedy of TASK §8: start every block at the
highest precision (fp16), repeatedly apply the candidate reduction with the
smallest estimated-error increase per byte saved,

    rho = (D_T(b') - D_T(b)) / (B(b) - B(b')),

stopping when the budget forbids any further reduction.  ``signal=
"contrastive"`` multiplies rho by (1 + lambda * c_g), where c_g in [0, 1] is
the block's normalized contrastive importance C_g = I_T,g - lambda *
mean_S I_S,g; blocks unique to the target domain are then compressed last.
This is a prioritization hypothesis, not a guarantee of better end-to-end
accuracy -- only full evaluation can tell.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .quant import PORTABLE_FORMATS, is_available, quantize_weight, quant_error, \
    diagonal_error, block_importance, storage_nbytes


def estimate_blocks(
    model: torch.nn.Module,
    domains_stats: dict[str, dict],
    formats: list[str],
) -> dict[str, dict]:
    """Per-Linear shape, importance and diagonal quantization error.

    Returns ``{layer_name: {"shape": (o, i), "nbytes": {fmt: int},
    "importance": {domain: float}, "error": {domain: {fmt: float}}}}``.
    """
    for fmt in formats:
        ok, reason = is_available(fmt)
        if not ok:
            raise ValueError(reason)

    names = dict(model.named_modules())
    estimates: dict[str, dict] = {}
    for name, stats in _iter_layer_stats(names, domains_stats):
        W = names[name].weight.detach().float()
        entry: dict = {
            "shape": tuple(W.shape),
            "nbytes": {fmt: storage_nbytes(tuple(W.shape), fmt) for fmt in formats},
            "importance": {},
            "error": {},
        }
        for domain, dstats in domains_stats.items():
            A, V = dstats[name]["A"], dstats[name]["V"]
            entry["importance"][domain] = block_importance(W, A)
            entry["error"][domain] = {}
            for fmt in formats:
                q = quantize_weight(W, fmt)
                entry["error"][domain][fmt] = diagonal_error(quant_error(W, q), V)
        estimates[name] = entry
    return estimates


def _iter_layer_stats(named_modules: dict, domains_stats: dict):
    """Yield (name, stats) for Linear layers that have stats in every domain."""
    any_domain = next(iter(domains_stats.values()))
    for name, stats in any_domain.items():
        module = named_modules.get(name)
        if module is None:
            raise KeyError(f"stats refer to unknown module {name!r}")
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"stats for {name!r} but module is {type(module).__name__}")
        for domain, dstats in domains_stats.items():
            if name not in dstats:
                raise KeyError(f"layer {name!r} missing from domain {domain!r}")
        yield name, stats


def _bits(fmt: str) -> int:
    return PORTABLE_FORMATS[fmt]


def build_plan(
    estimates: dict[str, dict],
    *,
    target: str,
    formats: list[str],
    error_budget: float | None = None,
    size_budget: int | None = None,
    secondary: list[str] | None = None,
    lam: float = 0.0,
    signal: str = "error",
    model: str = "",
) -> dict:
    """Greedy allocation; returns a JSON-serializable plan (see module docstring).

    Exactly one of ``error_budget`` (absolute sum of D_T over blocks) or
    ``size_budget`` (total payload bytes across blocks) must be given.
    """
    if (error_budget is None) == (size_budget is None):
        raise ValueError("provide exactly one of error_budget or size_budget")
    if "fp16" not in formats:
        raise ValueError("formats must include 'fp16' (the uncompressed baseline precision)")
    if signal not in ("error", "contrastive"):
        raise ValueError(f"unknown signal {signal!r}")
    for fmt in formats:
        if fmt not in PORTABLE_FORMATS:
            raise ValueError(
                f"allocator supports only portable formats {sorted(PORTABLE_FORMATS)}; "
                f"{fmt!r} has no exact analytic byte formula here"
            )
    if target not in next(iter(estimates.values()))["error"]:
        raise KeyError(f"target domain {target!r} not in estimates")
    secondary = secondary or []
    for d in secondary:
        if d not in next(iter(estimates.values()))["error"]:
            raise KeyError(f"secondary domain {d!r} not in estimates")

    # Normalized contrastive importance in [0, 1]; used only by signal="contrastive".
    contrast = _normalized_contrast(estimates, target, secondary, lam)

    state = {name: "fp16" for name in estimates}
    used_bytes = sum(e["nbytes"]["fp16"] for e in estimates.values())
    total_fp16_bytes = used_bytes
    est_error = 0.0
    decisions: list[dict] = []

    while True:
        candidates = _candidates(estimates, state, target, contrast, signal, lam, set(formats))
        if not candidates:
            break
        if error_budget is not None:
            candidates = [c for c in candidates if est_error + c["delta_error"] <= error_budget]
            if not candidates:
                break
        # rho first, then name/format for a deterministic tie-break.
        candidates.sort(key=lambda c: (c["rho"], c["block"], c["to"]))
        step = candidates[0]
        state[step["block"]] = step["to"]
        used_bytes -= step["delta_bytes"]
        est_error += step["delta_error"]
        decisions.append(step)
        if size_budget is not None and used_bytes <= size_budget:
            break

    if size_budget is not None and used_bytes > size_budget:
        raise ValueError(
            f"size budget {size_budget} unreachable: coarsest configuration is "
            f"{used_bytes} bytes (fp16 baseline {total_fp16_bytes})"
        )
    if error_budget is not None and est_error > error_budget:
        raise AssertionError("internal: greedy exceeded the error budget")  # pragma: no cover

    return {
        "version": 1,
        "model": model,
        "target_domain": target,
        "secondary_domains": secondary,
        "formats": formats,
        "signal": signal,
        "lambda": lam,
        "budget": (
            {"mode": "error", "value": error_budget}
            if error_budget is not None
            else {"mode": "size", "value": size_budget}
        ),
        "blocks": {name: {"format": fmt} for name, fmt in state.items()},
        "decisions": decisions,
        "totals": {
            "bytes_fp16": total_fp16_bytes,
            "bytes_final": used_bytes,
            "estimated_error_target": est_error,
        },
    }


def _normalized_contrast(
    estimates: dict[str, dict], target: str, secondary: list[str], lam: float
) -> dict[str, float]:
    """C_g = I_T - lam * mean_S I_S, min-max normalized to [0, 1] over blocks."""
    raw = {}
    for name, e in estimates.items():
        c = e["importance"][target]
        if secondary and lam > 0:
            c -= lam * sum(e["importance"][d] for d in secondary) / len(secondary)
        raw[name] = c
    lo, hi = min(raw.values()), max(raw.values())
    span = hi - lo
    if span <= 0:
        return {name: 0.0 for name in raw}
    return {name: (v - lo) / span for name, v in raw.items()}


def _candidates(
    estimates: dict[str, dict],
    state: dict[str, str],
    target: str,
    contrast: dict[str, float],
    signal: str,
    lam: float,
    allowed_formats: set[str],
) -> list[dict]:
    """All single-block reductions to a strictly coarser, byte-saving format."""
    out = []
    for name, cur in state.items():
        e = estimates[name]
        for fmt in e["nbytes"]:
            if fmt not in allowed_formats or _bits(fmt) >= _bits(cur):
                continue  # not an allowed format, or not coarser than current
            delta_bytes = e["nbytes"][cur] - e["nbytes"][fmt]
            if delta_bytes <= 0:
                continue  # never take a step that saves nothing
            delta_error = e["error"][target][fmt] - e["error"][target][cur]
            penalty = 1.0
            if signal == "contrastive" and lam > 0:
                penalty += lam * contrast[name]
            rho = delta_error * penalty / delta_bytes
            if not math.isfinite(rho):
                continue  # defensive: skip NaN/Inf cost ratios
            out.append(
                {
                    "block": name,
                    "from": cur,
                    "to": fmt,
                    "delta_error": delta_error,
                    "delta_bytes": delta_bytes,
                    "rho": rho,
                }
            )
    return out


def save_plan(plan: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(plan, indent=1, sort_keys=True))


def load_plan(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
