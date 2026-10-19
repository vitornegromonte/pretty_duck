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
from .stats import tied_weight_ptrs


def estimate_blocks(
    weights,
    domains_stats: dict[str, dict],
    formats: list[str],
) -> dict[str, dict]:
    """Per-Linear shape, importance and diagonal quantization error.

    ``weights`` is an iterable of ``(layer_name, weight_tensor)`` pairs, so
    callers can stream tensors from a safetensors file one layer at a time
    instead of materializing a full model (the low-RAM path used by the CLI).

    Returns ``{layer_name: {"shape": (o, i), "nbytes": {fmt: int},
    "importance": {domain: float}, "error": {domain: {fmt: float}}}}``.
    """
    for fmt in formats:
        ok, reason = is_available(fmt)
        if not ok:
            raise ValueError(reason)

    _validate_stats_coverage(domains_stats)
    estimates: dict[str, dict] = {}
    for name, weight in weights:
        if weight.dim() != 2:
            raise ValueError(f"{name}: expected a 2-D Linear weight, got {weight.dim()}-D")
        W = weight.detach().float()
        # Stats are stored on CPU; a model loaded on GPU would otherwise mix
        # devices in diagonal_error/block_importance.
        moments = {
            domain: (dstats[name]["A"].to(W.device), dstats[name]["V"].to(W.device))
            for domain, dstats in domains_stats.items()
        }
        errors = {d: {} for d in domains_stats}
        for fmt in formats:  # quantize once per format, reuse across domains
            q = quantize_weight(W, fmt)
            E = quant_error(W, q)
            for domain, (_, V) in moments.items():
                errors[domain][fmt] = diagonal_error(E, V)
        estimates[name] = {
            "shape": tuple(W.shape),
            "nbytes": {fmt: storage_nbytes(tuple(W.shape), fmt) for fmt in formats},
            "importance": {
                domain: block_importance(W, A) for domain, (A, _) in moments.items()
            },
            "error": errors,
        }
    missing = _stats_names(domains_stats) - set(estimates)
    if missing:
        raise KeyError(f"weights iterator did not yield layers: {sorted(missing)[:5]}")
    return estimates


def estimate_model_blocks(model, domains_stats: dict[str, dict], formats: list[str]) -> dict[str, dict]:
    """Convenience wrapper of :func:`estimate_blocks` for an in-memory model.

    Tied weights (e.g. a tied ``lm_head``) are skipped, matching the stats
    collector; they are not compressed independently.
    """
    tied = tied_weight_ptrs(model)
    linears = (
        (name, module.weight) for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and module.weight.data_ptr() not in tied
    )
    return estimate_blocks(linears, domains_stats, formats)


def _stats_names(domains_stats: dict[str, dict]) -> set[str]:
    return set(next(iter(domains_stats.values())))


def _validate_stats_coverage(domains_stats: dict[str, dict]) -> None:
    """Every domain must carry stats for exactly the same layers."""
    names = _stats_names(domains_stats)
    for domain, dstats in domains_stats.items():
        if set(dstats) != names:
            raise KeyError(f"domain {domain!r} layers differ from the other domains")


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
    contrast = (
        _normalized_contrast(estimates, target, secondary, lam)
        if signal == "contrastive" else {}
    )

    state = {name: "fp16" for name in estimates}
    used_bytes = sum(e["nbytes"]["fp16"] for e in estimates.values())
    total_fp16_bytes = used_bytes
    # fp16 is not error-free: start from its measured baseline so the reported
    # total is sum(error at the chosen formats), not the sum of the deltas.
    est_error = sum(e["error"][target]["fp16"] for e in estimates.values())
    if error_budget is not None and est_error > error_budget:
        raise ValueError(
            f"error budget {error_budget} unreachable: the fp16 baseline error "
            f"is already {est_error}"
        )
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
