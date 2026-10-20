"""Domain-contrast weight masks, the Persona-style sub-network probe.

The score for input channel ``j`` of a Linear is the signed standardized
activation difference between two calibration domains::

    z_j = (M_seek[j] - M_reject[j]) / sqrt(var_seek[j] + var_reject[j] + eps)

with ``var = V - M^2`` (``M`` is the signed mean added to ``stats.py``).  As in
Persona, the target-domain mask keeps the top ``(1 - sparsity)`` fraction of
``|W_ij| * relu(z_j)`` in each output row; the inverse mask uses
``relu(-z_j)``, i.e. the weights associated with the other domain.  The
``contrast_abs`` variant uses ``|z_j|`` (Persona's sparse mask), and
``magnitude``/``wanda``/``random`` are controls.  Weights are zeroed in place,
layer by layer, so no full-model mask tensor is ever materialized.
"""

from __future__ import annotations

import torch

from .stats import tied_weight_ptrs

EPS = 1e-6
MASK_KINDS = ("none", "contrast", "contrast_abs", "inverse", "magnitude", "wanda", "random")
MLP_MARKERS = ("gate_proj", "up_proj", "down_proj", ".mlp.")


def contrast_scores(stats: dict, seek: str, reject: str) -> dict[str, torch.Tensor]:
    """Signed per-input-channel contrast ``z`` for every layer in both domains."""
    if seek not in stats or reject not in stats:
        missing = [name for name in (seek, reject) if name not in stats]
        raise ValueError(
            f"stats has no domain(s) {missing}; available: {sorted(stats)}. "
            "Collect them with --calib math=... --calib law=..."
        )
    missing = [name for name in stats[seek] if name not in stats[reject]]
    if missing:
        raise ValueError(f"domains were collected on different layers: {missing[:3]} ...")
    scores = {}
    for name, s in stats[seek].items():
        if "M" not in s or "M" not in stats[reject][name]:
            raise ValueError(
                f"stats for {name} lack signed means M; recollect with the current "
                "collect_stats (old stats only have A and V)"
            )
        r = stats[reject][name]
        var_s = torch.clamp(s["V"] - s["M"].square(), min=0.0)
        var_r = torch.clamp(r["V"] - r["M"].square(), min=0.0)
        scores[name] = ((s["M"] - r["M"]) / torch.sqrt(var_s + var_r + EPS)).float()
    return scores


def _keep_per_row(score: torch.Tensor, sparsity: float) -> torch.Tensor:
    in_features = score.shape[1]
    keep_k = max(1, min(in_features, int(in_features * (1.0 - sparsity))))
    mask = torch.zeros_like(score, dtype=torch.bool)
    idx = torch.topk(score, k=keep_k, dim=1).indices
    mask.scatter_(1, idx, True)
    return mask


def apply_domain_mask(
    model: torch.nn.Module,
    stats: dict,
    *,
    kind: str = "none",
    seek: str | None = None,
    reject: str | None = None,
    sparsity: float = 0.3,
    seed: int = 0,
    only_mlp: bool = False,
) -> dict:
    """Zero weights in place and return a summary of what was masked.

    ``stats`` is the loaded ``domains`` dict from ``collect_domain``.  Only
    layers present in the stats file are touched, which excludes tied
    embeddings/heads, the vision tower, and any layer outside the text decoder.
    """
    if kind not in MASK_KINDS:
        raise ValueError(f"unknown mask kind {kind!r}; choose from {MASK_KINDS}")
    if kind == "none":
        return {
            "kind": "none", "layers": 0, "weights": 0, "kept": 0,
            "kept_fraction": 1.0, "only_mlp": only_mlp,
        }
    if not (0.0 <= sparsity < 1.0):
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    if not seek or not reject:
        raise ValueError("masking requires both --mask-seek and --mask-reject domains")
    missing = [name for name in (seek, reject) if name not in stats]
    if missing:
        raise ValueError(
            f"stats has no domain(s) {missing}; available: {sorted(stats)}. "
            "Collect them with --calib math=... --calib law=..."
        )

    z = None
    if kind in ("contrast", "contrast_abs", "inverse"):
        z = contrast_scores(stats, seek, reject)
        names = list(z)
    else:
        names = list(stats[seek])

    modules = dict(model.named_modules())
    tied = tied_weight_ptrs(model)
    generator = torch.Generator(device=next(model.parameters()).device)
    generator.manual_seed(seed)
    layers = weights = kept = 0
    for name in names:
        if name.endswith("lm_head"):
            continue
        if only_mlp and not any(marker in name for marker in MLP_MARKERS):
            continue
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise KeyError(f"stats layer {name!r} is not a Linear in this model")
        if module.weight.data_ptr() in tied:
            continue
        weight = module.weight.detach()
        if weight.ndim != 2:
            raise ValueError(f"{name}: expected a 2-D Linear weight, got {tuple(weight.shape)}")

        if kind in ("contrast", "contrast_abs", "inverse"):
            channel = z[name].to(weight.device)
            if kind == "contrast":
                channel = torch.relu(channel)
            elif kind == "contrast_abs":
                channel = channel.abs()
            else:
                channel = torch.relu(-channel)
            score = weight.abs().float() * channel.unsqueeze(0)
        elif kind == "magnitude":
            score = weight.abs().float()
        elif kind == "wanda":
            # Domain-agnostic activation-aware control: mean calibration norm.
            v = (stats[seek][name]["V"] + stats[reject][name]["V"]) / 2.0
            score = weight.abs().float() * torch.sqrt(torch.clamp(v, min=0.0) + EPS).float().to(weight.device).unsqueeze(0)
        else:  # random
            score = torch.rand(weight.shape, generator=generator, device=weight.device)

        mask = _keep_per_row(score, sparsity)
        module.weight.data.mul_(mask.to(weight.dtype))
        layers += 1
        weights += weight.numel()
        kept += int(mask.sum())

    return {
        "kind": kind,
        "seek": seek,
        "reject": reject,
        "sparsity": sparsity,
        "seed": seed,
        "layers": layers,
        "weights": weights,
        "kept": kept,
        "kept_fraction": kept / weights if weights else 1.0,
        "only_mlp": only_mlp,
    }
