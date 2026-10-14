"""Per-domain activation statistics for calibration data.

For every ``torch.nn.Linear`` module in a model we accumulate, over all *valid*
(non-padding) calibration tokens, the per-input-channel first absolute moment
A[j] = E|h_j| and second moment V[j] = E[h_j^2].  These are exactly the
quantities the Wanda-style importance and the diagonal quantization-error
estimator need (see ``quant.py``).

Axis convention (kept everywhere downstream):
    ``nn.Linear.weight`` has shape (out_features, in_features); the hidden
    state fed to a Linear has shape (..., in_features) and is indexed by j over
    in_features.  A and V are 1-D tensors of length ``in_features``.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

# Padding positions are excluded from every accumulation.  The mask that
# identifies them is the tokenizer's attention_mask (1 = real token).
_ACC_DTYPE = torch.float64  # accumulate in double to avoid long-sum rounding


class ActivationCollector:
    """Forward-hook collector for per-channel activation moments.

    Usage::

        with ActivationCollector(model) as collector:
            for batch in batches:
                collector.observe(batch["attention_mask"])
                model(**batch)
        stats = collector.layers()   # {name: {"A", "V", "count"}}

    ``observe`` must be called before each forward so the hook knows which
    positions are padding.  Hooks are removed even when the block raises.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._mask: torch.Tensor | None = None
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._sum_abs: dict[str, torch.Tensor] = {}
        self._sum_sq: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}

    def observe(self, attention_mask: torch.Tensor | None) -> None:
        """Set the attention mask used by hooks on the next forward pass."""
        self._mask = attention_mask

    def __enter__(self) -> "ActivationCollector":
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                d = module.in_features
                self._sum_abs[name] = torch.zeros(d, dtype=_ACC_DTYPE)
                self._sum_sq[name] = torch.zeros(d, dtype=_ACC_DTYPE)
                self._counts[name] = 0
                self._hooks.append(
                    module.register_forward_hook(self._make_hook(name, d))
                )
        return self

    def _make_hook(self, name: str, in_features: int):
        def hook(module, inputs, output):
            x = inputs[0].detach()
            if x.shape[-1] != in_features:
                raise RuntimeError(
                    f"hook for {name}: Linear input has last dim "
                    f"{x.shape[-1]}, expected in_features={in_features}"
                )
            flat = x.reshape(-1, in_features)
            # Align the attention mask with the flattened tokens when shapes
            # match (batch, seq); otherwise fall back to "everything is valid"
            # for exotic inputs, which keeps the hook robust.
            if self._mask is not None and self._mask.numel() == x.shape[0] * x.shape[1]:
                flat = flat[self._mask.reshape(-1).bool().to(flat.device)]
            if flat.numel() == 0:
                return  # batch that is entirely padding
            flat = flat.to(_ACC_DTYPE)
            self._sum_abs[name] += flat.abs().sum(dim=0).cpu()
            self._sum_sq[name] += flat.square().sum(dim=0).cpu()
            self._counts[name] += flat.shape[0]

        return hook

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        if exc_type is None:
            idle = [n for n, c in self._counts.items() if c == 0]
            if idle:
                raise RuntimeError(
                    "Linear layers never executed during calibration "
                    f"(no activations observed): {idle}"
                )

    def layers(self) -> dict[str, dict]:
        return {
            name: {
                "A": self._sum_abs[name] / self._counts[name],
                "V": self._sum_sq[name] / self._counts[name],
                "count": self._counts[name],
            }
            for name in self._counts
        }


def collect_domain(
    model: torch.nn.Module,
    tokenizer,
    lines: list[str],
    *,
    batch_size: int = 4,
    max_len: int = 512,
    n_samples: int | None = 128,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, dict]:
    """Collect A/V statistics for one domain from a file of text lines.

    Line selection is deterministic for a given ``seed``.  Only non-empty
    lines count towards ``n_samples``.  The model is put in eval mode; its
    weights are never modified.
    """
    lines = [ln.strip() for ln in lines if ln.strip()]
    if not lines:
        raise ValueError("calibration data contains no non-empty lines")
    rng = random.Random(seed)
    rng.shuffle(lines)
    if n_samples is not None:
        lines = lines[:n_samples]

    device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad(), ActivationCollector(model) as collector:
        for i in range(0, len(lines), batch_size):
            batch = tokenizer(
                lines[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len,
            )
            collector.observe(batch["attention_mask"])
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            model(input_ids=input_ids, attention_mask=attention_mask)
    return collector.layers()


def save_stats(domains: dict[str, dict], path: str | Path, meta: dict) -> None:
    """Persist ``{domain: {layer: {A, V, count}}}`` plus collection metadata."""
    torch.save({"meta": meta, "domains": domains}, Path(path))


def load_stats(path: str | Path) -> tuple[dict[str, dict], dict]:
    """Inverse of :func:`save_stats`; returns ``(domains, meta)``."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    return blob["domains"], blob["meta"]
