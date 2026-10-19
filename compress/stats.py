"""Per-domain activation statistics for calibration data.

For every ``torch.nn.Linear`` module in a model we accumulate, over all *valid*
(non-padding) calibration tokens, the per-input-channel first absolute moment
A[j] = E|h_j|, second moment V[j] = E[h_j^2], and signed mean M[j] = E[h_j].
A and V are exactly the quantities the Wanda-style importance and the diagonal
quantization-error estimator need (see ``quant.py``); M is what the
domain-contrast score needs (see ``contrast.py``).

Axis convention (kept everywhere downstream):
    ``nn.Linear.weight`` has shape (out_features, in_features); the hidden
    state fed to a Linear has shape (..., in_features) and is indexed by j over
    in_features.  A and V are 1-D tensors of length ``in_features``.
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import torch

# Padding positions are excluded from every accumulation.  The mask that
# identifies them is the tokenizer's attention_mask (1 = real token).
_ACC_DTYPE = torch.float64  # accumulate in double to avoid long-sum rounding


def tied_weight_ptrs(model: torch.nn.Module) -> set[int]:
    """data_ptrs of weights two modules share, e.g. a tied lm_head/embedding.

    Such a weight cannot be compressed on its own: rewriting it rewrites the
    other module too, which is what the split between compressed Linears and
    untied embeddings promises not to do.
    """
    seen: dict[int, int] = {}
    for _, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            seen[weight.data_ptr()] = seen.get(weight.data_ptr(), 0) + 1
    return {ptr for ptr, n in seen.items() if n > 1}


def _text_tower_prefix(model: torch.nn.Module) -> str:
    """Full name of the module holding the text decoder ('' means the model).

    Multimodal checkpoints (e.g. Qwen3.5-Base) also carry a vision tower whose
    Linears never run on text; their stats are neither collectable nor useful.
    """
    get_decoder = getattr(model, "get_decoder", None)
    if get_decoder is None:
        return ""
    try:
        text = get_decoder()
    except Exception:
        return ""
    for name, module in model.named_modules():
        if module is text:
            return "" if module is model else name
    return ""


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
        self._sum_signed: dict[str, torch.Tensor] = {}
        self._counts: dict[str, int] = {}

    def observe(self, attention_mask: torch.Tensor | None) -> None:
        """Set the attention mask used by hooks on the next forward pass."""
        self._mask = attention_mask

    def __enter__(self) -> "ActivationCollector":
        tied = tied_weight_ptrs(self.model)
        prefix = _text_tower_prefix(self.model)
        head = getattr(self.model, "get_output_embeddings", lambda: None)()
        device = next(self.model.parameters()).device
        for name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.Linear) or module.weight.data_ptr() in tied:
                continue
            # Skip non-text towers (vision) of multimodal models.
            if prefix and not (
                module is head or name == "lm_head" or name.startswith(prefix + ".")
            ):
                continue
            d = module.in_features
            self._sum_abs[name] = torch.zeros(d, dtype=_ACC_DTYPE, device=device)
            self._sum_sq[name] = torch.zeros(d, dtype=_ACC_DTYPE, device=device)
            self._sum_signed[name] = torch.zeros(d, dtype=_ACC_DTYPE, device=device)
            self._counts[name] = 0
            self._hooks.append(module.register_forward_hook(self._make_hook(name, d)))
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
            self._sum_abs[name] += flat.abs().sum(dim=0)
            self._sum_sq[name] += flat.square().sum(dim=0)
            self._sum_signed[name] += flat.sum(dim=0)
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
                "A": (self._sum_abs[name] / self._counts[name]).cpu(),
                "V": (self._sum_sq[name] / self._counts[name]).cpu(),
                "M": (self._sum_signed[name] / self._counts[name]).cpu(),
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
    # Serialise in memory: the zip writer seeks backwards, and some network
    # filesystems fail that with "unexpected pos" / basic_ios errors.
    buf = io.BytesIO()
    torch.save({"meta": meta, "domains": domains}, buf)
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_bytes(buf.getvalue())
        tmp.replace(path)  # atomic: a failed save never leaves a corrupt stats.pt
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_stats(path: str | Path) -> tuple[dict[str, dict], dict]:
    """Inverse of :func:`save_stats`; returns ``(domains, meta)``."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    return blob["domains"], blob["meta"]
