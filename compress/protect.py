"""Whole-model fake quantization with protected input columns.

``MixedQuant`` quantizes every named Linear to ``bits`` (symmetric per-row,
simulated: nothing is stored) while keeping the chosen input-channel columns of
a layer at full precision.  The per-row scales of a layer with protected
columns are computed over its *unprotected* columns only, as a real
mixed-precision layout would do.  Weights are restored on exit, also on errors.
"""

from __future__ import annotations

import torch

from .quant_probe import fake_quant, fake_quant_group


class MixedQuant:
    def __init__(self, model: torch.nn.Module, layers: list[str], bits: int,
                 protect: dict[str, torch.Tensor] | None = None, group: int | None = None,
                 offload: bool = False):
        self.model, self.layers, self.bits, self.group = model, layers, bits, group
        self.offload = offload  # keep the saved originals on the host (small GPUs)
        self.protect = protect or {}
        self.total_weights = 0
        self.protected_weights = 0
        self._saved: list[tuple[torch.nn.Linear, torch.Tensor]] = []

    def _quant(self, w: torch.Tensor) -> torch.Tensor:
        return (fake_quant_group(w, self.bits, self.group) if self.group else fake_quant(w, self.bits))

    @torch.no_grad()
    def __enter__(self) -> "MixedQuant":
        modules = dict(self.model.named_modules())
        try:
            for name in self.layers:
                module = modules.get(name)
                if not isinstance(module, torch.nn.Linear):
                    raise KeyError(f"layer {name!r} is not an nn.Linear in the model")
                w = module.weight.data
                self._saved.append((module, w.cpu().clone() if self.offload else w.clone()))
                self.total_weights += w.numel()
                keep = self.protect.get(name)
                if keep is None or keep.numel() == 0:
                    w.copy_(self._quant(w).to(w.dtype))
                    continue
                mask = torch.ones(w.shape[1], dtype=torch.bool, device=w.device)
                mask[keep.to(w.device)] = False
                rest = mask.nonzero().squeeze(1)
                self.protected_weights += w.shape[0] * (w.shape[1] - rest.numel())
                if rest.numel():
                    w[:, rest] = self._quant(w[:, rest]).to(w.dtype)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    @torch.no_grad()
    def __exit__(self, *exc) -> None:
        for module, original in reversed(self._saved):
            module.weight.data.copy_(original)
        self._saved.clear()


class FormatPlan:
    """Whole-model fake quantization with a format per layer and fp16-protected columns.

    ``formats`` maps layer name -> one of the allocator's formats (default
    ``default`` for unlisted layers); ``protect`` maps layer name -> input
    columns kept at the original precision.  Rows of a layer with protected
    columns are quantized over the remaining columns only, so block scales
    (nf4) and row scales (int) never see the protected values.  Weights are
    restored on exit, also on errors; nothing is stored.
    """

    def __init__(self, model: torch.nn.Module, layers: list[str], formats: dict[str, str] | None = None,
                 protect: dict[str, torch.Tensor] | None = None, default: str = "nf4"):
        self.model, self.layers = model, layers
        self.formats, self.protect, self.default = formats or {}, protect or {}, default
        self._saved: list[tuple[torch.nn.Linear, torch.Tensor]] = []

    @torch.no_grad()
    def __enter__(self) -> "FormatPlan":
        from .quant import quantize_weight

        modules = dict(self.model.named_modules())
        try:
            for name in self.layers:
                module = modules.get(name)
                if not isinstance(module, torch.nn.Linear):
                    raise KeyError(f"layer {name!r} is not an nn.Linear in the model")
                fmt = self.formats.get(name, self.default)
                w = module.weight.data
                self._saved.append((module, w.clone()))
                if fmt == "fp16":
                    continue
                keep = self.protect.get(name)
                if keep is None or keep.numel() == 0:
                    w.copy_(quantize_weight(w, fmt).dequant.to(w.dtype))
                    continue
                mask = torch.ones(w.shape[1], dtype=torch.bool, device=w.device)
                mask[keep.to(w.device)] = False
                rest = mask.nonzero().squeeze(1)
                if rest.numel():
                    w[:, rest] = quantize_weight(w[:, rest], fmt).dequant.to(w.dtype)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    @torch.no_grad()
    def __exit__(self, *exc) -> None:
        for module, original in reversed(self._saved):
            module.weight.data.copy_(original)
        self._saved.clear()


def plan_nbytes(shapes: dict[str, tuple[int, int]], formats: dict[str, str] | None = None,
                protect: dict[str, torch.Tensor] | None = None, default: str = "nf4") -> int:
    """Exact payload bytes of a :class:`FormatPlan`: protected columns cost fp16, the rest their format."""
    from .quant import storage_nbytes

    total = 0
    for name, (out, inc) in shapes.items():
        fmt = (formats or {}).get(name, default)
        n_keep = 0 if fmt == "fp16" or not protect or name not in protect else int(protect[name].numel())
        total += out * n_keep * 2 + (storage_nbytes((out, inc - n_keep), fmt) if inc > n_keep else 0)
    return total
