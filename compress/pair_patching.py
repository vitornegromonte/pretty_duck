"""Hooks for activation patching over the last positions of a sequence.

A clean and a corrupt prompt end in the same tokens, so positions are aligned from the end. ``SuffixCapture``
records the input of chosen Linear layers over the last ``suffix`` positions of the clean run. ``SuffixPatch``
overwrites chosen input channels of the corrupt run with those values. The patched values are real activations of
a real prompt, so the size of the perturbation is matched by construction. Patching every Linear input over the
suffix reproduces the clean run exactly, which is the built-in 100% reference. ``ratio_ci`` gives a ratio of sums
over questions with a bootstrap interval. Forward-only. Both hooks are removed on exit, also on errors.
"""

from __future__ import annotations

import re

import torch


class SuffixCapture:
    """Record the last ``suffix`` positions of every named Linear's input."""

    def __init__(self, model: torch.nn.Module, names: list[str], suffix: int):
        self.model, self.names, self.suffix = model, names, suffix
        self.inputs: dict[str, torch.Tensor] = {}
        self._hooks: list = []

    def __enter__(self) -> "SuffixCapture":
        modules = dict(self.model.named_modules())
        for name in self.names:
            self._hooks.append(modules[name].register_forward_pre_hook(self._make(name)))
        return self

    def _make(self, name: str):
        def hook(module, args):
            self.inputs[name] = args[0][:, -self.suffix:].detach().clone()
        return hook

    def __exit__(self, *exc) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class SuffixPatch:
    """Overwrite chosen input channels of the named Linears over the last ``suffix`` positions."""

    def __init__(self, model: torch.nn.Module, sets: dict[str, torch.Tensor],
                 clean: dict[str, torch.Tensor], suffix: int):
        self.model, self.sets, self.clean, self.suffix = model, sets, clean, suffix
        self._hooks: list = []

    def __enter__(self) -> "SuffixPatch":
        modules = dict(self.model.named_modules())
        try:
            for name, idx in self.sets.items():
                if idx.numel():
                    self._hooks.append(
                        modules[name].register_forward_pre_hook(self._make(name, idx)))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _make(self, name: str, idx: torch.Tensor):
        def hook(module, args):
            x = args[0].clone()
            src = self.clean[name]
            x[:, -self.suffix:, idx.to(x.device)] = src[:, :, idx.to(src.device)].to(device=x.device, dtype=x.dtype)
            return (x, *args[1:])
        return hook

    def __exit__(self, *exc) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def layer_of(name: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def ratio_ci(num: torch.Tensor, den: torch.Tensor, n_boot: int, seed: int):
    """Ratio of sums over questions with a percentile bootstrap CI."""
    g = torch.Generator().manual_seed(seed)
    point = float(num.sum() / den.sum())
    n = len(num)
    boots = torch.empty(n_boot, dtype=torch.float64)
    for b in range(n_boot):
        i = torch.randint(n, (n,), generator=g)
        boots[b] = num[i].sum() / den[i].sum()
    lo, hi = torch.quantile(boots, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return point, lo, hi
