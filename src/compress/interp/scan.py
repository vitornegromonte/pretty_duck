"""Activation patching of units, and the ranking of units into circuits.

* sufficiency of a unit set: patch the clean activations into the corrupt run,
  ``restoration = sum(LD_patched - LD_corrupt) / sum(LD_clean - LD_corrupt)``;
* necessity: patch the corrupt activations into the clean run,
  ``destroyed = sum(LD_clean - LD_patched) / sum(LD_clean - LD_corrupt)``.

Forward-only, no gradients; weights are restored on exit.
"""

from __future__ import annotations

import torch

from compress.interp.hooks import SuffixCapture, SuffixPatch, ratio_ci
from compress.interp.minimal_pairs import PairBatch
from compress.interp.units import Unit, unit_sets


class Scanner:
    """Clean/corrupt activations of the write projections for one pair batch."""

    def __init__(self, model: torch.nn.Module, batch: PairBatch, layers: list[str]):
        self.model, self.batch, self.layers = model, batch, layers
        self.device = next(model.parameters()).device
        self.L = batch.ids_clean.shape[1]
        self.ids_clean = batch.ids_clean.to(self.device)
        self.ids_corrupt = batch.ids_corrupt.to(self.device)
        self.tok_c = batch.tok_clean.to(self.device)
        self.tok_k = batch.tok_corrupt.to(self.device)
        model.eval()
        with torch.no_grad():
            with SuffixCapture(model, layers, self.L) as cap:
                self.ld_clean = self._ld(self.ids_clean)
            self.clean_in = cap.inputs
            with SuffixCapture(model, layers, self.L) as cap:
                self.ld_corrupt = self._ld(self.ids_corrupt)
            self.corrupt_in = cap.inputs
        self.den = self.ld_clean - self.ld_corrupt

    def offload(self) -> "Scanner":
        """Move the cached activations to the CPU; patching then copies only the patched slices back."""
        self.clean_in = {k: v.cpu() for k, v in self.clean_in.items()}
        self.corrupt_in = {k: v.cpu() for k, v in self.corrupt_in.items()}
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def _ld(self, ids: torch.Tensor) -> torch.Tensor:
        return self._ld_from(_last_logits(self.model, ids))  # last position only: no (batch, seq, vocab) tensor

    def _ld_from(self, logits: torch.Tensor) -> torch.Tensor:
        """Logit difference from last-position logits; the pair batch may be tiled along dim 0."""
        reps = logits.shape[0] // self.tok_c.shape[0]
        tc, tk = self.tok_c.repeat(reps), self.tok_k.repeat(reps)
        return (logits.gather(1, tc[:, None]) - logits.gather(1, tk[:, None])).squeeze(1)

    @torch.no_grad()
    def patched_ld(self, sets: dict[str, torch.Tensor], source: dict, target_ids: torch.Tensor) -> torch.Tensor:
        with SuffixPatch(self.model, sets, source, self.L):
            return self._ld(target_ids)

    def restoration(self, sets: dict[str, torch.Tensor], n_boot: int = 0, seed: int = 0):
        """Sufficiency (clean into corrupt); returns the point value, or (point, lo, hi)."""
        ld = self.patched_ld(sets, self.clean_in, self.ids_corrupt)
        return self._ratio(ld - self.ld_corrupt, n_boot, seed)

    def necessity(self, sets: dict[str, torch.Tensor], n_boot: int = 0, seed: int = 0):
        """Fraction of the LD destroyed by patching the corrupt activations into the clean run."""
        ld = self.patched_ld(sets, self.corrupt_in, self.ids_clean)
        return self._ratio(self.ld_clean - ld, n_boot, seed)

    def _ratio(self, num: torch.Tensor, n_boot: int, seed: int):
        num, den = num.double().cpu(), self.den.double().cpu()
        if n_boot:
            return ratio_ci(num, den, n_boot, seed)
        return float(num.sum() / den.sum())


def _last_logits(model, ids: torch.Tensor) -> torch.Tensor:
    """Logits of the last position only (avoids a (batch, seq, vocab) tensor when supported)."""
    try:
        return model(input_ids=ids, logits_to_keep=1).logits[:, -1].float()
    except TypeError:
        return model(input_ids=ids).logits[:, -1].float()


@torch.no_grad()
def scan_neurons(scanner: Scanner, layer: str, neurons: list[int], chunk: int = 16,
                 direction: str = "suff") -> list[float]:
    """Single-neuron sufficiency (or necessity) for many neurons of one write projection.

    Each sequence of a chunk carries a *different* patched neuron: the batch is
    ``chunk`` copies of the pair batch, and a pre-hook on ``layer`` overwrites
    input channel ``neurons[u]`` of copy ``u`` with the source activation.  The
    result equals one :meth:`Scanner.restoration` / :meth:`Scanner.necessity`
    call per neuron, in a fraction of the forward passes.
    """
    if direction not in ("suff", "nec"):
        raise ValueError(f"direction must be 'suff' or 'nec', got {direction!r}")
    module = dict(scanner.model.named_modules())[layer]
    src = (scanner.clean_in if direction == "suff" else scanner.corrupt_in)[layer]
    target = scanner.ids_corrupt if direction == "suff" else scanner.ids_clean
    n = target.shape[0]
    den = scanner.den.sum()
    out: list[float] = []
    for i in range(0, len(neurons), chunk):
        js = torch.tensor(neurons[i:i + chunk], device=scanner.device)
        u = len(js)

        def hook(mod, args, js=js, u=u):
            x = args[0].clone()
            xv = x.view(u, n, *x.shape[1:])
            xv[torch.arange(u, device=x.device), :, :, js] = src[:, :, js].permute(2, 0, 1).to(x.dtype)
            return (x, *args[1:])

        handle = module.register_forward_pre_hook(hook)
        try:
            ld = scanner._ld_from(_last_logits(scanner.model, target.repeat(u, 1))).view(u, n)
        finally:
            handle.remove()
        num = (ld - scanner.ld_corrupt[None]).sum(1) if direction == "suff" else (scanner.ld_clean[None] - ld).sum(1)
        out += (num / den).tolist()
    return out


def scan_units(scanner: Scanner, units: list[Unit], log=lambda msg: None) -> list[dict]:
    """Single-unit sufficiency and necessity for every unit."""
    out = []
    for i, u in enumerate(units):
        sets = unit_sets([u])
        out.append({"unit": u, "suff": scanner.restoration(sets), "nec": scanner.necessity(sets)})
        if (i + 1) % 400 == 0 or i + 1 == len(units):
            log(f"scanned {i + 1}/{len(units)} units")
    return out


def top_units(scores: list[dict], k: int, key: str = "suff") -> list[Unit]:
    return [s["unit"] for s in sorted(scores, key=lambda s: -s[key])[:k]]


def core_units(rankings: dict[str, list[dict]], top_k: int, min_domains: int,
               key: str = "suff") -> set[Unit]:
    """Units that sit in the top ``top_k`` of at least ``min_domains`` domains' rankings."""
    counts: dict[Unit, int] = {}
    for scores in rankings.values():
        for u in top_units(scores, top_k, key):
            counts[u] = counts.get(u, 0) + 1
    return {u for u, c in counts.items() if c >= min_domains}


def excluding(scores: list[dict], banned: set[Unit]) -> list[dict]:
    """The scored units that are not in ``banned``."""
    return [s for s in scores if s["unit"] not in banned]
