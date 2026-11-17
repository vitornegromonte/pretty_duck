"""GPTQ (Frantar et al. 2022), group-wise, with optional fp16-protected input columns and extensions.

``gptq_quantize`` quantizes one weight matrix given the Hessian ``H = E[x x^T]`` of its inputs: columns
are quantized left to right and each column's rounding error is propagated to the not-yet-quantized
columns through the inverse Hessian.  The default grid matches ``fake_quant_group`` (symmetric absmax
scale per ``group`` consecutive columns of a row), so GPTQ and round-to-nearest differ only in *how*
values are rounded.  Options: ``clip`` (per group and row, the range shrink with the smallest rounding
error), ``asym`` (min/max grid with a zero point) and ``act_order`` (columns by descending Hessian
diagonal; groups form in that order).  Protected columns stay exact and are removed from the problem:
the remaining columns are quantized with the matching sub-Hessian and form their groups among
themselves, exactly as ``MixedQuant`` lays them out.

``GPTQQuant`` applies it to a whole model, sequentially: layer ``i`` is quantized with Hessians measured
on the activations produced by the already-quantized layers ``< i``.  Extras: per-layer bit widths,
a calibration mixture ``(1-alpha) H_general + alpha H_domain``, per-column importance weights (the
Hessian becomes ``s H s`` with ``s = sqrt(weight)``) and a fp16 low-rank correction of the rounding error
whitened by a Hessian.  Weights are restored on exit.  Simulated quantization (nothing is stored).
"""

from __future__ import annotations

import re

import torch


class _Stop(Exception):
    pass


_SHRINK = torch.linspace(1.0, 0.5, 11).tolist()


def _make_quant(w: torch.Tensor, bits: int, asym: bool, p: float):
    """Row-wise grid for the columns ``w`` (out, g), range shrunk by ``p``: returns (quant, params)."""
    if asym:
        maxq = 2 ** bits - 1
        lo, hi = w.amin(1, keepdim=True) * p, w.amax(1, keepdim=True) * p
        scale = ((hi - lo) / maxq).clamp_min(1e-12)
        zero = torch.round(-lo / scale).clamp(0, maxq)
        return (lambda x: (torch.round(x / scale + zero).clamp(0, maxq) - zero) * scale), (scale, zero)
    qmax = 2 ** (bits - 1) - 1
    scale = (p * w.abs().amax(1, keepdim=True) / qmax).clamp_min(1e-12)
    return (lambda x: torch.round(x / scale).clamp(-qmax - 1, qmax) * scale), (scale, torch.zeros_like(scale))


def _fit_grid(w: torch.Tensor, bits: int, asym: bool, clip: bool):
    """Per-row grid for ``w`` (out, g); with ``clip`` the range shrink with the least rounding error per row."""
    if not clip:
        return _make_quant(w, bits, asym, 1.0)[0]
    best_err, best_p = None, None
    for i, p in enumerate(_SHRINK):
        q, _ = _make_quant(w, bits, asym, p)
        err = (w - q(w)).square().sum(1)
        if best_err is None:
            best_err, best_p = err, torch.zeros(w.shape[0], dtype=torch.long, device=w.device)
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_p = torch.where(better, torch.full_like(best_p, i), best_p)
    p_row = torch.tensor(_SHRINK, device=w.device)[best_p].unsqueeze(1)  # (out, 1)
    return _make_quant(w, bits, asym, p_row)[0]


@torch.no_grad()
def gptq_quantize(W: torch.Tensor, H: torch.Tensor, bits: int, group: int = 64,
                  protect: torch.Tensor | None = None, percdamp: float = 0.01,
                  blocksize: int = 128, *, clip: bool = False, asym: bool = False,
                  act_order: bool = False, sink: bool = False) -> torch.Tensor:
    """Return the GPTQ-quantized (dequantized float32) ``W``; ``H`` is (in, in).

    ``sink``: the protected columns are not frozen but kept at full precision *and adjustable*: they come last
    and absorb the rounding error of the quantized columns (continuous values, no rounding), as in SpQR-style
    outlier handling.  Without it they are removed from the problem and stay exactly as they were.
    """
    if bits < 2 or bits > 8:
        raise ValueError(f"bits must be in [2, 8], got {bits}")
    if blocksize % group:
        raise ValueError("blocksize must be a multiple of group")
    out_f, in_f = W.shape
    if H.shape != (in_f, in_f):
        raise ValueError(f"H must be {(in_f, in_f)}, got {tuple(H.shape)}")
    result = W.float().clone()
    keep = torch.ones(in_f, dtype=torch.bool, device=W.device)
    if protect is not None and protect.numel():
        keep[protect.to(W.device)] = False
    rest = keep.nonzero().squeeze(1)
    if rest.numel() == 0:
        return result
    n_q = rest.numel()  # columns that get rounded; with ``sink`` the protected ones follow them unrounded
    if sink and n_q < in_f:
        rest = torch.cat([rest, (~keep).nonzero().squeeze(1)])
    Wr = result[:, rest].clone()
    Hr = H.to(device=W.device, dtype=torch.float32)[rest][:, rest].clone()
    dead = torch.diagonal(Hr) == 0
    Hr[dead, dead] = 1.0
    Wr[:, dead] = 0.0
    if act_order:
        perm = torch.cat([torch.argsort(torch.diagonal(Hr)[:n_q], descending=True),
                          n_q + torch.arange(Hr.shape[0] - n_q, device=Hr.device)])
        Wr, Hr = Wr[:, perm], Hr[perm][:, perm]
    Hr += percdamp * torch.diagonal(Hr).mean() * torch.eye(Hr.shape[0], device=Hr.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hr)), upper=True)
    n = Wr.shape[1]
    Q = torch.zeros_like(Wr)
    for i1 in range(0, n, blocksize):
        i2 = min(i1 + blocksize, n)
        W1, Err = Wr[:, i1:i2].clone(), torch.zeros(out_f, i2 - i1, device=W.device)
        Hinv1 = Hinv[i1:i2, i1:i2]
        quant = None
        for i in range(i2 - i1):
            col = i1 + i
            if col < n_q and i % group == 0:  # grid from the already-updated weights of this group
                quant = _fit_grid(W1[:, i:min(i + group, n_q - i1)], bits, asym, clip)
            w = W1[:, i]
            q = quant(w.unsqueeze(1)).squeeze(1) if col < n_q else w
            Q[:, i1 + i] = q
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err[:, i] = err
        Wr[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    if act_order:
        Q = Q[:, torch.argsort(perm)]
    result[:, rest] = Q
    return result


@torch.no_grad()
def lowrank_correction(E: torch.Tensor, H: torch.Tensor, rank: int, percdamp: float = 0.01) -> torch.Tensor:
    """Rank-``rank`` matrix ``L`` minimising ``tr((E-L) H (E-L)^T)`` (the output error on inputs with Hessian H)."""
    E, H = E.float(), H.to(device=E.device, dtype=torch.float32)
    Hd = H + percdamp * torch.diagonal(H).mean() * torch.eye(H.shape[0], device=H.device)
    evals, U = torch.linalg.eigh(Hd)
    sq = evals.clamp_min(1e-8).sqrt()
    M = E @ (U * sq) @ U.T  # E H^{1/2}
    if min(M.shape) <= 256:
        u, s, vh = torch.linalg.svd(M, full_matrices=False)
        u, s, v = u[:, :rank], s[:rank], vh[:rank].T
    else:
        u, s, v = torch.svd_lowrank(M, q=rank + 8, niter=2)
        u, s, v = u[:, :rank], s[:rank], v[:, :rank]
    return ((u * s) @ v.T) @ ((U / sq) @ U.T)


_LAYER = re.compile(r"^(.*\.layers\.\d+)\.")


class GPTQQuant:
    """Context manager: GPTQ-quantize the named Linears sequentially, decoder layer by decoder layer.

    ``batches`` is a list of tokenizer outputs (``input_ids``, ``attention_mask``).  ``bits`` is an int or
    a layer -> bits map.  ``protect`` maps a layer name to input columns kept at full precision.
    ``domain_batches`` with ``alpha`` mix a second calibration set into the Hessian; ``col_weight`` maps a
    layer to per-input-column importance weights (>= 0); ``rank`` adds a fp16 low-rank correction of the
    rounding error, whitened by the general (``lr_source="gen"``) or domain (``"dom"``) Hessian.
    ``gptq_kwargs`` go to ``quantizer`` (``clip``, ``asym``, ``act_order``).
    """

    def __init__(self, model: torch.nn.Module, batches: list[dict], layers: list[str], bits,
                 protect: dict[str, torch.Tensor] | None = None, group: int = 64, quantizer=None, *,
                 domain_batches: list[dict] | None = None, alpha: float = 0.0,
                 col_weight: dict[str, torch.Tensor] | None = None, rank: int = 0,
                 lr_source: str = "gen", gptq_kwargs: dict | None = None):
        if alpha and not domain_batches:
            raise ValueError("alpha > 0 needs domain_batches")
        if lr_source not in ("gen", "dom") or (lr_source == "dom" and rank and not domain_batches):
            raise ValueError("lr_source must be 'gen' or 'dom' (the latter needs domain_batches)")
        self.quantizer = quantizer or gptq_quantize  # (W, H, bits, group, protect, **kw) -> Q
        self.model, self.batches, self.layers = model, batches, layers
        self.bits, self.group, self.protect = bits, group, protect or {}
        self.domain_batches, self.alpha, self.col_weight = domain_batches, alpha, col_weight or {}
        self.rank, self.lr_source, self.kw = rank, lr_source, gptq_kwargs or {}
        self.total_weights = self.protected_weights = self.lowrank_params = 0
        self._saved: list[tuple[torch.nn.Linear, torch.Tensor]] = []

    def _blocks(self) -> list[tuple[str, list[str]]]:
        by_block: dict[str, list[str]] = {}
        for name in self.layers:
            m = _LAYER.match(name)
            if m is None:
                raise KeyError(f"{name!r} is not inside a numbered decoder layer")
            by_block.setdefault(m.group(1), []).append(name)
        return sorted(by_block.items(), key=lambda kv: int(kv[0].rsplit(".", 1)[1]))

    @torch.no_grad()
    def _hessians(self, block: str, names: list[str], modules: dict, batches: list[dict]) -> dict[str, torch.Tensor]:
        Hs: dict[str, torch.Tensor] = {}
        counts = {n: 0 for n in names}
        state: dict = {}

        def make(name):
            def hook(module, inputs):
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                mask = state["mask"]
                if mask is not None:
                    x = x[mask]
                x = x.float()
                if name not in Hs:
                    Hs[name] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                Hs[name] += x.T @ x
                counts[name] += x.shape[0]
            return hook

        def stop(module, inputs, output):
            raise _Stop

        handles = [modules[n].register_forward_pre_hook(make(n)) for n in names]
        handles.append(modules[block].register_forward_hook(stop))
        device = next(self.model.parameters()).device
        try:
            for b in batches:
                ids, am = b["input_ids"].to(device), b["attention_mask"].to(device)
                state["mask"] = am.reshape(-1).bool()
                try:
                    self.model(input_ids=ids, attention_mask=am, use_cache=False)
                except _Stop:
                    pass
        finally:
            for h in handles:
                h.remove()
        return {n: Hs[n] / max(counts[n], 1) for n in names}

    def _bits(self, name: str) -> int:
        return self.bits[name] if isinstance(self.bits, dict) else self.bits

    @torch.no_grad()
    def __enter__(self) -> "GPTQQuant":
        modules = dict(self.model.named_modules())
        try:
            for block, names in self._blocks():
                Hg = self._hessians(block, names, modules, self.batches)
                Hd = (self._hessians(block, names, modules, self.domain_batches)
                      if self.domain_batches and (self.alpha or (self.rank and self.lr_source == "dom")) else None)
                for name in names:
                    module = modules[name]
                    if not isinstance(module, torch.nn.Linear):
                        raise KeyError(f"layer {name!r} is not an nn.Linear in the model")
                    w = module.weight.data
                    self._saved.append((module, w.cpu().clone()))  # on the host: the GPU may hold only the model
                    self.total_weights += w.numel()
                    keep = self.protect.get(name)
                    if keep is not None and keep.numel():
                        self.protected_weights += w.shape[0] * int(torch.unique(keep).numel())
                    H = Hg[name] if not self.alpha else (1 - self.alpha) * Hg[name] + self.alpha * Hd[name]
                    cw = self.col_weight.get(name)
                    if cw is not None:
                        s = cw.to(H.device, H.dtype).clamp_min(0).sqrt()
                        H = s[:, None] * H * s[None, :]
                    Q = self.quantizer(w, H, self._bits(name), self.group, keep, **self.kw)
                    if self.rank:
                        Hl = Hd[name] if self.lr_source == "dom" else Hg[name]
                        Q = Q + lowrank_correction(w.float() - Q, Hl, self.rank)
                        self.lowrank_params += self.rank * (w.shape[0] + w.shape[1])
                    w.copy_(Q.to(w.dtype))
                del Hg, Hd
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    @torch.no_grad()
    def __exit__(self, *exc) -> None:
        for module, original in reversed(self._saved):
            module.weight.data.copy_(original)
        self._saved.clear()
