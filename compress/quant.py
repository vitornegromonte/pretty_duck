"""Quantization formats, reconstruction error, and block importance.

All formats operate on a single ``nn.Linear.weight`` of shape
(out_features, in_features).  Quantization is *simulated* for error
estimation: we always reconstruct a float tensor, never claim low-bit
kernels or low-bit inference speedups.

Two families of formats:

* Portable packed formats (pure PyTorch): ``fp16``, ``int8``, ``int4``,
  ``nf4``.  ``int4``/``nf4`` store two weights per byte; their byte counts
  are exact payloads that ``serialize.py`` writes to disk.
* ``bnb_nf4``: bitsandbytes' own NF4 representation (requires the
  ``bitsandbytes`` package; the functional API works on CPU and CUDA).
  We store its packed codes and quantization state verbatim, so its disk
  footprint is bitsandbytes' real footprint.

Honesty rules encoded here:
* ``quantize_weight`` returns the *stored* codes/scales plus the float
  reconstruction; anything downstream must use ``QuantizedTensor.dequant``
  for inference, not the original weights.
* :func:`storage_nbytes` is the analytic payload size used by the
  allocator; final reports must use the measured file sizes produced by
  ``serialize.save_compressed``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# NF4 lookup table: quantiles of N(0, 1) with equal probability mass per bin.
# Verbatim from bitsandbytes 0.50 ``get_4bit_type('nf4')`` (QLoRA, Dettmers et al.).
NF4_CODEBOOK = torch.tensor(
    [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.3379152417182224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0,
    ],
    dtype=torch.float32,
)

# Portable formats and their nominal bits per weight.
PORTABLE_FORMATS: dict[str, int] = {"fp16": 16, "int8": 8, "int4": 4, "nf4": 4}
OPTIONAL_FORMATS: dict[str, int] = {"bnb_nf4": 4}
ALL_FORMATS = {**PORTABLE_FORMATS, **OPTIONAL_FORMATS}

# Scale granularity for nf4 (and bnb_nf4): number of consecutive in_features
# weights sharing one absmax scale, matching bitsandbytes' default blocksize.
NF4_BLOCKSIZE = 64

_SCALE_BYTES = 2  # every scale vector is stored as float16


@dataclass
class QuantizedTensor:
    """Result of quantizing one weight matrix.

    ``codes`` and ``scale`` are the stored representation (already packed for
    4-bit formats); ``dequant`` is the float32 reconstruction that inference
    must use; ``nbytes`` is the exact payload size of codes+scale.
    ``meta`` carries format-specific extras (the bitsandbytes QuantState for
    ``bnb_nf4``).
    """

    format: str
    codes: torch.Tensor
    scale: torch.Tensor | None
    dequant: torch.Tensor
    nbytes: int
    meta: dict | None = None


def is_available(fmt: str) -> tuple[bool, str]:
    """Return ``(ok, reason)``; ``ok=False`` means the format cannot be used."""
    if fmt not in ALL_FORMATS:
        return False, f"unknown format {fmt!r}; known: {sorted(ALL_FORMATS)}"
    if fmt in PORTABLE_FORMATS:
        return True, ""
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        return False, f"{fmt} requires the bitsandbytes package"
    return True, ""


def storage_nbytes(shape: tuple[int, int], fmt: str) -> int:
    """Analytic payload bytes (codes + scales) for a weight of ``shape``.

    Used by the allocator to reason about sizes; the authoritative number for
    reports is the measured on-disk size from ``serialize.save_compressed``.
    """
    out, inc = shape
    n = out * inc
    if fmt == "fp16":
        return n * 2
    if fmt == "int8":
        return n + out * _SCALE_BYTES
    if fmt == "int4":
        return (n + 1) // 2 + out * _SCALE_BYTES
    if fmt == "nf4":
        n_blocks = out * ((inc + NF4_BLOCKSIZE - 1) // NF4_BLOCKSIZE)
        # packed nibbles (row-tail padded to a full block) + absmax per block
        return n_blocks * NF4_BLOCKSIZE // 2 + n_blocks * _SCALE_BYTES
    if fmt == "bnb_nf4":
        n_blocks = (n + NF4_BLOCKSIZE - 1) // NF4_BLOCKSIZE
        # bitsandbytes 0.50 keeps absmax and the codebook in float32 when
        # quantizing without double quantization (verified on CPU).  This is a
        # best-effort estimate for reporting only -- the allocator refuses
        # bnb formats and reports always prefer measured manifest sizes.
        return (n + 1) // 2 + n_blocks * 4 + NF4_CODEBOOK.numel() * 4
    raise ValueError(f"no storage formula for format {fmt!r}")


# ---------------------------------------------------------------------------
# packing helpers (4-bit formats store two signed weights per byte)
# ---------------------------------------------------------------------------

def _pack_nibbles(codes0to15: torch.Tensor) -> torch.Tensor:
    """Pack a tensor of values in [0, 15] into uint8, low nibble = even index."""
    flat = codes0to15.reshape(-1)
    if flat.numel() % 2:
        flat = torch.cat([flat, flat.new_zeros(1)])
    lo = flat[0::2].to(torch.int16)
    hi = flat[1::2].to(torch.int16)
    return ((hi << 4) | lo).to(torch.uint8)


def _unpack_nibbles(packed: torch.Tensor, n: int, offset: int = 0) -> torch.Tensor:
    """Inverse of :func:`_pack_nibbles`; ``offset`` shifts into the value range."""
    lo = (packed & 0x0F).to(torch.int16)
    hi = ((packed >> 4) & 0x0F).to(torch.int16)
    flat = torch.empty(packed.numel() * 2, dtype=torch.int16, device=packed.device)
    flat[0::2], flat[1::2] = lo, hi
    return (flat[:n] - offset).to(torch.int8)


# ---------------------------------------------------------------------------
# per-format quantize / dequantize
# ---------------------------------------------------------------------------

def _quantize_int(W: torch.Tensor, bits: int) -> QuantizedTensor:
    """Symmetric per-output-channel (row-wise) integer quantization."""
    qmax = 2 ** (bits - 1) - 1  # 127 or 7
    w = W.float()
    # Quantize against the *stored* (float16) scale so the reconstruction is
    # exactly reproducible from the packed bytes.  Zero rows fall back to
    # scale 1 (their codes are 0, so the reconstruction stays exact).
    scale = (w.abs().amax(dim=1) / qmax).half()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale)).float()
    codes = torch.round(w / scale[:, None]).clamp(-qmax - 1, qmax).to(torch.int8)
    dequant = codes.float() * scale[:, None]
    if bits == 8:
        packed, nbytes = codes, codes.numel() + scale.numel() * _SCALE_BYTES
    else:  # int4: store two per byte
        packed = _pack_nibbles((codes.to(torch.int16) + 8).to(torch.uint8))
        nbytes = packed.numel() + scale.numel() * _SCALE_BYTES
    return QuantizedTensor("int8" if bits == 8 else "int4", packed, scale.half(), dequant, nbytes)


def _quantize_nf4_portable(W: torch.Tensor) -> QuantizedTensor:
    """Portable NF4: per-row blocks of ``NF4_BLOCKSIZE`` share one absmax."""
    out, inc = W.shape
    nblk = (inc + NF4_BLOCKSIZE - 1) // NF4_BLOCKSIZE
    pad = nblk * NF4_BLOCKSIZE - inc
    w = torch.nn.functional.pad(W.float(), (0, pad)).reshape(out, nblk, NF4_BLOCKSIZE)
    absmax = w.abs().amax(dim=-1)
    scale = absmax.half()
    # Sub- fp16-normal absmax (or a zero block) falls back to scale 1; codes
    # then point at the codebook's exact 0.0 entry and the error stays < 1e-7.
    scale = torch.where(scale > 0, scale, torch.ones_like(scale)).float()
    norm = w / scale[..., None]
    cb = NF4_CODEBOOK.to(W.device)
    # Compare in output-row chunks: the full (out, blocks, 64, 16) broadcast
    # would cost 16x the weight in memory on large layers.
    codes = torch.empty_like(norm, dtype=torch.uint8)
    for start in range(0, out, 64):
        stop = min(start + 64, out)
        codes[start:stop] = (
            norm[start:stop].unsqueeze(-1) - cb
        ).abs().argmin(dim=-1).to(torch.uint8)
    packed = _pack_nibbles(codes)
    dequant = (scale[..., None] * cb[codes.long()]).reshape(out, nblk * NF4_BLOCKSIZE)[:, :inc]
    nbytes = packed.numel() + scale.numel() * _SCALE_BYTES
    return QuantizedTensor("nf4", packed, scale.half(), dequant, nbytes)


def _quantize_bnb_nf4(W: torch.Tensor) -> QuantizedTensor:
    """Bitsandbytes NF4 via its functional API (CPU and CUDA both work)."""
    import bitsandbytes.functional as F

    q, state = F.quantize_4bit(
        W.float().contiguous(), blocksize=NF4_BLOCKSIZE, quant_type="nf4"
    )
    dequant = F.dequantize_4bit(q, state).float()
    nbytes = q.numel() * q.element_size()
    for v in vars(state).values():
        if torch.is_tensor(v):
            nbytes += v.numel() * v.element_size()
    # The QuantState is pickled verbatim: the stored format stays bitsandbytes',
    # not ours, and reconstruction uses their own dequantize path.
    return QuantizedTensor("bnb_nf4", q, None, dequant, nbytes, meta={"state": state})


def quantize_weight(W: torch.Tensor, fmt: str) -> QuantizedTensor:
    """Quantize one (out_features, in_features) weight matrix in format ``fmt``."""
    ok, reason = is_available(fmt)
    if not ok:
        raise ValueError(reason)
    if fmt == "fp16":
        codes = W.half()
        return QuantizedTensor("fp16", codes, None, codes.float(), codes.numel() * 2)
    if fmt == "int8":
        return _quantize_int(W, 8)
    if fmt == "int4":
        return _quantize_int(W, 4)
    if fmt == "nf4":
        return _quantize_nf4_portable(W)
    if fmt == "bnb_nf4":
        return _quantize_bnb_nf4(W)
    raise ValueError(fmt)


def dequantize_weight(
    codes: torch.Tensor, scale_or_state, fmt: str, shape: tuple[int, int]
) -> torch.Tensor:
    """Reconstruct the float32 weight from its *stored* representation.

    ``scale_or_state`` is the scale vector for portable formats, or the
    pickled bitsandbytes QuantState for ``bnb_nf4``.
    """
    if fmt == "fp16":
        return codes.float()
    if fmt == "int8":
        return codes.float() * scale_or_state.float()[:, None]
    if fmt == "int4":
        out, inc = shape
        signed = _unpack_nibbles(codes, out * inc, offset=8)
        return signed.float().reshape(out, inc) * scale_or_state.float()[:, None]
    if fmt == "nf4":
        out, inc = shape
        nblk = (inc + NF4_BLOCKSIZE - 1) // NF4_BLOCKSIZE
        idx = _unpack_nibbles(codes, out * nblk * NF4_BLOCKSIZE).long()
        idx = idx.reshape(out, nblk, NF4_BLOCKSIZE)
        cb = NF4_CODEBOOK.to(codes.device)
        return (scale_or_state.float()[..., None] * cb[idx]).reshape(out, nblk * NF4_BLOCKSIZE)[:, :inc]
    if fmt == "bnb_nf4":
        import bitsandbytes.functional as F

        return F.dequantize_4bit(codes, scale_or_state).float()
    raise ValueError(f"unknown format {fmt!r}")


# ---------------------------------------------------------------------------
# error estimation and importance (REQ §6-7, TASK §6-7)
# ---------------------------------------------------------------------------

def quant_error(W: torch.Tensor, q: QuantizedTensor) -> torch.Tensor:
    """E = W - W_q in float32."""
    return W.float() - q.dequant


def diagonal_error(E: torch.Tensor, V: torch.Tensor) -> float:
    """Cheap diagonal estimate D = sum_ij E_ij^2 * V_j.

    Interpreted as E[||x E^T||^2] under the assumption that input channels are
    uncorrelated (V = E[h_j^2] from stats.py).  Ignores cross-channel
    correlation; a local approximation, not the final model loss.
    """
    return float((E.double().square() * V.double()).sum().item())


def block_importance(W: torch.Tensor, A: torch.Tensor) -> float:
    """I_g = sum_ij |W_ij| * A_j with A_j = E|h_j| (Wanda-style aggregate)."""
    return float((W.double().abs() * A.double()).sum().item())
