import math

import pytest
import torch

from compress.quant import (
    NF4_BLOCKSIZE,
    NF4_CODEBOOK,
    PORTABLE_FORMATS,
    block_importance,
    dequantize_weight,
    diagonal_error,
    is_available,
    quant_error,
    quantize_weight,
    storage_nbytes,
)


@pytest.fixture()
def W():
    torch.manual_seed(0)
    return torch.randn(16, 48) * 0.5  # 48 is not a multiple of 64: exercises row padding


@pytest.mark.parametrize("fmt", ["fp16", "int8", "int4", "nf4", "bnb_nf4"])
def test_stored_bytes_reproduce_dequant_exactly(W, fmt):
    ok, reason = is_available(fmt)
    if not ok:
        pytest.skip(reason)
    q = quantize_weight(W, fmt)
    extra = q.scale if q.scale is not None else (q.meta or {}).get("state")
    again = dequantize_weight(q.codes, extra, fmt, tuple(W.shape))
    assert torch.allclose(again, q.dequant, atol=1e-6)
    assert tuple(q.dequant.shape) == tuple(W.shape)


def test_fp16_roundtrip_is_exact_for_representable_values():
    W = torch.tensor([[1.0, -2.5, 0.25, 8.0]], dtype=torch.float32)
    q = quantize_weight(W, "fp16")
    assert torch.equal(q.dequant, W)


@pytest.mark.parametrize("fmt", ["int8", "int4", "nf4"])
def test_error_decreases_with_precision(W, fmt):
    fine = quantize_weight(W, "int8" if fmt != "int8" else "fp16")
    coarse = quantize_weight(W, fmt)
    e_fine = quant_error(W, fine).square().mean()
    e_coarse = quant_error(W, coarse).square().mean()
    assert e_coarse >= e_fine  # coarser format is never more accurate


def test_zero_matrix_stays_zero():
    W = torch.zeros(8, 16)
    for fmt in ("int8", "int4", "nf4"):
        q = quantize_weight(W, fmt)
        assert torch.equal(q.dequant, torch.zeros_like(W)), fmt


def test_payload_size_matches_analytic_formula(W):
    for fmt, bits in PORTABLE_FORMATS.items():
        q = quantize_weight(W, fmt)
        assert q.nbytes == storage_nbytes(tuple(W.shape), fmt), fmt
        stored = q.codes.numel() * q.codes.element_size()
        if q.scale is not None:
            stored += q.scale.numel() * q.scale.element_size()
        assert q.nbytes == stored, fmt


def test_bnb_nf4_payload_matches_analytic_formula(W):
    ok, reason = is_available("bnb_nf4")
    if not ok:
        pytest.skip(reason)
    q = quantize_weight(W, "bnb_nf4")
    assert q.nbytes == storage_nbytes(tuple(W.shape), "bnb_nf4")


def test_nf4_chunked_argmin_matches_a_single_pass():
    """The row-chunked codebook comparison must equal the one-shot broadcast."""
    torch.manual_seed(0)
    W = torch.randn(80, 48) * 0.5  # 80 output rows -> more than one 64-row chunk
    q = quantize_weight(W, "nf4")
    nblk = (W.shape[1] + NF4_BLOCKSIZE - 1) // NF4_BLOCKSIZE
    pad = nblk * NF4_BLOCKSIZE - W.shape[1]
    w = torch.nn.functional.pad(W, (0, pad)).reshape(W.shape[0], nblk, NF4_BLOCKSIZE)
    absmax = w.abs().amax(dim=-1)
    scale = torch.where(absmax.half() > 0, absmax.half(), torch.ones_like(absmax)).float()
    ref = (scale[..., None] * NF4_CODEBOOK[
        ((w / scale[..., None]).unsqueeze(-1) - NF4_CODEBOOK).abs().argmin(-1)
    ]).reshape(W.shape[0], nblk * NF4_BLOCKSIZE)[:, : W.shape[1]]
    assert torch.equal(q.dequant, ref)


def test_4bit_formats_are_actually_packed():
    W = torch.randn(4, 64)
    q = quantize_weight(W, "int4")
    assert q.codes.numel() == W.numel() // 2  # two weights per byte
    q = quantize_weight(W, "nf4")
    assert q.codes.numel() == W.numel() // 2
    n_blocks = 4 * ((64 + NF4_BLOCKSIZE - 1) // NF4_BLOCKSIZE)
    assert q.scale.numel() == n_blocks


def test_nf4_codebook_has_exact_zero():
    assert (NF4_CODEBOOK == 0.0).any()


def test_unknown_format_rejected():
    ok, reason = is_available("int3")
    assert not ok and "unknown" in reason
    with pytest.raises(ValueError):
        quantize_weight(torch.randn(2, 2), "int3")


def test_diagonal_error_hand_computed():
    E = torch.ones(2, 3)
    V = torch.tensor([1.0, 2.0, 3.0])
    assert diagonal_error(E, V) == pytest.approx(6.0 * 2)  # sum_ij E^2 V_j = 2*(1+2+3)


def test_block_importance_hand_computed():
    W = torch.tensor([[1.0, -2.0]])
    A = torch.tensor([0.5, 0.25])
    assert block_importance(W, A) == pytest.approx(1.0)


def test_diagonal_error_scales_with_second_moment():
    W = torch.randn(4, 8)
    q = quantize_weight(W, "int4")
    E = quant_error(W, q)
    v_small = torch.full((8,), 1e-6)
    v_big = torch.full((8,), 1e6)
    assert diagonal_error(E, v_big) > diagonal_error(E, v_small) * 1e11
    assert math.isfinite(diagonal_error(E, v_small))
