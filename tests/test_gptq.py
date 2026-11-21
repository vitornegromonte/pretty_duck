import torch

from compress.gptq import GPTQQuant, gptq_quantize
from compress.quant_probe import fake_quant_group
from tests.conftest import CALIB_LINES


def _problem(seed=0, out_f=24, in_f=128):
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(in_f, in_f, generator=g) * 0.3 + torch.eye(in_f)
    X = torch.randn(2000, in_f, generator=g) @ mix  # correlated inputs
    return torch.randn(out_f, in_f, generator=g), X.T @ X / X.shape[0], X


def _out_err(W, Q, X):
    return float(((X @ (W - Q).T) ** 2).mean())


def test_gptq_beats_round_to_nearest_on_output_error():
    W, H, X = _problem()
    rtn = fake_quant_group(W, 3, 64)
    gptq = gptq_quantize(W, H, 3, 64)
    assert _out_err(W, gptq, X) < 0.8 * _out_err(W, rtn, X)


def test_gptq_stays_on_the_group_grid():
    W, H, _ = _problem(1)
    Q = gptq_quantize(W, H, 4, 64)
    levels = Q.reshape(Q.shape[0], -1, 64)
    assert all(len(torch.unique(levels[i, j])) <= 16 for i in range(levels.shape[0]) for j in range(levels.shape[1]))


def test_protected_columns_are_exact_and_the_rest_is_optimized_without_them():
    W, H, X = _problem(2)
    keep = torch.tensor([3, 40, 77, 100])
    Q = gptq_quantize(W, H, 3, 64, keep)
    assert torch.equal(Q[:, keep], W[:, keep])
    rest = torch.tensor([j for j in range(W.shape[1]) if j not in keep.tolist()])
    rtn = fake_quant_group(W[:, rest], 3, 64)
    assert _out_err(W[:, rest], Q[:, rest], X[:, rest]) < _out_err(W[:, rest], rtn, X[:, rest])
    assert torch.equal(gptq_quantize(W, H, 3, 64, torch.arange(W.shape[1])), W)


def test_model_gptq_changes_weights_and_restores_them(tiny_model, tokenizer):
    names = [n for n, m in tiny_model.named_modules()
             if isinstance(m, torch.nn.Linear) and ".layers." in n]
    batches = [tokenizer(CALIB_LINES[i:i + 4], return_tensors="pt", padding=True) for i in (0, 4)]
    before = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    keep = {names[0]: torch.tensor([1, 2])}
    with GPTQQuant(tiny_model, batches, names, 4, keep, group=8) as q:
        mod = dict(tiny_model.named_modules())
        assert not torch.equal(mod[names[1]].weight, before[names[1] + ".weight"])
        assert torch.equal(mod[names[0]].weight[:, [1, 2]], before[names[0] + ".weight"][:, [1, 2]])
        assert q.protected_weights == 2 * mod[names[0]].weight.shape[0]
        assert q.total_weights == sum(mod[n].weight.numel() for n in names)
    assert all(torch.equal(p, before[n]) for n, p in tiny_model.named_parameters())


def test_grid_options_keep_the_grid_size_and_clipping_never_hurts_rounding_error():
    from compress.gptq import _fit_grid
    W, H, X = _problem(3)
    w = W[:, :64]
    for asym in (False, True):
        plain, clipped = _fit_grid(w, 3, asym, False), _fit_grid(w, 3, asym, True)
        assert float((w - clipped(w)).square().sum()) <= float((w - plain(w)).square().sum()) + 1e-6
        assert all(len(torch.unique(clipped(w)[i])) <= 8 for i in range(w.shape[0]))
    for kw in ({"clip": True}, {"asym": True}, {"act_order": True}, {"clip": True, "asym": True, "act_order": True}):
        Q = gptq_quantize(W, H, 3, 64, **kw)
        assert Q.shape == W.shape and _out_err(W, Q, X) < _out_err(W, fake_quant_group(W, 3, 64), X)
    keep = torch.tensor([5, 60])
    assert torch.equal(gptq_quantize(W, H, 3, 64, keep, act_order=True)[:, keep], W[:, keep])


def test_lowrank_correction_is_optimal_monotone_and_exact_at_full_rank():
    from compress.gptq import lowrank_correction
    W, H, X = _problem(4, out_f=16, in_f=64)
    E = W - fake_quant_group(W, 2, 64)
    errs = [_out_err(E, lowrank_correction(E, H, r), X) for r in (0, 4, 16)]
    base = _out_err(E, torch.zeros_like(E), X)
    assert errs[0] == base and errs[0] > errs[1] > errs[2]
    assert errs[2] < 0.02 * base  # rank 16 = full rank of a 16x64 matrix (up to damping)
    assert _out_err(E, lowrank_correction(E, H, 4), X) < _out_err(E, lowrank_correction(E, torch.eye(64), 4), X)


def test_calibration_mixture_column_weights_bits_and_rank_in_the_model_wrapper(tiny_model, tokenizer):
    names = [n for n, m in tiny_model.named_modules() if isinstance(m, torch.nn.Linear) and ".layers." in n]
    gen = [tokenizer(CALIB_LINES[i:i + 4], return_tensors="pt", padding=True) for i in (0, 4)]
    dom = [tokenizer(CALIB_LINES[i:i + 4][::-1], return_tensors="pt", padding=True) for i in (0, 4)]
    before = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    mods = dict(tiny_model.named_modules())
    out = {}
    for tag, kw in {"plain": {}, "mix": {"domain_batches": dom, "alpha": 0.5},
                    "weighted": {"col_weight": {names[0]: torch.arange(mods[names[0]].in_features).float() + 1}},
                    "rank": {"domain_batches": dom, "rank": 2, "lr_source": "dom"}}.items():
        with GPTQQuant(tiny_model, gen, names, {n: 3 + (i % 2) for i, n in enumerate(names)}, None, 8, **kw) as q:
            out[tag] = mods[names[0]].weight.detach().clone()
            if tag == "rank":
                assert q.lowrank_params == sum(2 * sum(mods[n].weight.shape) for n in names)
        assert all(torch.equal(p, before[n]) for n, p in tiny_model.named_parameters())
    assert not torch.equal(out["plain"], out["weighted"]) and not torch.equal(out["plain"], out["rank"])
    import pytest
    with pytest.raises(ValueError):
        GPTQQuant(tiny_model, gen, names, 4, alpha=0.5)
