import pytest
import torch

from compress.protect import MixedQuant
from compress.quant_probe import fake_quant

Q0 = "model.layers.0.self_attn.q_proj"
U1 = "model.layers.1.mlp.up_proj"


def _w(model, name):
    return dict(model.named_modules())[name].weight.detach()


def test_unprotected_layers_are_fully_quantized_and_restored(tiny_model):
    before = _w(tiny_model, Q0).clone()
    with MixedQuant(tiny_model, [Q0], 4) as mq:
        assert torch.equal(_w(tiny_model, Q0).float(), fake_quant(before, 4).to(before.dtype).float())
        assert mq.total_weights == before.numel() and mq.protected_weights == 0
    assert torch.equal(_w(tiny_model, Q0), before)


def test_protected_columns_stay_exact_and_scales_use_only_the_rest(tiny_model):
    before = _w(tiny_model, Q0).clone()
    keep = torch.tensor([2, 7, 11])
    rest = [j for j in range(before.shape[1]) if j not in keep.tolist()]
    with MixedQuant(tiny_model, [Q0, U1], 3, {Q0: keep}) as mq:
        now = _w(tiny_model, Q0)
        assert torch.equal(now[:, keep], before[:, keep])
        assert torch.equal(now[:, rest].float(), fake_quant(before[:, rest], 3).to(before.dtype).float())
        assert mq.protected_weights == before.shape[0] * 3
        assert mq.total_weights == before.numel() + _w(tiny_model, U1).numel()
    assert torch.equal(_w(tiny_model, Q0), before)


def test_weights_restored_on_error_and_bad_layer_rolls_back(tiny_model):
    before = _w(tiny_model, Q0).clone()
    with pytest.raises(RuntimeError, match="boom"):
        with MixedQuant(tiny_model, [Q0], 2):
            raise RuntimeError("boom")
    assert torch.equal(_w(tiny_model, Q0), before)
    with pytest.raises(KeyError, match="not an nn.Linear"):
        MixedQuant(tiny_model, [Q0, "model.embed_tokens"], 2).__enter__()
    assert torch.equal(_w(tiny_model, Q0), before)


def test_protecting_everything_changes_nothing(tiny_model):
    before = _w(tiny_model, Q0).clone()
    with MixedQuant(tiny_model, [Q0], 2, {Q0: torch.arange(before.shape[1])}) as mq:
        assert torch.equal(_w(tiny_model, Q0), before) and mq.protected_weights == before.numel()


def test_format_plan_applies_a_format_per_layer_and_protects_columns(tiny_model):
    from compress.protect import FormatPlan, plan_nbytes
    from compress.quant import quantize_weight, storage_nbytes
    before_q, before_u = _w(tiny_model, Q0).clone(), _w(tiny_model, U1).clone()
    keep = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    with FormatPlan(tiny_model, [Q0, U1], {U1: "int8"}, {Q0: keep}):
        rest = [j for j in range(before_q.shape[1]) if j not in keep.tolist()]
        assert torch.equal(_w(tiny_model, Q0)[:, keep], before_q[:, keep])
        assert torch.equal(_w(tiny_model, Q0)[:, rest].float(),
                           quantize_weight(before_q[:, rest], "nf4").dequant.to(before_q.dtype).float())
        assert torch.equal(_w(tiny_model, U1).float(), quantize_weight(before_u, "int8").dequant.to(before_u.dtype).float())
    assert torch.equal(_w(tiny_model, Q0), before_q) and torch.equal(_w(tiny_model, U1), before_u)
    shapes = {Q0: tuple(before_q.shape), U1: tuple(before_u.shape)}
    out, inc = shapes[Q0]
    assert plan_nbytes(shapes, {U1: "int8"}, {Q0: keep}) == (
        out * 8 * 2 + storage_nbytes((out, inc - 8), "nf4") + storage_nbytes(shapes[U1], "int8"))
    assert plan_nbytes(shapes) == sum(storage_nbytes(s, "nf4") for s in shapes.values())
    with FormatPlan(tiny_model, [Q0], {Q0: "fp16"}):
        assert torch.equal(_w(tiny_model, Q0), before_q)


def test_group_quantization_is_finer_than_per_row_and_handles_ragged_widths():
    from compress.quant_probe import fake_quant_group
    torch.manual_seed(0)
    W = torch.randn(8, 150) * torch.tensor([1.0] * 100 + [5.0] * 50)  # outlier columns at the end
    err = lambda q: float((W - q).square().mean())
    assert fake_quant_group(W, 3, 64).shape == W.shape  # 150 = 2 full blocks + a block of 22
    assert err(fake_quant_group(W, 3, 64)) < err(fake_quant(W, 3))  # blocks isolate the outlier columns
    assert err(fake_quant_group(W, 4, 64)) < err(fake_quant_group(W, 3, 64)) < err(fake_quant_group(W, 2, 64))
    # one group per row is the per-row scheme (up to the fp16 rounding of its scale): same error
    e_g, e_r = err(fake_quant_group(W, 8, 150)), err(fake_quant(W, 8))
    assert abs(e_g - e_r) / e_r < 0.1
    with pytest.raises(ValueError, match="group"):
        fake_quant_group(W, 3, 0)


def test_mixed_quant_with_groups_keeps_protected_columns_and_restores(tiny_model):
    from compress.quant_probe import fake_quant_group
    before = _w(tiny_model, U1).clone()  # 32 x 16... in_features = 16
    keep = torch.tensor([0, 1, 2, 3])
    rest = [j for j in range(before.shape[1]) if j not in keep.tolist()]
    with MixedQuant(tiny_model, [U1], 3, {U1: keep}, group=4):
        now = _w(tiny_model, U1)
        assert torch.equal(now[:, keep], before[:, keep])
        assert torch.equal(now[:, rest].float(), fake_quant_group(before[:, rest], 3, 4).to(before.dtype).float())
    assert torch.equal(_w(tiny_model, U1), before)


def test_outlier_columns_fill_the_weight_budget_by_dose_density(tiny_model):
    from compress.protect import outlier_columns
    names = [n for n, m in tiny_model.named_modules() if isinstance(m, torch.nn.Linear) and ".layers." in n]
    modules = dict(tiny_model.named_modules())
    V = {n: torch.rand(modules[n].in_features, dtype=torch.double) + 0.1 for n in names}
    V[names[0]][5] = 1e6  # one column with an enormous dose must be picked first
    budget = 3 * modules[names[0]].weight.shape[0]
    cols = outlier_columns(modules, names, V, 3, 8, budget)
    assert 5 in cols[names[0]].tolist()
    protected = sum(modules[n].weight.shape[0] * c.numel() for n, c in cols.items())
    assert budget <= protected <= budget + max(modules[n].weight.shape[0] for n in names)
