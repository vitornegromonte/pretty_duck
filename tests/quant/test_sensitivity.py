import pytest
import torch

from compress.quant.formats import quantize_weight
from compress.quant.sensitivity import QuantizeLayers, measure_plans, measure_sensitivity
from tests.conftest import CALIB_LINES

Q0 = "model.layers.0.self_attn.q_proj"
U1 = "model.layers.1.mlp.up_proj"
EVAL = {"t": CALIB_LINES[:4], "s": CALIB_LINES[4:]}


def test_quantize_layers_uses_the_real_format_and_restores_exactly(tiny_model):
    module = dict(tiny_model.named_modules())[Q0]
    before = module.weight.detach().clone()
    with QuantizeLayers(tiny_model, {Q0: "int4"}):
        assert torch.equal(module.weight.detach().float(),
                           quantize_weight(before, "int4").dequant.to(before.dtype).float())
        assert not torch.equal(module.weight.detach(), before)
    assert torch.equal(module.weight.detach(), before)


def test_weights_restored_when_the_body_raises_or_a_layer_is_bad(tiny_model):
    module = dict(tiny_model.named_modules())[Q0]
    before = module.weight.detach().clone()
    with pytest.raises(RuntimeError, match="boom"):
        with QuantizeLayers(tiny_model, {Q0: "int4"}):
            raise RuntimeError("boom")
    assert torch.equal(module.weight.detach(), before)
    with pytest.raises(KeyError, match="not an nn.Linear"):
        QuantizeLayers(tiny_model, {Q0: "int4", "model.embed_tokens": "int4"}).__enter__()
    assert torch.equal(module.weight.detach(), before)  # the first layer was rolled back


def test_sensitivity_table_shape_and_monotone_in_precision(tiny_model, tokenizer):
    rep = measure_sensitivity(tiny_model, tokenizer, EVAL, [Q0, U1], ["fp16", "int8", "int4"],
                              batch_size=4)
    assert set(rep["layers"]) == {Q0, U1} and set(rep["layers"][Q0]) == {"fp16", "int8", "int4"}
    cell = rep["layers"][Q0]["int4"]["t"]
    assert len(cell["lines"]) == len(EVAL["t"]) and cell["delta"] == pytest.approx(
        sum(cell["lines"]) / len(cell["lines"]))
    for layer in (Q0, U1):  # int4 hurts at least as much as int8, in total
        tot = lambda f: sum(abs(rep["layers"][layer][f][d]["delta"]) for d in EVAL)
        assert tot("int4") >= tot("int8")
    assert all(abs(rep["layers"][Q0]["fp16"][d]["delta"]) < 1e-2 for d in EVAL)


def test_measure_plans_matches_a_single_layer_table_for_one_layer_plans(tiny_model, tokenizer):
    rep = measure_sensitivity(tiny_model, tokenizer, EVAL, [Q0], ["int4"], batch_size=4)
    plans = measure_plans(tiny_model, tokenizer, EVAL, {"one": {Q0: "int4"}, "two": {Q0: "int4", U1: "int4"}},
                          batch_size=4)
    for d in EVAL:
        assert plans["one"][d]["delta"] == pytest.approx(rep["layers"][Q0]["int4"][d]["delta"], abs=1e-6)
    assert plans["two"]["t"]["delta"] != plans["one"]["t"]["delta"]
