import pytest
import torch

from compress.pair_patching import SuffixCapture, SuffixPatch, layer_of, ratio_ci


def test_layer_of_reads_the_decoder_layer_index():
    assert layer_of("model.layers.12.mlp.up_proj") == 12 and layer_of("lm_head") is None


def test_patch_hooks_removed_when_the_body_raises(tiny_model, tokenizer):
    ids = tokenizer(["w1 w2 w3 w4"], return_tensors="pt")["input_ids"]
    before = tiny_model(input_ids=ids).logits.detach()
    name = "model.layers.0.self_attn.q_proj"
    clean = {name: torch.zeros(1, 2, 16)}
    with pytest.raises(RuntimeError, match="boom"):
        with SuffixPatch(tiny_model, {name: torch.arange(8)}, clean, suffix=2):
            raise RuntimeError("boom")
    assert torch.equal(tiny_model(input_ids=ids).logits, before)
    with SuffixCapture(tiny_model, [name], suffix=2) as cap:
        tiny_model(input_ids=ids)
    assert cap.inputs[name].shape == (1, 2, 16)


def test_ratio_ci_brackets_the_ratio_of_sums():
    num, den = torch.tensor([1.0, 2.0, 3.0, 2.0]), torch.tensor([2.0, 2.0, 4.0, 2.0])
    point, lo, hi = ratio_ci(num, den, n_boot=200, seed=0)
    assert point == pytest.approx(8.0 / 10.0) and lo <= point <= hi
