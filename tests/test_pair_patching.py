import pytest
import torch

from compress.pair_patching import (
    SuffixCapture,
    SuffixPatch,
    common_suffix,
    encode_pair,
    layer_of,
    pair_patch_probe,
)
from compress.patching import select_channels

CLEAN, CORRUPT = "w1 w2", "w3 w4 w5"
ROWS = [
    {"question": "w6 w7", "A": "w8 w9", "B": "w10 w11 w12", "matching": "A"},
    {"question": "w13 w14 w15", "A": "w16", "B": "w17 w18", "matching": "B"},
    {"question": "w19", "A": "w20 w21", "B": "w22", "matching": "A"},
]


def _linear_names(model, with_head=False):
    return [n for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and (with_head or n != "lm_head")]


def _all_channels(model, names):
    modules = dict(model.named_modules())
    return {n: torch.arange(modules[n].in_features) for n in names}


def test_common_suffix_and_layer_of():
    assert common_suffix([1, 2, 3, 4], [9, 3, 4]) == 2
    assert common_suffix([1, 2], [3, 4]) == 0
    assert layer_of("model.layers.12.mlp.up_proj") == 12 and layer_of("lm_head") is None


def test_encode_pair_aligns_the_shared_suffix(tokenizer):
    pair = encode_pair(tokenizer, "w6 w7", "w8 w9", CLEAN, CORRUPT, style="plain")
    assert pair.n_answer == 2
    assert pair.ids_clean.shape[1] != pair.ids_corrupt.shape[1]  # prompts differ in length
    assert pair.suffix >= pair.n_answer + 1
    assert torch.equal(pair.ids_clean[0, -pair.suffix:], pair.ids_corrupt[0, -pair.suffix:])


def test_patching_every_linear_input_reproduces_the_clean_run(tiny_model, tokenizer):
    names = _linear_names(tiny_model)
    conds = {"all": _all_channels(tiny_model, names), "none": {n: torch.tensor([], dtype=torch.long) for n in names}}
    rep = pair_patch_probe(tiny_model, tokenizer, ROWS, clean_sys=CLEAN, corrupt_sys=CORRUPT,
                           conditions=conds, style="plain", n_boot=20)
    assert rep["n_questions"] == len(ROWS) and rep["n_skipped"] == 0
    assert rep["mean_margin"]["clean"] != pytest.approx(rep["mean_margin"]["corrupt"], abs=1e-3)
    assert rep["conditions"]["all"]["restoration"] == pytest.approx(1.0, abs=1e-3)
    assert rep["conditions"]["none"]["restoration"] == pytest.approx(0.0, abs=1e-6)
    assert rep["conditions"]["all"]["channels"] > 0 and rep["conditions"]["none"]["channels"] == 0


def test_partial_patch_lands_between_corrupt_and_clean_runs_and_leaves_no_hooks(tiny_model, tokenizer):
    ids = tokenizer(["w1 w2 w3 w4"], return_tensors="pt")["input_ids"]
    before = tiny_model(input_ids=ids).logits.detach()
    one_layer = [n for n in _linear_names(tiny_model) if layer_of(n) == 0]
    rep = pair_patch_probe(tiny_model, tokenizer, ROWS, clean_sys=CLEAN, corrupt_sys=CORRUPT,
                           conditions={"layer0": _all_channels(tiny_model, one_layer)},
                           style="plain", n_boot=20)
    r = rep["conditions"]["layer0"]
    lo, hi = r["ci95"]
    assert lo <= r["restoration"] <= hi
    assert torch.equal(tiny_model(input_ids=ids).logits, before)  # hooks are gone


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


def test_zdiff_picks_channels_whose_mean_differs_in_standard_units():
    def stat(M, var):
        M = torch.tensor(M); V = torch.tensor(var) + M.square()
        return {"L": {"A": M.abs() + 1, "V": V, "M": M, "count": 10}}
    stats = {"T": stat([0.0, 3.0, 0.0, -2.0], [1.0] * 4), "X": stat([0.0, 0.0, 0.0, 0.0], [1.0] * 4)}
    got = select_channels(stats, "T", ["X"], 0.5, "zdiff")["L"].tolist()
    assert got == [1, 3]
