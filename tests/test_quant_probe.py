import pytest
import torch

from compress.quant_probe import QuantizeColumns, fake_quant, quant_persona_probe

Q0 = "model.layers.0.self_attn.q_proj"
ROWS = [
    {"question": "w6 w7", "A": "w8 w9", "B": "w10 w11 w12", "matching": "A"},
    {"question": "w13 w14 w15", "A": "w16", "B": "w17 w18", "matching": "B"},
]
PROMPTS = {"neutral": "", "seek": "w1 w2", "reject": "w3 w4 w5"}


def _logits(model, tokenizer):
    with torch.no_grad():
        return model(**tokenizer(["w1 w2 w3 w4"], return_tensors="pt")).logits


def test_fake_quant_error_shrinks_with_more_bits():
    torch.manual_seed(0)
    W = torch.randn(8, 32)
    errs = [float((W - fake_quant(W, b)).square().mean()) for b in (2, 3, 4, 8)]
    assert errs == sorted(errs, reverse=True) and errs[-1] < errs[0] / 100
    with pytest.raises(ValueError, match="bits"):
        fake_quant(W, 1)


def test_quantize_columns_touches_only_the_set_and_restores_exactly(tiny_model, tokenizer):
    module = dict(tiny_model.named_modules())[Q0]
    before_w, before_out = module.weight.detach().clone(), _logits(tiny_model, tokenizer)
    idx = torch.tensor([1, 5, 9])
    V = {Q0: torch.ones(16)}
    with QuantizeColumns(tiny_model, {Q0: idx}, 2, V) as q:
        changed = (module.weight.detach() != before_w).any(dim=0).nonzero().flatten().tolist()
        assert set(changed) <= {1, 5, 9} and changed
        assert q.dose > 0
        assert not torch.equal(_logits(tiny_model, tokenizer), before_out)
    assert torch.equal(module.weight.detach(), before_w)
    assert torch.equal(_logits(tiny_model, tokenizer), before_out)


def test_weights_restored_when_the_body_raises(tiny_model):
    module = dict(tiny_model.named_modules())[Q0]
    before = module.weight.detach().clone()
    with pytest.raises(RuntimeError, match="boom"):
        with QuantizeColumns(tiny_model, {Q0: torch.arange(8)}, 2):
            raise RuntimeError("boom")
    assert torch.equal(module.weight.detach(), before)
    with pytest.raises(KeyError, match="not an nn.Linear"):
        QuantizeColumns(tiny_model, {"model.embed_tokens": torch.arange(2)}, 4).__enter__()


def test_dose_grows_with_coarser_quantization(tiny_model):
    sets, V = {Q0: torch.arange(16)}, {Q0: torch.ones(16)}
    doses = []
    for bits in (8, 4, 2):
        with QuantizeColumns(tiny_model, sets, bits, V) as q:
            doses.append(q.dose)
    assert doses == sorted(doses) and doses[-1] > 100 * doses[0]


def test_probe_retention_is_one_without_damage_and_reports_damage(tiny_model, tokenizer):
    all_layers = {n: torch.arange(m.in_features) for n, m in tiny_model.named_modules()
                  if isinstance(m, torch.nn.Linear) and n != "lm_head"}
    none = {Q0: torch.tensor([], dtype=torch.long)}
    rep = quant_persona_probe(
        tiny_model, tokenizer, ROWS, prompts=PROMPTS, n_boot=20, keep_per_question=True,
        conditions={"none": (none, 4), "all2": (all_layers, 2)})
    assert rep["conditions"]["none"]["retention"] == pytest.approx(1.0)
    assert rep["conditions"]["none"]["generic_damage"] == pytest.approx(0.0, abs=1e-9)
    assert rep["conditions"]["all2"]["generic_damage"] > 0
    lo, hi = rep["conditions"]["all2"]["ci95"]
    assert lo <= rep["conditions"]["all2"]["retention"] <= hi
    assert len(rep["per_question"]["conditions"]["all2"]["seek"]) == len(ROWS)
