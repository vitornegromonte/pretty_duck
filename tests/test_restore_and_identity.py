"""Properties that protect the scientific claims: weights are restored, and 'protect nothing' equals uniform."""

import pytest
import torch

from compress.quant.fake import QuantizeColumns, fake_quant_group, fake_quant_group_keep_blocks
from compress.quant.gptq import GPTQQuant
from compress.quant.protect import FormatPlan, MixedQuant
from tests.conftest import CALIB_LINES


def _names(model):
    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and ".layers." in n]


def _batches(tokenizer):
    return [tokenizer(CALIB_LINES[i:i + 4], return_tensors="pt", padding=True) for i in (0, 4)]


def _managers(model, tokenizer):
    names = _names(model)
    keep = {names[0]: torch.tensor([1, 5, 9])}
    cols = {names[0]: torch.tensor([0, 3, 4])}
    return {
        "mixed": lambda: MixedQuant(model, names, 3, keep, 8),
        "mixed_keep_blocks": lambda: MixedQuant(model, names, 3, keep, 8, keep_blocks=True),
        "mixed_offload": lambda: MixedQuant(model, names, 3, keep, 8, offload=True),
        "format_plan": lambda: FormatPlan(model, names, default="int4", protect=keep),
        "gptq": lambda: GPTQQuant(model, _batches(tokenizer), names, 3, keep, 8),
        "gptq_lowrank": lambda: GPTQQuant(model, _batches(tokenizer), names, 3, None, 8, rank=2),
        "columns": lambda: QuantizeColumns(model, cols, 2),
    }


@pytest.mark.parametrize("kind", ["mixed", "mixed_keep_blocks", "mixed_offload", "format_plan", "gptq",
                                  "gptq_lowrank", "columns"])
def test_every_weight_changing_manager_restores_the_weights_exactly(tiny_model, tokenizer, kind):
    before = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    make = _managers(tiny_model, tokenizer)[kind]
    with make():
        changed = any(not torch.equal(p, before[n]) for n, p in tiny_model.named_parameters())
    assert changed, "the manager did not change any weight, so the test proves nothing"
    assert all(torch.equal(p, before[n]) for n, p in tiny_model.named_parameters())
    with pytest.raises(RuntimeError, match="boom"):
        with make():
            raise RuntimeError("boom")
    assert all(torch.equal(p, before[n]) for n, p in tiny_model.named_parameters())


def test_protecting_no_columns_equals_uniform_quantization(tiny_model, tokenizer):
    names = _names(tiny_model)
    modules = dict(tiny_model.named_modules())

    def snapshot(manager):
        with manager:
            return {n: modules[n].weight.detach().clone() for n in names}

    uniform = snapshot(MixedQuant(tiny_model, names, 3, None, 8))
    empty = {n: torch.tensor([], dtype=torch.long) for n in names}
    for manager in (MixedQuant(tiny_model, names, 3, empty, 8),
                    MixedQuant(tiny_model, names, 3, empty, 8, keep_blocks=True)):
        got = snapshot(manager)
        assert all(torch.equal(got[n], uniform[n]) for n in names)
    gptq_plain = snapshot(GPTQQuant(tiny_model, _batches(tokenizer), names, 3, None, 8))
    gptq_empty = snapshot(GPTQQuant(tiny_model, _batches(tokenizer), names, 3, empty, 8))
    assert all(torch.equal(gptq_empty[n], gptq_plain[n]) for n in names)


def test_keep_blocks_leaves_blocks_without_protected_columns_identical_to_uniform():
    torch.manual_seed(0)
    W = torch.randn(6, 64)
    protect = torch.tensor([3, 40])  # scattered: one column in block 0 and one in block 5 (group 8)
    kept = fake_quant_group_keep_blocks(W, 3, 8, protect)
    uniform = fake_quant_group(W, 3, 8)
    assert torch.equal(kept[:, protect], W[:, protect])
    untouched = [b for b in range(8) if not any(b * 8 <= int(c) < (b + 1) * 8 for c in protect)]
    for b in untouched:
        assert torch.equal(kept[:, b * 8:(b + 1) * 8], uniform[:, b * 8:(b + 1) * 8])
    # re-forming blocks over the rest (the old behavior) moves every boundary after the first protected column
    rest = torch.tensor([j for j in range(64) if j not in protect.tolist()])
    regrouped = fake_quant_group(W[:, rest], 3, 8)
    assert not torch.equal(regrouped[:, 10:], uniform[:, rest][:, 10:])


def test_keep_blocks_equals_regrouping_for_an_aligned_contiguous_unit():
    torch.manual_seed(1)
    W = torch.randn(4, 64)
    protect = torch.arange(16, 32)  # two whole blocks of 8
    rest = torch.tensor([j for j in range(64) if j not in protect.tolist()])
    kept = fake_quant_group_keep_blocks(W, 3, 8, protect)
    assert torch.equal(kept[:, rest], fake_quant_group(W[:, rest], 3, 8))
