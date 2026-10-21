import pytest
import torch

from compress import serialize
from compress.allocate import load_plan, save_plan
from compress.quant import quantize_weight
from compress.stats import tied_weight_ptrs


def _full_plan(model, fmt="int4"):
    """Plan that quantizes every Linear of ``model`` to one format."""
    blocks = {
        name: {"format": fmt}
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    return {"version": 1, "model": "tiny", "target_domain": "T", "blocks": blocks,
            "decisions": [], "totals": {}}


def test_saved_block_reconstructs_exactly(tiny_model, tmp_path):
    plan = _full_plan(tiny_model, "int4")
    manifest = serialize.save_compressed(tiny_model, plan, tmp_path / "c")
    name = next(iter(manifest["blocks"]))
    stored = serialize.load_block(manifest, tmp_path / "c", name)
    module = dict(tiny_model.named_modules())[name]
    q = quantize_weight(module.weight.detach(), "int4")
    assert torch.equal(stored, q.dequant)


def test_manifest_records_actual_file_sizes(tiny_model, tmp_path):
    out = tmp_path / "c"
    manifest = serialize.save_compressed(tiny_model, _full_plan(tiny_model, "int4"), out)
    for name, info in manifest["blocks"].items():
        actual = (out / info["file"]).stat().st_size
        assert info["file_nbytes"] == actual
        assert info["payload_nbytes"] <= info["file_nbytes"]  # pickle overhead >= 0
        assert info["payload_nbytes"] == info["analytic_nbytes"]  # formula stays honest
    assert manifest["totals"]["file_nbytes"] == sum(
        i["file_nbytes"] for i in manifest["blocks"].values()
    )
    assert manifest["totals"]["fp16_reference_nbytes"] == sum(
        m.weight.numel() * 2 for m in tiny_model.modules() if isinstance(m, torch.nn.Linear)
    )
    # dir size includes the manifest itself
    assert serialize.measure_bytes(out) >= manifest["totals"]["file_nbytes"]


def test_apply_to_model_requires_a_manifest(tiny_model, tmp_path):
    with pytest.raises(FileNotFoundError, match="quantize_model"):
        serialize.apply_to_model(tiny_model, tmp_path / "missing")


def test_apply_to_model_mutates_weights_and_counts(tiny_model, tmp_path):
    out = tmp_path / "c"
    plan = _full_plan(tiny_model, "nf4")  # includes the tied lm_head
    serialize.save_compressed(tiny_model, plan, out)
    tied = tied_weight_ptrs(tiny_model)
    linears = {
        n: m for n, m in tiny_model.named_modules()
        if isinstance(m, torch.nn.Linear) and m.weight.data_ptr() not in tied
    }
    before = {n: m.weight.detach().clone() for n, m in linears.items()}
    embeddings = tiny_model.model.embed_tokens.weight.detach().clone()
    n = serialize.apply_to_model(tiny_model, out)
    assert n == len(linears)
    for name, module in linears.items():
        assert not torch.equal(module.weight, before[name])  # weights replaced
        expected = quantize_weight(before[name], "nf4").dequant
        assert torch.allclose(module.weight.cpu(), expected, atol=1e-6)
    # the tied lm_head in the plan must not leak into the embedding matrix
    assert torch.equal(tiny_model.model.embed_tokens.weight, embeddings)


def test_save_removes_stale_block_files(tiny_model, tmp_path):
    out = tmp_path / "c"
    serialize.save_compressed(tiny_model, _full_plan(tiny_model, "int8"), out)
    stale = out / "tensors" / "a_block_no_longer_planned.pt"
    stale.write_bytes(b"left over from an earlier plan")
    serialize.save_compressed(tiny_model, _full_plan(tiny_model, "int4"), out)
    assert not stale.exists()  # measure_bytes must not count it either


def test_plan_roundtrip_on_disk(tmp_path):
    plan = _full_plan(torch.nn.Linear(4, 4), "int8")
    path = tmp_path / "plan.json"
    save_plan(plan, path)
    assert load_plan(path)["blocks"] == plan["blocks"]


def test_shape_mismatch_detected(tiny_model, tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    out = tmp_path / "c"
    serialize.save_compressed(tiny_model, _full_plan(tiny_model, "int8"), out)
    # Same layer names, different hidden size -> stored blocks cannot fit.
    other = LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=8, intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
    )).eval()
    with pytest.raises(ValueError, match="shape mismatch"):
        serialize.apply_to_model(other, out)
