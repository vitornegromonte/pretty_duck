from pathlib import Path

import pytest
import torch

from compress.stats import (
    ActivationCollector,
    collect_domain,
    load_stats,
    save_stats,
    tied_weight_ptrs,
)
from tests.conftest import CALIB_LINES


def _manual_moments(model, tokenizer, lines, max_len=16):
    """Reference implementation: single-line batches, explicit masking."""
    sums, sqs, signed, counts = {}, {}, {}, {}
    hooks = []
    tied = tied_weight_ptrs(model)

    def make_hook(name, d):
        def hook(module, inputs, output):
            x = inputs[0].reshape(-1, d)
            sums[name] = sums.get(name, 0) + x.double().abs().sum(0)
            sqs[name] = sqs.get(name, 0) + x.double().square().sum(0)
            signed[name] = signed.get(name, 0) + x.double().sum(0)
            counts[name] = counts.get(name, 0) + x.shape[0]
        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.weight.data_ptr() not in tied:
            hooks.append(module.register_forward_hook(make_hook(name, module.in_features)))
    with torch.no_grad():
        for line in lines:
            enc = tokenizer(line, return_tensors="pt", truncation=True, max_length=max_len)
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    for h in hooks:
        h.remove()
    return {
        n: (sums[n] / counts[n], sqs[n] / counts[n], signed[n] / counts[n], counts[n])
        for n in sums
    }


def test_batching_and_masking_match_manual_reference(tiny_model, tokenizer):
    got = collect_domain(
        tiny_model, tokenizer, CALIB_LINES, batch_size=3, max_len=16, n_samples=None
    )
    want = _manual_moments(tiny_model, tokenizer, CALIB_LINES)
    assert set(got) == set(want)
    for name, (A, V, M, count) in want.items():
        assert got[name]["count"] == count
        assert torch.allclose(got[name]["A"], A, atol=1e-10)
        assert torch.allclose(got[name]["V"], V, atol=1e-10)
        assert torch.allclose(got[name]["M"], M, atol=1e-10)


def test_padding_positions_are_excluded(tiny_model, tokenizer):
    line = "w1 w2 w3"
    short = collect_domain(tiny_model, tokenizer, [line], max_len=16, n_samples=1)
    # Same line, right-padded to length 16; pad positions must not contribute.
    enc = tokenizer([line], return_tensors="pt", padding="max_length", max_length=16)
    with ActivationCollector(tiny_model) as c:
        c.observe(enc["attention_mask"])
        with torch.no_grad():
            tiny_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        got = c.layers()
    for name in got:
        assert got[name]["count"] == int(enc["attention_mask"].sum())  # pads not counted
        assert torch.allclose(got[name]["A"], short[name]["A"], atol=1e-10)
        assert torch.allclose(got[name]["V"], short[name]["V"], atol=1e-10)
        assert torch.allclose(got[name]["M"], short[name]["M"], atol=1e-10)


def test_tied_lm_head_is_not_collected(tiny_model, tokenizer):
    """The head shares storage with the embeddings, so it is not our target."""
    assert tiny_model.lm_head.weight.data_ptr() == (
        tiny_model.model.embed_tokens.weight.data_ptr()
    )
    got = collect_domain(tiny_model, tokenizer, CALIB_LINES, n_samples=None)
    assert "lm_head" not in got


def test_weights_are_not_modified(tiny_model, tokenizer):
    before = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    collect_domain(tiny_model, tokenizer, CALIB_LINES, n_samples=None)
    for n, p in tiny_model.named_parameters():
        assert torch.equal(before[n], p)


def test_sample_selection_is_seed_deterministic(tiny_model, tokenizer):
    a = collect_domain(tiny_model, tokenizer, CALIB_LINES, n_samples=4, seed=7)
    b = collect_domain(tiny_model, tokenizer, CALIB_LINES, n_samples=4, seed=7)
    for name in a:
        assert torch.equal(a[name]["A"], b[name]["A"])
        assert a[name]["count"] == b[name]["count"]


def test_empty_calibration_raises(tiny_model, tokenizer):
    with pytest.raises(ValueError, match="no non-empty"):
        collect_domain(tiny_model, tokenizer, ["", "   "])


def test_unused_linear_in_text_tower_raises(tiny_model, tokenizer):
    tiny_model.model.unused = torch.nn.Linear(16, 16)  # never runs in forward
    with pytest.raises(RuntimeError, match="never executed"):
        collect_domain(tiny_model, tokenizer, CALIB_LINES, n_samples=None)


def test_linear_outside_text_tower_is_ignored(tiny_model, tokenizer):
    # Stands in for a multimodal model's vision tower: attached to the wrapper,
    # outside the decoder, and never executed on text.
    tiny_model.vision = torch.nn.Linear(4, 4)
    got = collect_domain(tiny_model, tokenizer, CALIB_LINES, n_samples=2)
    assert "vision" not in got
    assert "model.layers.0.self_attn.q_proj" in got


def test_hooks_removed_on_exception(tiny_model, tokenizer):
    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom):
        with ActivationCollector(tiny_model):
            raise Boom()
    assert len(tiny_model._forward_hooks) == 0


def test_save_load_roundtrip(tmp_path, tiny_model, tokenizer):
    domains = {
        "a": collect_domain(tiny_model, tokenizer, CALIB_LINES, n_samples=None),
        "b": collect_domain(tiny_model, tokenizer, CALIB_LINES, n_samples=None),
    }
    path = tmp_path / "stats.pt"
    save_stats(domains, path, meta={"seed": 42})
    loaded, meta = load_stats(path)
    assert meta == {"seed": 42}
    assert set(loaded) == {"a", "b"}
    for name in domains["a"]:
        assert torch.equal(loaded["a"][name]["A"], domains["a"][name]["A"])
        assert loaded["a"][name]["count"] == domains["a"][name]["count"]


def test_failed_save_leaves_no_stats_file(tmp_path, monkeypatch):
    path = tmp_path / "stats.pt"

    def no_space(self, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_bytes", no_space)
    with pytest.raises(OSError, match="No space left"):
        save_stats({}, path, meta={})
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []  # temp file cleaned up too
