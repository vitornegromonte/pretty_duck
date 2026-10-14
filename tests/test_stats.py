import pytest
import torch

from compress.stats import ActivationCollector, collect_domain, load_stats, save_stats
from tests.conftest import CALIB_LINES


def _manual_moments(model, tokenizer, lines, max_len=16):
    """Reference implementation: single-line batches, explicit masking."""
    sums, sqs, counts = {}, {}, {}
    hooks = []

    def make_hook(name, d):
        def hook(module, inputs, output):
            x = inputs[0].reshape(-1, d)
            sums[name] = sums.get(name, 0) + x.double().abs().sum(0)
            sqs[name] = sqs.get(name, 0) + x.double().square().sum(0)
            counts[name] = counts.get(name, 0) + x.shape[0]
        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name, module.in_features)))
    with torch.no_grad():
        for line in lines:
            enc = tokenizer(line, return_tensors="pt", truncation=True, max_length=max_len)
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    for h in hooks:
        h.remove()
    return {
        n: (sums[n] / counts[n], sqs[n] / counts[n], counts[n]) for n in sums
    }


def test_batching_and_masking_match_manual_reference(tiny_model, tokenizer):
    got = collect_domain(
        tiny_model, tokenizer, CALIB_LINES, batch_size=3, max_len=16, n_samples=None
    )
    want = _manual_moments(tiny_model, tokenizer, CALIB_LINES)
    assert set(got) == set(want)
    for name, (A, V, count) in want.items():
        assert got[name]["count"] == count
        assert torch.allclose(got[name]["A"], A, atol=1e-10)
        assert torch.allclose(got[name]["V"], V, atol=1e-10)


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


def test_unused_linear_raises(tiny_model, tokenizer):
    tiny_model.unused = torch.nn.Linear(16, 16)  # never runs in forward
    with pytest.raises(RuntimeError, match="never executed"):
        collect_domain(tiny_model, tokenizer, CALIB_LINES, n_samples=None)


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
