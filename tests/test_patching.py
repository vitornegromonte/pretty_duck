import random

import pytest
import torch

from compress.patching import (
    ChannelPatch,
    bootstrap_ci,
    causal_probe,
    energy_matched_random,
    line_nll,
    matched_random,
    perturbation_energy,
    reference_means,
    select_channels,
    target_contrast_scores,
)
from compress.stats import collect_domain
from tests.conftest import CALIB_LINES

Q0 = "model.layers.0.self_attn.q_proj"


def _logits(model, tokenizer, line="w1 w2 w3 w4"):
    batch = tokenizer([line], return_tensors="pt")
    with torch.no_grad():
        return model(**batch).logits


def test_empty_set_leaves_output_unchanged(tiny_model, tokenizer):
    base = _logits(tiny_model, tokenizer)
    sets = {Q0: torch.tensor([], dtype=torch.long)}
    with ChannelPatch(tiny_model, sets, {Q0: torch.zeros(16)}):
        assert torch.equal(_logits(tiny_model, tokenizer), base)


def test_patching_changes_output_and_hooks_are_removed(tiny_model, tokenizer):
    base = _logits(tiny_model, tokenizer)
    weights = {n: p.clone() for n, p in tiny_model.named_parameters()}
    with ChannelPatch(tiny_model, {Q0: torch.arange(8)}, {Q0: torch.full((16,), 5.0)}):
        assert not torch.allclose(_logits(tiny_model, tokenizer), base)
    assert torch.equal(_logits(tiny_model, tokenizer), base)  # hook gone
    assert all(torch.equal(p, weights[n]) for n, p in tiny_model.named_parameters())


def test_hooks_removed_when_the_body_raises(tiny_model, tokenizer):
    base = _logits(tiny_model, tokenizer)
    with pytest.raises(RuntimeError, match="boom"):
        with ChannelPatch(tiny_model, {Q0: torch.arange(8)}, {Q0: torch.ones(16)}):
            raise RuntimeError("boom")
    assert torch.equal(_logits(tiny_model, tokenizer), base)


def test_bad_patch_target_is_reported_and_leaves_no_hooks(tiny_model, tokenizer):
    base = _logits(tiny_model, tokenizer)
    sets = {Q0: torch.arange(4), "model.embed_tokens": torch.arange(4)}
    values = {Q0: torch.zeros(16), "model.embed_tokens": torch.zeros(16)}
    with pytest.raises(KeyError, match="not an nn.Linear"):
        ChannelPatch(tiny_model, sets, values).__enter__()
    assert torch.equal(_logits(tiny_model, tokenizer), base)


def test_patching_with_the_exact_activation_is_the_identity(tiny_model, tokenizer):
    # Layer 0's q_proj input is norm(embed("w1")) at every position of a constant
    # line, so its signed mean M *is* the activation: patching every channel
    # with it must reproduce the unpatched logits.
    line = "w1 w1 w1 w1"
    stats = {"d": collect_domain(tiny_model, tokenizer, [line], n_samples=1)}
    base = _logits(tiny_model, tokenizer, line)
    values = reference_means(stats, ["d"], "mean")
    with ChannelPatch(tiny_model, {Q0: torch.arange(16)}, values):
        assert torch.allclose(_logits(tiny_model, tokenizer, line), base, atol=1e-5)


def test_line_nll_ignores_padding(tiny_model, tokenizer):
    short, long = "w1 w2 w3", "w5 w4 w3 w2 w1 w1 w2 w3"
    alone = line_nll(tiny_model, tokenizer, [short], batch_size=1)[0]
    batched = line_nll(tiny_model, tokenizer, [short, long], batch_size=2)[0]
    assert batched == pytest.approx(alone, abs=1e-5)
    tokenizer.padding_side = "left"
    try:
        assert line_nll(tiny_model, tokenizer, [short, long], batch_size=2)[0] == pytest.approx(
            alone, abs=1e-5)
        assert tokenizer.padding_side == "left"  # caller's setting is restored
    finally:
        tokenizer.padding_side = "right"


def _fake_stats():
    """One layer, 8 channels: channel 0 leans to the target, 7 to the others."""
    A_T = torch.tensor([9.0, 5, 5, 5, 5, 5, 5, 1])
    A_o = torch.tensor([1.0, 5, 5, 5, 5, 5, 5, 9])
    stat = lambda A: {"L": {"A": A, "V": A.square(), "M": A.clone(), "count": 10}}
    return {"T": stat(A_T), "X": stat(A_o), "Y": stat(A_o + 0.01)}


def test_target_contrast_ranks_target_leaning_channels_first():
    stats = _fake_stats()
    s = target_contrast_scores({d: stats[d]["L"]["A"] for d in stats}, "T", ["X", "Y"])
    assert int(s.argmax()) == 0 and int(s.argmin()) == 7


def test_select_channels_kinds_and_sizes():
    stats = _fake_stats()
    tarc = select_channels(stats, "T", ["X", "Y"], 0.25, "tarc")
    anti = select_channels(stats, "T", ["X", "Y"], 0.25, "anti")
    imp = select_channels(stats, "T", ["X", "Y"], 0.25, "importance")
    assert 0 in tarc["L"].tolist() and 7 in anti["L"].tolist() and 0 in imp["L"].tolist()
    assert all(s["L"].numel() == 2 for s in (tarc, anti, imp))
    with pytest.raises(ValueError, match="k_frac"):
        select_channels(stats, "T", ["X"], 0.9, "tarc")
    with pytest.raises(ValueError, match="ref_sets"):
        select_channels(stats, "T", ["X"], 0.25, "random")


def test_matched_random_is_disjoint_same_size_and_v_matched():
    g = torch.Generator().manual_seed(0)
    V = torch.rand(400, generator=g)
    ref = torch.randperm(400, generator=g)[:40].sort().values
    drawn = matched_random(V, ref, random.Random(1))
    assert drawn.numel() == 40 and not set(drawn.tolist()) & set(ref.tolist())
    assert len(set(drawn.tolist())) == 40
    # V profile is much closer than an unmatched draw would be
    gap = (V[ref].sort().values - V[drawn].sort().values).abs().mean()
    unmatched = V[torch.randperm(360)[:40]].sort().values
    assert gap < 0.02 and gap < (V[ref].sort().values - unmatched).abs().mean()
    with pytest.raises(ValueError, match="disjoint"):
        matched_random(V[:10], torch.arange(6), random.Random(0))


def test_bootstrap_ci_brackets_the_point_estimate():
    point, lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 2.0], {"o": [0.0, 0.1, -0.1, 0.0]}, n_boot=200)
    assert point == pytest.approx(2.0)
    assert lo <= point <= hi
    flat = bootstrap_ci([1.0] * 5, {"o": [0.5] * 5}, n_boot=50)
    assert flat == pytest.approx((0.5, 0.5, 0.5))


def test_causal_probe_runs_end_to_end_and_reports_all_conditions(tiny_model, tokenizer):
    stats = {
        "T": collect_domain(tiny_model, tokenizer, CALIB_LINES[:4], n_samples=4),
        "X": collect_domain(tiny_model, tokenizer, CALIB_LINES[4:], n_samples=4),
        "Y": collect_domain(tiny_model, tokenizer, CALIB_LINES[::2], n_samples=4),
    }
    weights = {n: p.clone() for n, p in tiny_model.named_parameters()}
    report = causal_probe(
        tiny_model, tokenizer, stats, target="T", others=["X", "Y"],
        eval_target=("t", CALIB_LINES[:4]), eval_others={"x": CALIB_LINES[4:], "y": CALIB_LINES[::2]},
        k_fracs=[0.25], n_random=2, n_boot=20,
    )
    kinds = [(c["kind"], c["replicate"]) for c in report["conditions"]]
    assert kinds == [("tarc", None), ("importance", None), ("anti", None),
                     ("random", 0), ("random", 1), ("energy_random", 0), ("energy_random", 1)]
    for c in report["conditions"]:
        assert set(c["delta_nll"]) == {"t", "x", "y"}
        lo, hi = c["selectivity_ci95"]
        assert lo <= c["selectivity"] <= hi
    tarc, rand, erand = (report["conditions"][i] for i in (0, 3, 5))
    assert tarc["set"]["channels"] == rand["set"]["channels"] == erand["set"]["channels"]
    assert 0 < tarc["set"]["E_target_share"] < 1 and 0 < tarc["set"]["E_others_share"] < 1
    assert any(abs(v) > 0 for c in report["conditions"] for v in c["delta_nll"].values())
    assert all(torch.equal(p, weights[n]) for n, p in tiny_model.named_parameters())


def test_perturbation_energy_is_the_mean_squared_deviation_from_the_replacement():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1000, 6, generator=g).double() * 2 + 1
    m = torch.tensor([0.0, 0.5, 1.0, -1.0, 2.0, 3.0]).double()
    layer = {"V": x.square().mean(0), "M": x.mean(0)}
    assert torch.allclose(perturbation_energy(layer, m), ((x - m) ** 2).mean(0))


def test_energy_matched_random_is_disjoint_and_matches_both_energies():
    g = torch.Generator().manual_seed(0)
    feat = torch.randn(600, 2, generator=g)
    ref = (feat[:, 0] - feat[:, 1]).topk(30).indices.sort().values  # asymmetric pairs
    drawn = energy_matched_random(feat, ref, random.Random(3))
    assert drawn.numel() == 30 and len(set(drawn.tolist())) == 30
    assert not set(drawn.tolist()) & set(ref.tolist())
    # the asymmetry (target minus others) survives matching; a plain draw would lose it
    gap = lambda i: float((feat[i, 0] - feat[i, 1]).mean())
    plain = torch.randperm(570)[:30]
    assert abs(gap(drawn) - gap(ref)) < abs(gap(plain) - gap(ref))
    with pytest.raises(ValueError, match="disjoint"):
        energy_matched_random(feat[:10], torch.arange(6), random.Random(0))
