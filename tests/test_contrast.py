"""Domain-contrast scores and the in-place, per-row masks."""

import math

import pytest
import torch

from compress.contrast import apply_domain_mask, contrast_scores

LAYER = "model.layers.0.mlp.down_proj"


def _stats(in_features, z_signs, layer=LAYER):
    """Synthetic two-domain stats whose contrast z is proportional to z_signs."""
    signs = torch.tensor(z_signs, dtype=torch.float64)
    seek = {
        "M": signs,
        "V": signs.square() + 1.0,  # var = 1
        "A": torch.ones(in_features, dtype=torch.float64),
        "count": 1,
    }
    reject = {
        "M": torch.zeros(in_features, dtype=torch.float64),
        "V": torch.ones(in_features, dtype=torch.float64),
        "A": torch.ones(in_features, dtype=torch.float64),
        "count": 1,
    }
    return {"seek": {layer: seek}, "reject": {layer: reject}}


def test_contrast_score_uses_pooled_variance():
    stats = _stats(2, [1, -1])
    z = contrast_scores(stats, "seek", "reject")[LAYER]
    assert torch.allclose(z, torch.tensor([1.0, -1.0]) / math.sqrt(2.0), atol=1e-6)


def test_contrast_score_clip_bounds_outliers():
    stats = _stats(2, [1000, -1000])
    z = contrast_scores(stats, "seek", "reject", clip=5.0)[LAYER]
    assert z.abs().max() <= 5.0
    assert z.tolist() == [5.0, -5.0]


def test_contrast_mask_keeps_seek_channels_and_inverse_keeps_the_rest(tiny_model):
    layer = dict(tiny_model.named_modules())[LAYER]
    out_features, in_features = layer.weight.shape
    assert in_features == 32
    with torch.no_grad():
        layer.weight.fill_(1.0)
    stats = _stats(in_features, [1] * 16 + [-1] * 16)

    summary = apply_domain_mask(
        tiny_model, stats, kind="contrast", seek="seek", reject="reject", sparsity=0.5
    )
    assert summary["layers"] == 1
    assert summary["kept"] == out_features * 16
    assert summary["kept_fraction"] == pytest.approx(0.5)
    assert torch.count_nonzero(layer.weight[:, :16]) == out_features * 16
    assert torch.count_nonzero(layer.weight[:, 16:]) == 0

    # Reset and use the anti-domain mask: it must keep the other half.
    with torch.no_grad():
        layer.weight.fill_(1.0)
    apply_domain_mask(
        tiny_model, stats, kind="inverse", seek="seek", reject="reject", sparsity=0.5
    )
    assert torch.count_nonzero(layer.weight[:, :16]) == 0
    assert torch.count_nonzero(layer.weight[:, 16:]) == out_features * 16


@pytest.mark.parametrize("kind", ["magnitude", "wanda", "random"])
def test_control_masks_keep_exactly_the_requested_fraction(tiny_model, kind):
    layer = dict(tiny_model.named_modules())[LAYER]
    out_features, in_features = layer.weight.shape
    stats = _stats(in_features, [1] * in_features)
    summary = apply_domain_mask(
        tiny_model, stats, kind=kind, seek="seek", reject="reject", sparsity=0.25, seed=7
    )
    assert summary["layers"] == 1
    assert summary["kept"] == out_features * in_features * 0.75
    assert torch.count_nonzero(layer.weight) == summary["kept"]


def test_none_leaves_weights_untouched(tiny_model):
    before = tiny_model.model.layers[0].mlp.down_proj.weight.detach().clone()
    summary = apply_domain_mask(tiny_model, {}, kind="none")
    assert summary["layers"] == 0
    assert torch.equal(before, tiny_model.model.layers[0].mlp.down_proj.weight)


def test_old_stats_without_signed_means_are_rejected(tiny_model):
    stats = _stats(32, [1] * 32)
    del stats["seek"][LAYER]["M"]
    with pytest.raises(ValueError, match="signed means"):
        apply_domain_mask(
            tiny_model, stats, kind="contrast", seek="seek", reject="reject"
        )


def test_only_mlp_skips_attention_linears(tiny_model):
    attn = "model.layers.0.self_attn.q_proj"
    stats = _stats(16, [1] * 16, layer=attn)
    mlp = _stats(32, [1] * 32, layer=LAYER)
    stats["seek"].update(mlp["seek"])
    stats["reject"].update(mlp["reject"])
    before = dict(tiny_model.named_modules())[attn].weight.detach().clone()

    summary = apply_domain_mask(
        tiny_model, stats, kind="magnitude", seek="seek", reject="reject",
        sparsity=0.25, only_mlp=True,
    )
    assert summary["layers"] == 1
    assert torch.equal(before, dict(tiny_model.named_modules())[attn].weight)
    assert torch.count_nonzero(dict(tiny_model.named_modules())[LAYER].weight) == summary["kept"]


def test_missing_domain_lists_the_available_ones(tiny_model):
    stats = {"law": _stats(32, [1] * 32)["seek"]}
    with pytest.raises(ValueError, match=r"available: \['law'\]"):
        apply_domain_mask(
            tiny_model, stats, kind="contrast", seek="math", reject="law"
        )


def test_sparsity_outside_half_open_range_is_rejected(tiny_model):
    with pytest.raises(ValueError, match="sparsity"):
        apply_domain_mask(
            tiny_model, _stats(32, [1] * 32), kind="magnitude",
            seek="seek", reject="reject", sparsity=1.0,
        )
