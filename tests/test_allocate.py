import json

import pytest
import torch

from compress.allocate import (
    build_plan, estimate_model_blocks, load_plan, ranknorm, save_plan,
    target_anchored_contrast,
)
from compress.stats import collect_domain
from tests.conftest import CALIB_LINES


def test_estimate_model_blocks_skips_tied_lm_head(tiny_model, tokenizer):
    stats = {"d": collect_domain(tiny_model, tokenizer, CALIB_LINES, n_samples=2)}
    est = estimate_model_blocks(tiny_model, stats, ["fp16", "int8"])
    assert "lm_head" not in est
    assert set(est) == set(stats["d"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_estimate_model_blocks_handles_gpu_weights_with_cpu_stats(tiny_model, tokenizer):
    stats = {"d": collect_domain(tiny_model, tokenizer, CALIB_LINES, n_samples=2)}
    assert stats["d"]["model.layers.0.self_attn.q_proj"]["V"].device.type == "cpu"
    tiny_model.to("cuda")
    est = estimate_model_blocks(tiny_model, stats, ["fp16", "int8"])
    assert est and all(e["error"]["d"]["int8"] >= 0 for e in est.values())


def _est(shape, errors_by_domain_fmt, importance_by_domain):
    """Fabricate one estimate entry: errors[domain][fmt], importance[domain]."""
    out = {"shape": shape, "importance": importance_by_domain, "error": errors_by_domain_fmt}
    out["nbytes"] = {
        "fp16": shape[0] * shape[1] * 2,
        "int8": shape[0] * shape[1] + shape[0] * 2,
        "int4": (shape[0] * shape[1] + 1) // 2 + shape[0] * 2,
        "nf4": shape[0] * ((shape[1] + 63) // 64) * 64 // 2 + shape[0] * ((shape[1] + 63) // 64) * 2,
    }
    return out


def _two_block_estimates():
    """Block "sensitive" costs a lot of error when coarsened; "robust" costs ~0."""
    return {
        "sensitive": _est(
            (10, 100),
            {"T": {"fp16": 0.0, "int8": 10.0, "int4": 100.0}, "S": {"fp16": 0.0, "int8": 1.0, "int4": 10.0}},
            {"T": 100.0, "S": 1.0},
        ),
        "robust": _est(
            (10, 100),
            {"T": {"fp16": 0.0, "int8": 0.01, "int4": 0.02}, "S": {"fp16": 0.0, "int8": 0.01, "int4": 0.02}},
            {"T": 0.1, "S": 50.0},
        ),
    }


def test_error_budget_is_respected():
    est = _two_block_estimates()
    plan = build_plan(est, target="T", formats=["fp16", "int8", "int4"], error_budget=0.05)
    total = sum(est[name]["error"]["T"][spec["format"]] for name, spec in plan["blocks"].items())
    assert total <= 0.05
    assert plan["totals"]["estimated_error_target"] == pytest.approx(total)


def test_greedy_prefers_best_error_per_byte():
    plan = build_plan(_two_block_estimates(), target="T", formats=["fp16", "int8", "int4"],
                      error_budget=0.02)
    # "robust" is nearly free -> quantized first; "sensitive" stays at fp16 under
    # this tight budget.
    assert plan["blocks"]["robust"]["format"] in ("int8", "int4")
    assert plan["blocks"]["sensitive"]["format"] == "fp16"


def test_unlimited_budget_compresses_everything():
    plan = build_plan(_two_block_estimates(), target="T", formats=["fp16", "int8", "int4"],
                      error_budget=1e12)
    assert all(spec["format"] in ("int4",) for spec in plan["blocks"].values())
    assert plan["totals"]["bytes_final"] < plan["totals"]["bytes_fp16"]


def test_size_budget_is_enforced():
    est = _two_block_estimates()
    fp16_bytes = sum(e["nbytes"]["fp16"] for e in est.values())
    target_bytes = fp16_bytes // 2
    plan = build_plan(est, target="T", formats=["fp16", "int8", "int4"], size_budget=target_bytes)
    assert plan["totals"]["bytes_final"] <= target_bytes


def test_size_budget_unreachable_raises():
    with pytest.raises(ValueError, match="unreachable"):
        build_plan(_two_block_estimates(), target="T", formats=["fp16"],
                   size_budget=1)  # fp16 only: nothing can shrink


def test_contrastive_signal_changes_choice_under_tie():
    # Two blocks with identical quantization costs; only their importance
    # profiles differ.  The error signal breaks the tie lexicographically;
    # the contrastive signal must pick the block that is NOT target-unique.
    errors = {"T": {"fp16": 0.0, "int8": 1.0}, "S": {"fp16": 0.0, "int8": 1.0}}
    est = {
        "a_block": _est((10, 100), errors, {"T": 100.0, "S": 0.0}),  # target-unique
        "b_block": _est((10, 100), errors, {"T": 1.0, "S": 100.0}),  # secondary-unique
    }
    common = dict(target="T", secondary=["S"], formats=["fp16", "int8"],
                  error_budget=1.0, lam=1.0)
    plain = build_plan(est, signal="error", **common)
    assert plain["blocks"]["a_block"]["format"] == "int8"  # lexicographic tie-break
    assert plain["blocks"]["b_block"]["format"] == "fp16"

    contrast = build_plan(est, signal="contrastive", **common)
    assert contrast["blocks"]["b_block"]["format"] == "int8"
    assert contrast["blocks"]["a_block"]["format"] == "fp16"  # protected


def test_ranknorm_scales_to_unit_interval_with_average_ties():
    assert ranknorm([5.0]) == [0.5]
    assert ranknorm([3.0, 1.0, 2.0]) == [1.0, 0.0, 0.5]
    assert ranknorm([1.0, 1.0, 2.0]) == [0.25, 0.25, 1.0]


def test_tarc_cancels_structure_shared_by_all_domains():
    # Block size drives every domain's importance the same way; only "c" leans
    # on the target more than the other domains do.
    errors = {d: {"fp16": 0.0, "int8": 1.0} for d in "TXYZ"}
    imp = {
        "a": {"T": 1.0, "X": 1.0, "Y": 1.0, "Z": 1.0},
        "b": {"T": 2.0, "X": 2.0, "Y": 2.0, "Z": 2.0},
        "c": {"T": 3.0, "X": 0.5, "Y": 0.5, "Z": 0.5},
    }
    est = {n: _est((10, 100), errors, imp[n]) for n in imp}
    s_t = target_anchored_contrast(est, "T")
    assert s_t == {"a": -0.5, "b": -0.5, "c": 1.0}  # a, b follow the shared trend
    assert all(-1.0 <= v <= 1.0 for v in s_t.values())


def test_tarc_signal_protects_target_leaning_block():
    errors = {d: {"fp16": 0.0, "int8": 1.0} for d in "TXY"}
    est = {
        "a_block": _est((10, 100), errors, {"T": 100.0, "X": 1.0, "Y": 2.0}),
        "b_block": _est((10, 100), errors, {"T": 1.0, "X": 100.0, "Y": 50.0}),
    }
    common = dict(target="T", formats=["fp16", "int8"], error_budget=1.0, lam=1.0)
    assert build_plan(est, signal="error", **common)["blocks"]["a_block"]["format"] == "int8"
    plan = build_plan(est, signal="tarc", **common)  # others default to X, Y
    assert plan["secondary_domains"] == ["X", "Y"]
    assert plan["blocks"]["a_block"]["format"] == "fp16"  # protected
    assert plan["blocks"]["b_block"]["format"] == "int8"


def test_budget_mode_exactly_one():
    with pytest.raises(ValueError, match="exactly one"):
        build_plan(_two_block_estimates(), target="T", formats=["fp16", "int8"])
    with pytest.raises(ValueError, match="exactly one"):
        build_plan(_two_block_estimates(), target="T", formats=["fp16", "int8"],
                   error_budget=1.0, size_budget=100)


def test_missing_domain_raises():
    with pytest.raises(KeyError):
        build_plan(_two_block_estimates(), target="nope", formats=["fp16", "int8"], error_budget=1.0)


def test_non_portable_format_rejected():
    with pytest.raises(ValueError, match="portable"):
        build_plan(_two_block_estimates(), target="T", formats=["fp16", "bnb_nf4"], error_budget=1.0)


def test_plan_is_deterministic_and_serializable(tmp_path):
    a = build_plan(_two_block_estimates(), target="T", formats=["fp16", "int8", "int4"],
                   error_budget=0.5)
    b = build_plan(_two_block_estimates(), target="T", formats=["fp16", "int8", "int4"],
                   error_budget=0.5)
    assert a == b
    path = tmp_path / "plan.json"
    save_plan(a, path)
    assert load_plan(path) == a
    json.loads(path.read_text())  # valid JSON on disk


def test_error_budget_includes_the_fp16_baseline():
    est = _two_block_estimates()
    for e in est.values():
        e["error"]["T"]["fp16"] = 0.001  # fp16 is never free
    baseline = sum(e["error"]["T"]["fp16"] for e in est.values())
    # A budget below what fp16 already costs cannot be met.
    with pytest.raises(ValueError, match="unreachable"):
        build_plan(est, target="T", formats=["fp16", "int8", "int4"], error_budget=0.0005)
    plan = build_plan(est, target="T", formats=["fp16", "int8", "int4"],
                      error_budget=baseline + 0.001)
    assert plan["totals"]["estimated_error_target"] == pytest.approx(baseline)


def test_decisions_record_each_step():
    plan = build_plan(_two_block_estimates(), target="T", formats=["fp16", "int8", "int4"],
                      error_budget=1e12)
    # every block ends where its last decision left it
    last = {}
    for d in plan["decisions"]:
        last[d["block"]] = d["to"]
    for name, spec in plan["blocks"].items():
        assert spec["format"] == last[name]
    for d in plan["decisions"]:
        assert d["delta_bytes"] > 0 and d["to"] != d["from"]


def test_row_group_units_partition_the_layer_estimates(tiny_model, tokenizer):
    stats = {"d": collect_domain(tiny_model, tokenizer, CALIB_LINES, n_samples=4),
             "e": collect_domain(tiny_model, tokenizer, CALIB_LINES[::-1], n_samples=4)}
    formats = ["fp16", "int8", "int4", "nf4"]
    whole = estimate_model_blocks(tiny_model, stats, formats)
    grouped = estimate_model_blocks(tiny_model, stats, formats, row_group=6)
    assert all("@" in name for name in grouped)
    for layer, w in whole.items():
        units = [g for name, g in grouped.items() if name.startswith(layer + "@")]
        assert sum(u["shape"][0] for u in units) == w["shape"][0]
        for d in stats:
            assert sum(u["importance"][d] for u in units) == pytest.approx(w["importance"][d])
            for fmt in formats:
                assert sum(u["error"][d][fmt] for u in units) == pytest.approx(w["error"][d][fmt])
        for fmt in ("fp16", "int8"):  # int4/nf4 differ only by per-group byte padding
            assert sum(u["nbytes"][fmt] for u in units) == w["nbytes"][fmt]


def test_row_group_plan_mixes_formats_within_one_layer():
    errors = {"T": {"fp16": 0.0, "int8": 1.0}}
    est = {
        "L@0:4": _est((4, 100), errors, {"T": 1.0}),
        "L@4:8": _est((4, 100), {"T": {"fp16": 0.0, "int8": 50.0}}, {"T": 1.0}),
    }
    plan = build_plan(est, target="T", formats=["fp16", "int8"], error_budget=2.0)
    assert plan["blocks"] == {"L@0:4": {"format": "int8"}, "L@4:8": {"format": "fp16"}}
