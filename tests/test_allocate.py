import json

import pytest

from compress.allocate import build_plan, estimate_model_blocks, load_plan, save_plan
from compress.stats import collect_domain
from tests.conftest import CALIB_LINES


def test_estimate_model_blocks_skips_tied_lm_head(tiny_model, tokenizer):
    stats = {"d": collect_domain(tiny_model, tokenizer, CALIB_LINES, n_samples=2)}
    est = estimate_model_blocks(tiny_model, stats, ["fp16", "int8"])
    assert "lm_head" not in est
    assert set(est) == set(stats["d"])


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
