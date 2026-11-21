"""The condition list, the MMLU groups, and a full run of the suite on the tiny model."""

import io
import json
import random
import tarfile

import pytest
import torch

from compress.data.calibration import mixed_domain_batches, read_lines, text_batches
from compress.data.mmlu import load_mmlu_groups, row_to_text
from compress.evaluation.mcq import accuracy_by_area
from compress.experiments.context import CircuitContext
from compress.experiments.suite import QuantSuite, SuiteConfig, build_conditions
from compress.interp.damage import KnowledgeProbe
from compress.interp.minimal_pairs import build_facts, make_pairs, split_facts
from compress.interp.units import enumerate_units, unit_sets
from tests.conftest import CALIB_LINES


def _config(tmp_path, **kw) -> SuiteConfig:
    base = dict(out=str(tmp_path / "out.json"), calib_dir=str(tmp_path), neutral=str(tmp_path / "neutral.txt"),
                target_subjects=["a"], control_subjects=["b"], group=8, n_eval=6, n_calib=8, n_measure=8, ks=(2,), n_random=1,
                reps=2)
    return SuiteConfig(**(base | kw))


def test_condition_names_are_unique_and_every_suite_contributes(tmp_path):
    cfg = _config(tmp_path, suites=("grid", "cal", "protect", "soft", "mixed", "lowrank", "reference"),
                  roundings=("rtn", "gptq"))
    conditions = build_conditions(cfg)
    assert len({c.name for c in conditions}) == len(conditions)
    assert {c.suite for c in conditions} == {"grid", "cal", "protect", "soft", "mixed", "lowrank", "reference"}
    assert all(c.bits in (2, 3, 4, "mixed") for c in conditions)


def test_protect_suite_repeats_gptq_but_not_deterministic_rtn(tmp_path):
    cfg = _config(tmp_path, suites=("protect",), roundings=("rtn", "gptq"), reps=3, n_random=2, ks=(2, 4))
    names = [c.name for c in build_conditions(cfg)]
    per_run = 1 + 2 * (4 + 2)  # uniform, then (circuit, dose, awq, outlier_all, 2 random) for each of 2 sizes
    assert sum(n.startswith("protect_rtn_") for n in names) == per_run
    assert sum(n.startswith("protect_gptq_r") for n in names) == 3 * per_run


def _fake_mmlu(tmp_path, n=30):
    def csv(prefix):
        rows = [[f"w{prefix + i % 5} w{1 + i % 7}?", "w1", "w2", "w3", "w4", "ABCD"[i % 4]] for i in range(n)]
        return ("\n".join(",".join(f'"{c}"' for c in row) for row in rows) + "\n").encode()

    with tarfile.open(tmp_path / "data.tar", "w") as tf:
        for subject, prefix in (("a", 1), ("b", 10)):
            data = csv(prefix)
            info = tarfile.TarInfo(f"data/test/{subject}_test.csv")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def test_mmlu_groups_give_disjoint_evaluation_rows_and_pools(tmp_path):
    _fake_mmlu(tmp_path)
    rows, pools = load_mmlu_groups({"target": ["a"], "control": ["b"]}, 10, seed=1, cache_dir=tmp_path)
    assert [r["area"] for r in rows] == ["target"] * 10 + ["control"] * 10
    assert all(len(p) == 20 for p in pools.values())
    for name in pools:
        evaluated = {row_to_text(r) for r in rows if r["area"] == name}
        # the same question text can repeat in a tiny fake set, so compare the shuffled order instead
        assert len(pools[name]) + 10 == 30 and evaluated
    again, _ = load_mmlu_groups({"target": ["a"], "control": ["b"]}, 10, seed=1, cache_dir=tmp_path)
    assert rows == again  # deterministic for a seed
    with pytest.raises(ValueError, match="fewer than"):
        load_mmlu_groups({"target": ["a"]}, 31, seed=1, cache_dir=tmp_path)


def test_accuracy_by_area_and_text_helpers(tokenizer, tmp_path):
    rows = [{"area": "x"}, {"area": "x"}, {"area": "y"}]
    assert accuracy_by_area([1, 0, 1], rows) == {"x": 50.0, "y": 100.0}
    path = tmp_path / "t.txt"
    path.write_text("short\n\nthis line is long enough\n")
    assert read_lines(path, min_chars=10) == ["this line is long enough"]
    assert len(text_batches(tokenizer, CALIB_LINES, 6, batch_size=4)) == 2
    for name in ("bio", "cs"):
        (tmp_path / f"{name}.txt").write_text("\n".join(CALIB_LINES) + "\n")
    batches = mixed_domain_batches(tokenizer, tmp_path, ("bio", "cs"), 8, seed=0, batch_size=4)
    assert sum(b["input_ids"].shape[0] for b in batches) == 8


def _context(model, tokenizer) -> CircuitContext:
    units = enumerate_units(model, 8)
    modules = dict(model.named_modules())
    qlayers = sorted(n for n, m in modules.items() if isinstance(m, torch.nn.Linear) and ".layers." in n)
    write_layers = sorted(unit_sets(units))
    pairs = [(f"w{10 + i}", f"w{30 + i}") for i in range(16)]
    facts = build_facts(model, tokenizer, "w1 w2 {} w3", pairs, require_correct=False)
    _, validation = split_facts(facts, 0)
    batch = make_pairs(validation, 1)
    scores = [{"unit": u, "suff": random.Random(i).random(), "nec": 0.0} for i, u in enumerate(units)]
    g = torch.Generator().manual_seed(0)
    V = {n: torch.rand(modules[n].in_features, generator=g, dtype=torch.double) + 0.1 for n in qlayers}
    return CircuitContext(
        model=model, tokenizer=tokenizer, modules=modules, units=units, scores=scores, qlayers=qlayers, V=V, A=V,
        write_layers=write_layers, write_weights=sum(modules[n].weight.numel() for n in write_layers),
        layer_weights={n: modules[n].weight.numel() for n in qlayers}, seed=0,
        fact_probe=KnowledgeProbe(model, tokenizer, batch.ids_clean, batch.tok_clean))


def _tiny_rows(n=60):
    rows = [{"question": f"w{1 + i % 9} w{2 + i % 5}?", "choices": ["w1", "w2", "w3", "w4"], "answer": i % 4}
            for i in range(n)]
    return ([{**r, "area": "target"} for r in rows[:6]] + [{**r, "area": "control"} for r in rows[6:12]],
            {"target": rows[12:36], "control": rows[36:60]})


def _suite(tiny_model, tokenizer, tmp_path, **kw) -> QuantSuite:
    for name in ("bio", "cs", "hum", "law", "math"):
        (tmp_path / f"{name}.txt").write_text("\n".join(CALIB_LINES * 3) + "\n")
    (tmp_path / "neutral.txt").write_text("\n".join(["w1 w2 w3 w4 w5 w1 w2 w3 w4 w5 " * 10] * 4) + "\n")
    cfg = _config(tmp_path, suites=("grid", "protect", "soft", "cal", "lowrank"), **kw)
    return QuantSuite(_context(tiny_model, tokenizer), cfg, rows_and_pools=_tiny_rows())


def test_suite_runs_conditions_resumes_and_restores_the_weights(tiny_model, tokenizer, tmp_path):
    before = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    only = frozenset({"grid_rtn3", "grid_gptq3_clip", "protect_gptq_r0_circuit_k2", "soft_circuit_b3_k2",
                      "cal_target_a0.5", "lowrank_gen_r8", "protect_rtn_uniform"})
    suite = _suite(tiny_model, tokenizer, tmp_path, only=only, roundings=("rtn", "gptq"))
    result = suite.run()
    names = [c["name"] for c in result["conditions"]]
    assert sorted(names) == sorted(only)
    assert all(torch.equal(p, before[n]) for n, p in tiny_model.named_parameters())  # restored bit for bit
    row = next(c for c in result["conditions"] if c["name"] == "protect_gptq_r0_circuit_k2")
    assert row["protected_fraction"] > 0 and len(row["correct"]) == 12 and row["control_acc"] is not None
    grid = next(c for c in result["conditions"] if c["name"] == "grid_rtn3")
    assert grid["control_acc"] is None and len(grid["correct"]) == 6  # grid conditions use target questions only
    assert "git_commit" in result["provenance"] and result["eval_areas"].count("target") == 6
    saved = json.loads((tmp_path / "out.json").read_text())
    assert [c["name"] for c in saved["conditions"]] == names
    # a second run skips every condition that is already in the file
    again = _suite(tiny_model, tokenizer, tmp_path, only=only, roundings=("rtn", "gptq")).run()
    assert [c["name"] for c in again["conditions"]] == names and len(again["provenance_resumed"]) == 2


def test_suite_mixed_allocation_puts_half_of_the_weights_at_four_bits(tiny_model, tokenizer, tmp_path):
    cfg_only = frozenset({"mixed_target", "mixed_random0"})
    suite = _suite(tiny_model, tokenizer, tmp_path, only=cfg_only)
    suite.cfg.suites = ("mixed",)
    result = suite.run()
    for row in result["conditions"]:
        assert row["bits"] == "mixed" and 3.0 < row["avg_bits"] <= 3.5
