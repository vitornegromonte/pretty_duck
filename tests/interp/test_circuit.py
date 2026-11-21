import random

import pytest
import torch

from compress.interp.damage import KnowledgeProbe, additivity
from compress.interp.minimal_pairs import build_facts, make_pairs, split_facts
from compress.interp.scan import Scanner, scan_units, top_units
from compress.interp.units import Unit, enumerate_units, match_random, unit_sets
from tests.conftest import CALIB_LINES

TEMPLATE = "w1 w2 {} w3"
PAIRS = [(f"w{10 + i}", f"w{30 + i}") for i in range(16)]  # distinct subjects and answers


def _facts(model, tokenizer, **kw):
    return build_facts(model, tokenizer, TEMPLATE, PAIRS, require_correct=False, **kw)


def test_units_tile_every_write_projection_and_follow_head_counts(tiny_model):
    units = enumerate_units(tiny_model, mlp_group=8)
    by_layer: dict[str, list[Unit]] = {}
    for u in units:
        by_layer.setdefault(u.layer, []).append(u)
    modules = dict(tiny_model.named_modules())
    assert len(by_layer) == 4  # o_proj and down_proj in each of 2 layers
    for layer, us in by_layer.items():
        edge = 0
        for u in sorted(us, key=lambda u: u.start):
            assert u.start == edge and u.stop > u.start
            edge = u.stop
        assert edge == modules[layer].in_features
    assert sum(u.kind == "head" for u in units) == 2 * 2  # 2 heads x 2 layers
    assert {u.cls for u in units} == {"head8", "mlp8"}
    sets = unit_sets(units)
    assert all(torch.equal(v, torch.arange(modules[k].in_features)) for k, v in sets.items())


def test_facts_share_a_length_and_pairs_swap_only_the_subject(tiny_model, tokenizer):
    facts = _facts(tiny_model, tokenizer)
    assert len(facts) == len(PAIRS)
    batch = make_pairs(facts, seed=0)
    assert batch.ids_clean.shape == batch.ids_corrupt.shape
    assert bool((batch.tok_clean != batch.tok_corrupt).all())
    differs = (batch.ids_clean != batch.ids_corrupt).sum(1)
    assert bool((differs == 1).all())  # exactly the subject token
    a, b = split_facts(facts, seed=1)
    assert not {f.subject for f in a} & {f.subject for f in b} and len(a) + len(b) == len(facts)
    with pytest.raises(ValueError, match="at least two"):
        make_pairs(facts[:1])


def test_patching_nothing_changes_nothing_and_self_patching_is_identity(tiny_model, tokenizer):
    layers = sorted(unit_sets(enumerate_units(tiny_model, 8)))
    sc = Scanner(tiny_model, make_pairs(_facts(tiny_model, tokenizer), 0), layers)
    empty = {layer: torch.tensor([], dtype=torch.long) for layer in layers}
    assert sc.restoration(empty) == pytest.approx(0.0, abs=1e-6)
    assert sc.necessity(empty) == pytest.approx(0.0, abs=1e-6)
    everything = unit_sets(enumerate_units(tiny_model, 8))
    # corrupt activations patched into the corrupt run, and clean into clean: unchanged logits
    assert torch.allclose(sc.patched_ld(everything, sc.corrupt_in, sc.ids_corrupt), sc.ld_corrupt, atol=1e-4)
    assert torch.allclose(sc.patched_ld(everything, sc.clean_in, sc.ids_clean), sc.ld_clean, atol=1e-4)


def test_scan_scores_every_unit_and_bootstrap_brackets_the_point(tiny_model, tokenizer):
    units = enumerate_units(tiny_model, 8)
    sc = Scanner(tiny_model, make_pairs(_facts(tiny_model, tokenizer), 0), sorted(unit_sets(units)))
    scores = scan_units(sc, units)
    assert len(scores) == len(units) and all(set(s) == {"unit", "suff", "nec"} for s in scores)
    top = top_units(scores, 3)
    assert len(top) == 3 and scores[[s["unit"] for s in scores].index(top[0])]["suff"] == max(
        s["suff"] for s in scores)
    point, lo, hi = sc.restoration(unit_sets(top), n_boot=50)
    assert lo <= point <= hi
    assert point == pytest.approx(sc.restoration(unit_sets(top)))


def test_match_random_is_disjoint_and_class_matched():
    units = [Unit("a", i * 8, i * 8 + 8, "head") for i in range(4)] + [
        Unit("b", i * 8, i * 8 + 8, "mlp") for i in range(8)]
    circuit = [units[0], units[4], units[5]]
    drawn = match_random(units, circuit, random.Random(0))
    assert sorted(u.cls for u in drawn) == sorted(u.cls for u in circuit)
    assert not set(drawn) & set(circuit) and len(set(drawn)) == 3
    with pytest.raises(ValueError, match="no unused unit"):
        match_random(units[:5], units[:4], random.Random(0))


def test_quantization_damage_restores_weights_and_additivity_is_reported(tiny_model, tokenizer):
    facts = _facts(tiny_model, tokenizer)
    batch = make_pairs(facts, 0)
    probe = KnowledgeProbe(tiny_model, tokenizer, batch.ids_clean, batch.tok_clean,
                           neutral_lines=CALIB_LINES[:4], batch_size=4)
    units = enumerate_units(tiny_model, 8)
    before = {n: p.detach().clone() for n, p in tiny_model.named_parameters()}
    dmg = probe.damage(unit_sets(units), 2)
    assert set(dmg) == {"dlogp", "dacc", "dnll"}
    assert dmg["dlogp"] != 0
    assert all(torch.equal(p, before[n]) for n, p in tiny_model.named_parameters())
    rep = additivity(probe, units[:3], 2)
    assert rep["n"] == 3 and set(rep) == {"joint", "sum_single", "n"}
    assert probe.damage(unit_sets([]), 2)["dlogp"] == pytest.approx(0.0, abs=1e-6)


def test_dose_is_reported_and_dose_matching_picks_the_closest_sets(tiny_model, tokenizer):
    from compress.interp.damage import dose_matched_random, dose_of
    batch = make_pairs(_facts(tiny_model, tokenizer), 0)
    probe = KnowledgeProbe(tiny_model, tokenizer, batch.ids_clean, batch.tok_clean)
    units = enumerate_units(tiny_model, 8)
    V = {layer: torch.rand(tiny_model.get_submodule(layer).in_features) + 0.1 for layer in unit_sets(units)}
    circ = units[:3]
    dmg = probe.damage(unit_sets(circ), 2, V)
    assert dmg["dose"] == pytest.approx(dose_of(tiny_model, circ, 2, V)) and dmg["dose"] > 0
    assert "dose" not in probe.damage(unit_sets(circ), 2)
    picked = dose_matched_random(tiny_model, units, circ, V, 2, random.Random(0), n_keep=3, n_candidates=40)
    target = dose_of(tiny_model, circ, 2, V)
    gaps = [abs(dose_of(tiny_model, c, 2, V) - target) for c in picked]
    rand = [abs(dose_of(tiny_model, match_random(units, circ, random.Random(i)), 2, V) - target) for i in range(10)]
    assert len(picked) == 3 and max(gaps) <= sorted(rand)[len(rand) // 2] + 1e-9


def test_core_and_exclusion_follow_the_top_lists():
    from compress.interp.scan import core_units, excluding
    u = [Unit("l", i, i + 1, "mlp") for i in range(6)]
    mk = lambda order: [{"unit": u[i], "suff": float(len(order) - r), "nec": 0.0} for r, i in enumerate(order)]
    rankings = {"a": mk([0, 1, 2, 3, 4, 5]), "b": mk([1, 0, 4, 5, 2, 3]), "c": mk([5, 1, 0, 2, 3, 4])}
    assert core_units(rankings, 2, 3) == {u[1]}  # top-2 sets are {0,1}, {1,0}, {5,1}
    assert core_units(rankings, 3, 3) == {u[0], u[1]}
    assert core_units(rankings, 2, 2) == {u[0], u[1]}  # unit 0 is in the top 2 of a and b
    left = excluding(rankings["a"], {u[0], u[1]})
    assert [s["unit"] for s in left] == u[2:] and top_units(left, 2) == [u[2], u[3]]


def test_batched_neuron_scan_matches_one_unit_at_a_time(tiny_model, tokenizer):
    from compress.interp.scan import scan_neurons
    units = enumerate_units(tiny_model, 8)
    layers = sorted(unit_sets(units))
    sc = Scanner(tiny_model, make_pairs(_facts(tiny_model, tokenizer), 0), layers)
    down = "model.layers.1.mlp.down_proj"
    neurons = [0, 3, 7, 12, 31]
    for direction, one in (("suff", sc.restoration), ("nec", sc.necessity)):
        for chunk in (2, 16):  # a partial last chunk and a single full chunk
            batched = scan_neurons(sc, down, neurons, chunk=chunk, direction=direction)
            single = [one({down: torch.tensor([j])}) for j in neurons]
            assert batched == pytest.approx(single, abs=1e-4)
    with pytest.raises(ValueError, match="direction"):
        scan_neurons(sc, down, neurons, direction="x")
    # hooks are gone and weights untouched afterwards
    assert sc.restoration({down: torch.tensor([], dtype=torch.long)}) == pytest.approx(0.0, abs=1e-6)


def test_offloaded_scanner_gives_the_same_numbers(tiny_model, tokenizer):
    units = enumerate_units(tiny_model, 8)
    layers = sorted(unit_sets(units))
    sc = Scanner(tiny_model, make_pairs(_facts(tiny_model, tokenizer), 0), layers)
    sets = unit_sets(units[:4])
    before = (sc.restoration(sets), sc.necessity(sets))
    sc.offload()
    assert all(v.device.type == "cpu" for v in sc.clean_in.values())
    assert (sc.restoration(sets), sc.necessity(sets)) == pytest.approx(before, abs=1e-6)
