"""Causal probes: channel-level activation patching on Linear inputs.

The per-domain statistics (``stats.py``) say a channel is *more active* on the
target domain; they do not say the model *needs* it.  Here we intervene: for a
chosen set of input channels of every Linear, replace their activation with a
reference value (the mean over other domains, or zero) and measure how much
the language-model NLL degrades on the target domain versus other domains.

Candidate channel sets (per Linear, ``k_frac`` of its input channels):

* ``tarc``       top channels by the target-anchored relative contrast
                 S_T = R_T - median_{d != T} R_d of per-channel A = E|h|
* ``importance`` top channels by the target activation A_T alone
* ``anti``       bottom channels by S_T (the other domains' side)
* ``zdiff``      top channels by |M_T - mean_o M_o| / sqrt(var_T + mean_o var_o),
                 the standardized difference of signed means (closest to the
                 mean-difference contrast Persona uses)
* ``random``     same size, matched on V = E[h^2] rank to a reference set and
                 disjoint from it (the null that S_T has to beat)
* ``energy_random``  same size, disjoint from the reference set, matched
                 channel by channel on the *perturbation energy* the patch
                 inflicts: E[(h - m)^2] on the target and on the others.  S_T
                 picks channels where the target differs from the others'
                 mean m, so patching with m perturbs target text more by
                 construction; this control gives the same dose to both sides,
                 so any remaining selectivity is not just dose.

Selectivity of an intervention is ``dNLL_target - mean_o dNLL_other`` (positive
= hurts the target more than the others).  Everything is forward-only: no
gradients, no weight changes (hooks are removed on exit, also on errors).
"""

from __future__ import annotations

import bisect
import random
from collections.abc import Iterable

import torch

from .allocate import ranknorm

SET_KINDS = ("tarc", "importance", "anti", "random", "energy_random", "zdiff")


# ---------------------------------------------------------------------------
# candidate channel sets
# ---------------------------------------------------------------------------

def target_contrast_scores(A: dict[str, torch.Tensor], target: str, others: list[str]) -> torch.Tensor:
    """Per-channel S_T = R_T - median_{d in others} R_d for one layer (A: domain -> A_j)."""
    if not others:
        raise ValueError("need at least one non-target domain")
    rank = lambda d: torch.tensor(ranknorm(A[d].double().tolist()))
    stacked = torch.stack([rank(d) for d in others])
    return rank(target) - stacked.median(dim=0).values  # lower median for even counts


def matched_random(
    V: torch.Tensor, ref: torch.Tensor, rng: random.Random, window: int = 5
) -> torch.Tensor:
    """Random channels with the same size and V-rank profile as ``ref``, disjoint from it.

    For each reference channel (in V order) one channel is drawn from the
    ``window`` nearest-by-V-rank channels that are neither in ``ref`` nor
    already drawn.
    """
    n, k = V.numel(), ref.numel()
    if 2 * k > n:
        raise ValueError(f"cannot draw {k} disjoint matched channels out of {n}")
    rank = V.double().argsort().argsort().tolist()
    excluded = set(ref.tolist())
    pool = sorted((j for j in range(n) if j not in excluded), key=rank.__getitem__)
    pool_rank = [rank[j] for j in pool]
    chosen = []
    for j in sorted(ref.tolist(), key=rank.__getitem__):
        pos = bisect.bisect_left(pool_rank, rank[j])
        lo = max(0, min(pos - window // 2, len(pool) - window))
        pick = rng.randrange(lo, min(lo + window, len(pool)))
        chosen.append(pool.pop(pick))
        pool_rank.pop(pick)
    return torch.tensor(sorted(chosen), dtype=torch.long)


def perturbation_energy(stats_layer: dict, value: torch.Tensor) -> torch.Tensor:
    """Per channel E[(h - m)^2] = V - 2 m M + m^2 for replacement ``value`` = m."""
    V, M = stats_layer["V"].double(), stats_layer["M"].double()
    m = value.double()
    return V - 2 * m * M + m * m


def energy_matched_random(
    feat: torch.Tensor, ref: torch.Tensor, rng: random.Random, window: int = 5
) -> torch.Tensor:
    """Disjoint random channels matched to ``ref`` in a per-channel feature space.

    ``feat`` is (n_channels, d), e.g. log energy on the target and on the
    others.  Features are standardized over the candidate pool, then each ref
    channel takes one of its ``window`` nearest unused pool channels at random.
    """
    n, k = feat.shape[0], ref.numel()
    if 2 * k > n:
        raise ValueError(f"cannot draw {k} disjoint matched channels out of {n}")
    pool_mask = torch.ones(n, dtype=torch.bool)
    pool_mask[ref] = False
    f = feat.double()
    f = (f - f[pool_mask].mean(0)) / f[pool_mask].std(0).clamp_min(1e-9)
    chosen = []
    for j in ref[torch.argsort(feat[ref, 0])].tolist():
        pool = pool_mask.nonzero().squeeze(1)
        dist = (f[pool] - f[j]).square().sum(1)
        near = dist.topk(min(window, pool.numel()), largest=False).indices
        pick = int(pool[near[rng.randrange(near.numel())]])
        chosen.append(pick)
        pool_mask[pick] = False
    return torch.tensor(sorted(chosen), dtype=torch.long)


def select_channels(
    domains_stats: dict[str, dict],
    target: str,
    others: list[str],
    k_frac: float,
    kind: str,
    *,
    ref_sets: dict[str, torch.Tensor] | None = None,
    values: dict[str, torch.Tensor] | None = None,
    seed: int = 0,
    only_layers: Iterable[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Per-layer input-channel indices for one candidate-set ``kind``.

    ``kind="random"`` / ``"energy_random"`` need ``ref_sets`` (the sets to
    match, usually the ``tarc`` sets for the same ``k_frac``); ``energy_random``
    also needs the replacement ``values`` the patch will use.
    """
    if kind not in SET_KINDS:
        raise ValueError(f"unknown set kind {kind!r}; use one of {SET_KINDS}")
    if not 0 < k_frac <= 0.5:
        raise ValueError(f"k_frac must be in (0, 0.5], got {k_frac}")
    if kind in ("random", "energy_random") and ref_sets is None:
        raise ValueError(f"kind={kind!r} needs ref_sets to match")
    if kind == "energy_random" and values is None:
        raise ValueError("kind='energy_random' needs the replacement values")
    rng = random.Random(seed)
    layers = sorted(domains_stats[target])
    if only_layers is not None:
        keep = set(only_layers)
        layers = [name for name in layers if name in keep]
    out = {}
    for name in layers:
        A = {d: domains_stats[d][name]["A"] for d in [target, *others]}
        n = A[target].numel()
        k = max(1, round(k_frac * n))
        if kind == "random":
            out[name] = matched_random(domains_stats[target][name]["V"], ref_sets[name], rng)
            continue
        if kind == "energy_random":
            floor = 1e-12
            e_t = perturbation_energy(domains_stats[target][name], values[name]).clamp_min(floor)
            e_o = torch.stack([perturbation_energy(domains_stats[d][name], values[name])
                               for d in others]).mean(0).clamp_min(floor)
            out[name] = energy_matched_random(
                torch.stack([e_t.log(), e_o.log()], dim=1), ref_sets[name], rng)
            continue
        if kind == "zdiff":
            st = domains_stats
            m_o = torch.stack([st[d][name]["M"].double() for d in others]).mean(0)
            var_t = (st[target][name]["V"].double() - st[target][name]["M"].double() ** 2).clamp_min(0)
            var_o = torch.stack([(st[d][name]["V"].double() - st[d][name]["M"].double() ** 2).clamp_min(0)
                                 for d in others]).mean(0)
            z = (st[target][name]["M"].double() - m_o).abs() / (var_t + var_o).sqrt().clamp_min(1e-12)
            out[name] = z.topk(k).indices.sort().values
            continue
        if kind == "importance":
            idx = A[target].topk(k).indices
        else:
            s = target_contrast_scores(A, target, others)
            idx = s.topk(k).indices if kind == "tarc" else (-s).topk(k).indices
        out[name] = idx.sort().values
    return out


def reference_means(
    domains_stats: dict[str, dict], others: list[str], replace: str = "mean"
) -> dict[str, torch.Tensor]:
    """Per-layer replacement vectors: mean of the others' signed means, or zeros."""
    if replace not in ("mean", "zero"):
        raise ValueError(f"unknown replacement {replace!r}; use 'mean' or 'zero'")
    first = domains_stats[others[0]] if others else next(iter(domains_stats.values()))
    if replace == "zero":
        return {name: torch.zeros_like(v["M"]) for name, v in first.items()}
    return {
        name: torch.stack([domains_stats[d][name]["M"] for d in others]).mean(0)
        for name in first
    }


# ---------------------------------------------------------------------------
# the intervention
# ---------------------------------------------------------------------------

class ChannelPatch:
    """Context manager: overwrite chosen input channels of the named Linears.

    ``sets`` maps layer name -> channel indices; ``values`` maps layer name ->
    the replacement vector over *all* input channels (only the selected ones
    are used).  Layers with an empty set are left untouched.  Hooks are always
    removed on exit; the model's weights are never modified.
    """

    def __init__(self, model: torch.nn.Module, sets: dict[str, torch.Tensor],
                 values: dict[str, torch.Tensor]):
        self.model, self.sets, self.values = model, sets, values
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "ChannelPatch":
        modules = dict(self.model.named_modules())
        try:
            for name, idx in self.sets.items():
                if idx.numel() == 0:
                    continue
                module = modules.get(name)
                if not isinstance(module, torch.nn.Linear):
                    raise KeyError(f"patch target {name!r} is not an nn.Linear in the model")
                if int(idx.max()) >= module.in_features:
                    raise ValueError(f"{name}: channel {int(idx.max())} >= in_features")
                self._hooks.append(
                    module.register_forward_pre_hook(self._make_hook(idx, self.values[name]))
                )
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    @staticmethod
    def _make_hook(idx: torch.Tensor, value: torch.Tensor):
        def hook(module, args):
            x = args[0].clone()
            x[..., idx.to(x.device)] = value[idx].to(device=x.device, dtype=x.dtype)
            return (x, *args[1:])

        return hook

    def __exit__(self, exc_type, exc, tb) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def line_nll(
    model: torch.nn.Module, tokenizer, lines: list[str], *, batch_size: int = 8, max_len: int = 128
) -> list[float]:
    """Mean next-token NLL per line, over valid (non-padding) tokens only."""
    model.eval()
    device = next(model.parameters()).device
    side, tokenizer.padding_side = tokenizer.padding_side, "right"  # positions stay 0..n-1
    try:
        out: list[float] = []
        for i in range(0, len(lines), batch_size):
            batch = tokenizer(lines[i:i + batch_size], return_tensors="pt", padding=True,
                              truncation=True, max_length=max_len)
            ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
            nll = torch.nn.functional.cross_entropy(
                logits.transpose(1, 2), ids[:, 1:], reduction="none"
            )
            valid = mask[:, 1:].float()
            out.extend(((nll * valid).sum(1) / valid.sum(1).clamp_min(1)).tolist())
    finally:
        tokenizer.padding_side = side
    return out


def bootstrap_ci(
    delta_target: list[float], deltas_other: dict[str, list[float]],
    *, n_boot: int = 1000, seed: int = 0, level: float = 0.95,
) -> tuple[float, float, float]:
    """Selectivity ``mean(dT) - mean_o mean(dO)`` with a percentile bootstrap CI over lines."""
    g = torch.Generator().manual_seed(seed)
    groups = [torch.tensor(delta_target, dtype=torch.float64),
              *[torch.tensor(v, dtype=torch.float64) for v in deltas_other.values()]]

    def stat(gs):
        return gs[0].mean() - torch.stack([x.mean() for x in gs[1:]]).mean()

    point = float(stat(groups))
    boots = torch.empty(n_boot, dtype=torch.float64)
    for b in range(n_boot):
        boots[b] = stat([x[torch.randint(len(x), (len(x),), generator=g)] for x in groups])
    lo, hi = torch.quantile(boots, torch.tensor([(1 - level) / 2, (1 + level) / 2],
                                                dtype=torch.float64)).tolist()
    return point, lo, hi


def _set_summary(sets: dict[str, torch.Tensor], domains_stats: dict[str, dict], target: str,
                 others: list[str], values: dict[str, torch.Tensor]) -> dict:
    """Size, activation-mass coverage and patch dose of a set (mean over layers).

    ``E_*_share`` is the set's share of the layer's total perturbation energy
    sum_j E[(h_j - m_j)^2] on the target / on the other domains.
    """
    layers = list(sets)
    share = lambda x, idx: float(x[idx].sum() / x.sum())
    mean = lambda f: sum(f(n, sets[n]) for n in layers) / len(layers)
    e_t = lambda n: perturbation_energy(domains_stats[target][n], values[n])
    e_o = lambda n: torch.stack(
        [perturbation_energy(domains_stats[d][n], values[n]) for d in others]).mean(0)
    return {
        "channels": sum(int(i.numel()) for i in sets.values()),
        "A_mass_share": mean(lambda n, i: share(domains_stats[target][n]["A"], i)),
        "V_mass_share": mean(lambda n, i: share(domains_stats[target][n]["V"], i)),
        "E_target_share": mean(lambda n, i: share(e_t(n), i)),
        "E_others_share": mean(lambda n, i: share(e_o(n), i)),
    }


def causal_probe(
    model: torch.nn.Module,
    tokenizer,
    domains_stats: dict[str, dict],
    *,
    target: str,
    others: list[str],
    eval_target: tuple[str, list[str]],
    eval_others: dict[str, list[str]],
    k_fracs: list[float],
    n_random: int = 3,
    replace: str = "mean",
    seed: int = 42,
    only_layers: Iterable[str] | None = None,
    batch_size: int = 8,
    max_len: int = 128,
    n_boot: int = 1000,
    log=lambda msg: None,
) -> dict:
    """Ablate candidate channel sets and report per-domain NLL degradation.

    ``target``/``others`` name *stats* domains (used to pick channels and the
    replacement mean); the eval texts are separate, ideally held out from the
    calibration lines the stats came from.
    """
    eval_sets = {eval_target[0]: eval_target[1], **eval_others}
    values = reference_means(domains_stats, others, replace)
    base = {d: line_nll(model, tokenizer, lines, batch_size=batch_size, max_len=max_len)
            for d, lines in eval_sets.items()}
    report = {
        "config": {"target": target, "others": others, "eval_target": eval_target[0],
                   "eval_others": list(eval_others), "k_fracs": k_fracs, "n_random": n_random,
                   "replace": replace, "seed": seed, "n_eval": {d: len(v) for d, v in eval_sets.items()}},
        "baseline_nll": {d: sum(v) / len(v) for d, v in base.items()},
        "conditions": [],
    }

    def run(kind: str, k_frac: float, sets: dict[str, torch.Tensor], rep: int | None = None):
        with ChannelPatch(model, sets, values):
            patched = {d: line_nll(model, tokenizer, lines, batch_size=batch_size, max_len=max_len)
                       for d, lines in eval_sets.items()}
        delta = {d: [p - b for p, b in zip(patched[d], base[d])] for d in eval_sets}
        d_other = {d: delta[d] for d in eval_others}
        sel, lo, hi = bootstrap_ci(delta[eval_target[0]], d_other, n_boot=n_boot, seed=seed)
        entry = {
            "kind": kind, "k_frac": k_frac, "replicate": rep,
            "set": _set_summary(sets, domains_stats, target, others, values),
            "delta_nll": {d: sum(v) / len(v) for d, v in delta.items()},
            "selectivity": sel, "selectivity_ci95": [lo, hi],
        }
        report["conditions"].append(entry)
        log(f"k={k_frac} {kind}{'' if rep is None else f'#{rep}'}: "
            f"dNLL {entry['delta_nll']} selectivity {sel:+.4f} [{lo:+.4f}, {hi:+.4f}]")

    for k_frac in k_fracs:
        sets = {kind: select_channels(domains_stats, target, others, k_frac, kind,
                                      only_layers=only_layers)
                for kind in ("tarc", "importance", "anti")}
        for kind, s in sets.items():
            run(kind, k_frac, s)
        for kind in ("random", "energy_random"):
            for rep in range(n_random):
                rand = select_channels(domains_stats, target, others, k_frac, kind,
                                       ref_sets=sets["tarc"], values=values, seed=seed + rep,
                                       only_layers=only_layers)
                run(kind, k_frac, rand, rep)
    return report
