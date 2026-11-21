"""Units: slices of the residual-stream write projections. A circuit is a set of units.

A unit is a block of input channels of a projection that writes to the residual stream: an attention head
(a block of ``o_proj`` or ``out_proj`` inputs) or a group of MLP neurons (a block of ``down_proj`` inputs).
Patching a unit replaces what that head or neuron group writes; quantizing its weight columns damages what
reads it. Units may span many layers, so a circuit is measured jointly, not layer by layer.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Unit:
    layer: str   # module name of the write projection
    start: int   # input-channel slice [start, stop)
    stop: int
    kind: str    # "head" or "mlp"

    @property
    def size(self) -> int:
        return self.stop - self.start

    @property
    def cls(self) -> str:
        """Size class: random controls are drawn from the same class."""
        return f"{self.kind}{self.size}"


def enumerate_units(model: torch.nn.Module, mlp_group: int = 64) -> list[Unit]:
    """Heads and MLP neuron groups over the write projections of the text layers."""
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    units = []
    for name, m in model.named_modules():
        if not isinstance(m, torch.nn.Linear) or ".layers." not in name:
            continue
        if name.endswith("mlp.down_proj"):
            kind, size = "mlp", mlp_group
        elif name.endswith("self_attn.o_proj"):
            kind, n_heads = "head", cfg.num_attention_heads
        elif name.endswith("linear_attn.out_proj"):
            kind, n_heads = "head", cfg.linear_num_value_heads
        else:
            continue
        if kind == "head":
            if m.in_features % n_heads:
                raise ValueError(f"{name}: in_features {m.in_features} not divisible by {n_heads} heads")
            size = m.in_features // n_heads
        units += [Unit(name, s, min(s + size, m.in_features), kind)
                  for s in range(0, m.in_features, size)]
    if not units:
        raise ValueError("no write projections found (expected o_proj / out_proj / down_proj)")
    return units


def unit_sets(units: list[Unit]) -> dict[str, torch.Tensor]:
    """Per-layer sorted input-channel indices covered by ``units``."""
    out: dict[str, list[int]] = {}
    for u in units:
        out.setdefault(u.layer, []).extend(range(u.start, u.stop))
    return {layer: torch.tensor(sorted(set(idx)), dtype=torch.long) for layer, idx in out.items()}


def match_random(all_units: list[Unit], circuit: list[Unit], rng: random.Random) -> list[Unit]:
    """Random units of the same size classes as ``circuit``, disjoint from it."""
    taken = set(circuit)
    pools: dict[str, list[Unit]] = {}
    for u in all_units:
        if u not in taken:
            pools.setdefault(u.cls, []).append(u)
    drawn = []
    for u in circuit:
        pool = pools.get(u.cls, [])
        if not pool:
            raise ValueError(f"no unused unit of class {u.cls} left for a matched control")
        drawn.append(pool.pop(rng.randrange(len(pool))))
    return drawn


def class_matched_top(ranked_units: list[Unit], like: list[Unit]) -> list[Unit]:
    """The best-ranked units with the same class counts as ``like`` (for example 20 heads and 12 MLP groups)."""
    need: dict[str, int] = {}
    for unit in like:
        need[unit.cls] = need.get(unit.cls, 0) + 1
    chosen: list[Unit] = []
    for unit in ranked_units:
        if need.get(unit.cls, 0) > 0:
            chosen.append(unit)
            need[unit.cls] -= 1
    return chosen
