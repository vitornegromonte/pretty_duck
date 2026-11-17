"""Circuit-level probes on templated factual recall.

A *unit* is a slice of the input of a residual-stream write projection:
an attention head (a block of ``o_proj`` / ``out_proj`` input channels) or a
group of MLP neurons (a block of ``down_proj`` input channels).  Patching a
unit's input channels replaces that head's / neuron group's output; quantizing
the matching weight columns damages exactly what reads it.  A *circuit* is a
set of units that may span many layers, so effects are measured jointly, not
layer by layer.

Task: entity-swap minimal pairs.  The clean prompt asks about subject ``s_c``
(answer first-token ``a_c``), the corrupt one about ``s_k`` (``a_k``); both have
identical token length.  The metric is the logit difference
``LD = logit[a_c] - logit[a_k]`` at the last position.

* sufficiency of a unit set: patch the clean activations into the corrupt run,
  ``restoration = sum(LD_patched - LD_corrupt) / sum(LD_clean - LD_corrupt)``;
* necessity: patch the corrupt activations into the clean run,
  ``destroyed = sum(LD_clean - LD_patched) / sum(LD_clean - LD_corrupt)``.

Forward-only, no gradients; weights are restored on exit.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .pair_patching import SuffixCapture, SuffixPatch, _ratio_ci
from .patching import line_nll
from .quant_probe import QuantizeColumns

WRITE_SUFFIXES = ("self_attn.o_proj", "linear_attn.out_proj", "mlp.down_proj")


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# facts and minimal pairs
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    subject: str
    answer: str
    ids: list[int]   # prompt token ids
    token: int       # first token of " <answer>"


@torch.no_grad()
def build_facts(model, tokenizer, template: str, pairs: list[tuple[str, str]], *,
                require_correct: bool = True, batch_size: int = 32) -> list[Fact]:
    """Facts whose prompts share one token length (the most common) and tokenize cleanly.

    With ``require_correct`` only facts whose next-token argmax is the answer
    token are kept, so patching measures knowledge the model actually has.
    """
    facts = []
    for subject, answer in pairs:
        prompt = template.format(subject)
        p_ids = tokenizer(prompt)["input_ids"]
        f_ids = tokenizer(prompt + " " + answer)["input_ids"]
        if len(f_ids) > len(p_ids) and f_ids[: len(p_ids)] == p_ids:
            facts.append(Fact(subject, answer, p_ids, f_ids[len(p_ids)]))
    if not facts:
        return []
    lengths = [len(f.ids) for f in facts]
    mode = max(set(lengths), key=lengths.count)
    facts = [f for f in facts if len(f.ids) == mode]
    if not require_correct:
        return facts
    device = next(model.parameters()).device
    kept = []
    for i in range(0, len(facts), batch_size):
        chunk = facts[i:i + batch_size]
        ids = torch.tensor([f.ids for f in chunk], device=device)
        top = model(input_ids=ids).logits[:, -1].argmax(-1).tolist()
        kept += [f for f, t in zip(chunk, top) if t == f.token]
    return kept


@dataclass
class PairBatch:
    ids_clean: torch.Tensor     # (N, L)
    ids_corrupt: torch.Tensor   # (N, L)
    tok_clean: torch.Tensor     # (N,)
    tok_corrupt: torch.Tensor   # (N,)


def make_pairs(facts: list[Fact], seed: int = 0) -> PairBatch:
    """Each fact once as the clean side, paired with a random fact of a different answer token."""
    rng = random.Random(seed)
    clean, corrupt = [], []
    for i, f in enumerate(facts):
        options = [g for j, g in enumerate(facts) if j != i and g.token != f.token]
        if options:
            clean.append(f)
            corrupt.append(rng.choice(options))
    if not clean:
        raise ValueError("need at least two facts with different answer tokens")
    return PairBatch(
        torch.tensor([f.ids for f in clean]), torch.tensor([g.ids for g in corrupt]),
        torch.tensor([f.token for f in clean]), torch.tensor([g.token for g in corrupt]))


def split_facts(facts: list[Fact], seed: int = 0) -> tuple[list[Fact], list[Fact]]:
    shuffled = facts[:]
    random.Random(seed).shuffle(shuffled)
    half = len(shuffled) // 2
    return shuffled[:half], shuffled[half:]


# ---------------------------------------------------------------------------
# patching
# ---------------------------------------------------------------------------

class Scanner:
    """Clean/corrupt activations of the write projections for one pair batch."""

    def __init__(self, model: torch.nn.Module, batch: PairBatch, layers: list[str]):
        self.model, self.batch, self.layers = model, batch, layers
        self.device = next(model.parameters()).device
        self.L = batch.ids_clean.shape[1]
        self.ids_clean = batch.ids_clean.to(self.device)
        self.ids_corrupt = batch.ids_corrupt.to(self.device)
        self.tok_c = batch.tok_clean.to(self.device)
        self.tok_k = batch.tok_corrupt.to(self.device)
        model.eval()
        with torch.no_grad():
            with SuffixCapture(model, layers, self.L) as cap:
                self.ld_clean = self._ld(self.ids_clean)
            self.clean_in = cap.inputs
            with SuffixCapture(model, layers, self.L) as cap:
                self.ld_corrupt = self._ld(self.ids_corrupt)
            self.corrupt_in = cap.inputs
        self.den = self.ld_clean - self.ld_corrupt

    def offload(self) -> "Scanner":
        """Move the cached activations to the CPU; patching then copies only the patched slices back."""
        self.clean_in = {k: v.cpu() for k, v in self.clean_in.items()}
        self.corrupt_in = {k: v.cpu() for k, v in self.corrupt_in.items()}
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def _ld(self, ids: torch.Tensor) -> torch.Tensor:
        return self._ld_from(_last_logits(self.model, ids))  # last position only: no (batch, seq, vocab) tensor

    def _ld_from(self, logits: torch.Tensor) -> torch.Tensor:
        """Logit difference from last-position logits; the pair batch may be tiled along dim 0."""
        reps = logits.shape[0] // self.tok_c.shape[0]
        tc, tk = self.tok_c.repeat(reps), self.tok_k.repeat(reps)
        return (logits.gather(1, tc[:, None]) - logits.gather(1, tk[:, None])).squeeze(1)

    @torch.no_grad()
    def patched_ld(self, sets: dict[str, torch.Tensor], source: dict, target_ids: torch.Tensor) -> torch.Tensor:
        with SuffixPatch(self.model, sets, source, self.L):
            return self._ld(target_ids)

    def restoration(self, sets: dict[str, torch.Tensor], n_boot: int = 0, seed: int = 0):
        """Sufficiency (clean into corrupt); returns the point value, or (point, lo, hi)."""
        ld = self.patched_ld(sets, self.clean_in, self.ids_corrupt)
        return self._ratio(ld - self.ld_corrupt, n_boot, seed)

    def necessity(self, sets: dict[str, torch.Tensor], n_boot: int = 0, seed: int = 0):
        """Fraction of the LD destroyed by patching the corrupt activations into the clean run."""
        ld = self.patched_ld(sets, self.corrupt_in, self.ids_clean)
        return self._ratio(self.ld_clean - ld, n_boot, seed)

    def _ratio(self, num: torch.Tensor, n_boot: int, seed: int):
        num, den = num.double().cpu(), self.den.double().cpu()
        if n_boot:
            return _ratio_ci(num, den, n_boot, seed)
        return float(num.sum() / den.sum())


def _last_logits(model, ids: torch.Tensor) -> torch.Tensor:
    """Logits of the last position only (avoids a (batch, seq, vocab) tensor when supported)."""
    try:
        return model(input_ids=ids, logits_to_keep=1).logits[:, -1].float()
    except TypeError:
        return model(input_ids=ids).logits[:, -1].float()


@torch.no_grad()
def scan_neurons(scanner: Scanner, layer: str, neurons: list[int], chunk: int = 16,
                 direction: str = "suff") -> list[float]:
    """Single-neuron sufficiency (or necessity) for many neurons of one write projection.

    Each sequence of a chunk carries a *different* patched neuron: the batch is
    ``chunk`` copies of the pair batch, and a pre-hook on ``layer`` overwrites
    input channel ``neurons[u]`` of copy ``u`` with the source activation.  The
    result equals one :meth:`Scanner.restoration` / :meth:`Scanner.necessity`
    call per neuron, in a fraction of the forward passes.
    """
    if direction not in ("suff", "nec"):
        raise ValueError(f"direction must be 'suff' or 'nec', got {direction!r}")
    module = dict(scanner.model.named_modules())[layer]
    src = (scanner.clean_in if direction == "suff" else scanner.corrupt_in)[layer]
    target = scanner.ids_corrupt if direction == "suff" else scanner.ids_clean
    n = target.shape[0]
    den = scanner.den.sum()
    out: list[float] = []
    for i in range(0, len(neurons), chunk):
        js = torch.tensor(neurons[i:i + chunk], device=scanner.device)
        u = len(js)

        def hook(mod, args, js=js, u=u):
            x = args[0].clone()
            xv = x.view(u, n, *x.shape[1:])
            xv[torch.arange(u, device=x.device), :, :, js] = src[:, :, js].permute(2, 0, 1).to(x.dtype)
            return (x, *args[1:])

        handle = module.register_forward_pre_hook(hook)
        try:
            ld = scanner._ld_from(_last_logits(scanner.model, target.repeat(u, 1))).view(u, n)
        finally:
            handle.remove()
        num = (ld - scanner.ld_corrupt[None]).sum(1) if direction == "suff" else (scanner.ld_clean[None] - ld).sum(1)
        out += (num / den).tolist()
    return out


def scan_units(scanner: Scanner, units: list[Unit], log=lambda msg: None) -> list[dict]:
    """Single-unit sufficiency and necessity for every unit."""
    out = []
    for i, u in enumerate(units):
        sets = unit_sets([u])
        out.append({"unit": u, "suff": scanner.restoration(sets), "nec": scanner.necessity(sets)})
        if (i + 1) % 400 == 0 or i + 1 == len(units):
            log(f"scanned {i + 1}/{len(units)} units")
    return out


def top_units(scores: list[dict], k: int, key: str = "suff") -> list[Unit]:
    return [s["unit"] for s in sorted(scores, key=lambda s: -s[key])[:k]]


def core_units(rankings: dict[str, list[dict]], top_k: int, min_domains: int,
               key: str = "suff") -> set[Unit]:
    """Units that sit in the top ``top_k`` of at least ``min_domains`` domains' rankings."""
    counts: dict[Unit, int] = {}
    for scores in rankings.values():
        for u in top_units(scores, top_k, key):
            counts[u] = counts.get(u, 0) + 1
    return {u for u, c in counts.items() if c >= min_domains}


def excluding(scores: list[dict], banned: set[Unit]) -> list[dict]:
    """The scored units that are not in ``banned``."""
    return [s for s in scores if s["unit"] not in banned]


# ---------------------------------------------------------------------------
# quantization damage
# ---------------------------------------------------------------------------

class KnowledgeProbe:
    """Baseline answer statistics of clean prompts, and damage from column quantization."""

    def __init__(self, model, tokenizer, ids: torch.Tensor, tokens: torch.Tensor,
                 neutral_lines: list[str] | None = None, batch_size: int = 16, max_len: int = 128):
        self.model, self.tokenizer = model, tokenizer
        self.device = next(model.parameters()).device
        self.ids, self.tokens = ids.to(self.device), tokens.to(self.device)
        self.neutral, self.kw = neutral_lines, dict(batch_size=batch_size, max_len=max_len)
        self.base = self._stats()

    def stats(self) -> dict:
        """Mean answer log-prob, top-1 accuracy (and neutral NLL) of the model as it is now."""
        return self._stats()

    @torch.no_grad()
    def _stats(self) -> dict:
        logits = self.model(input_ids=self.ids).logits[:, -1].float()
        logp = logits.log_softmax(-1).gather(1, self.tokens[:, None]).squeeze(1)
        logp = torch.nan_to_num(logp, nan=-30.0, neginf=-30.0).clamp_min(-30.0)  # broken models score -30
        out = {"logp": float(logp.mean()), "acc": float((logits.argmax(-1) == self.tokens).float().mean())}
        if self.neutral:
            nll = line_nll(self.model, self.tokenizer, self.neutral, **self.kw)
            out["nll"] = sum(nll) / len(nll)
        return out

    def damage(self, sets: dict[str, torch.Tensor], bits: int,
               V: dict[str, torch.Tensor] | None = None) -> dict:
        """Loss of mean answer log-prob / accuracy (and neutral NLL increase) under quantization.

        With ``V`` (layer -> E[h^2] per input channel) the diagonal dose
        ``sum_ij E_ij^2 V_j`` of the touched columns is reported as ``dose``.
        """
        with QuantizeColumns(self.model, sets, bits, V) as qc:
            q = self._stats()
        out = {"dlogp": self.base["logp"] - q["logp"], "dacc": self.base["acc"] - q["acc"]}
        if "nll" in q:
            out["dnll"] = q["nll"] - self.base["nll"]
        if V is not None:
            out["dose"] = qc.dose
        return out


@torch.no_grad()
def dose_of(model: torch.nn.Module, units: list[Unit], bits: int, V: dict[str, torch.Tensor]) -> float:
    """Diagonal dose of quantizing the columns of ``units`` (no forward pass)."""
    with QuantizeColumns(model, unit_sets(units), bits, V) as qc:
        return qc.dose


def dose_matched_random(model: torch.nn.Module, all_units: list[Unit], circuit: list[Unit],
                        V: dict[str, torch.Tensor], bits: int, rng: random.Random,
                        n_keep: int = 10, n_candidates: int = 200) -> list[list[Unit]]:
    """Class-matched random sets whose dose is closest to the circuit's (out of many draws)."""
    target = dose_of(model, circuit, bits, V)
    cands = [match_random(all_units, circuit, rng) for _ in range(n_candidates)]
    doses = [dose_of(model, c, bits, V) for c in cands]
    order = sorted(range(len(cands)), key=lambda i: abs(doses[i] - target))
    return [cands[i] for i in order[:n_keep]]


def additivity(probe: KnowledgeProbe, units: list[Unit], bits: int) -> dict:
    """Joint damage of a unit set against the sum of its members' separate damages."""
    joint = probe.damage(unit_sets(units), bits)["dlogp"]
    singles = [probe.damage(unit_sets([u]), bits)["dlogp"] for u in units]
    return {"joint": joint, "sum_single": sum(singles), "n": len(units)}
