"""Hooks for activation patching over the last positions of a sequence.

A clean and a corrupt prompt end in the same tokens, so positions are aligned from the end. ``SuffixCapture``
records the input of chosen Linear layers over the last ``suffix`` positions of the clean run. ``SuffixPatch``
overwrites chosen input channels of the corrupt run with those values. The patched values are real activations of
a real prompt, so the size of the perturbation is matched by construction. Patching every Linear input over the
suffix reproduces the clean run exactly, which is the built-in 100% reference. ``ratio_ci`` gives a ratio of sums
over questions with a bootstrap interval. Forward-only. Both hooks are removed on exit, also on errors.
"""

from __future__ import annotations

import re

import torch


class Pair:
    ids_clean: torch.Tensor      # (1, n_clean)
    ids_corrupt: torch.Tensor    # (1, n_corrupt)
    n_answer: int                # answer tokens (identical in both)
    suffix: int                  # common suffix length, >= n_answer + 1


class SuffixCapture:
    """Record the last ``suffix`` positions of every named Linear's input."""

    def __init__(self, model: torch.nn.Module, names: list[str], suffix: int):
        self.model, self.names, self.suffix = model, names, suffix
        self.inputs: dict[str, torch.Tensor] = {}
        self._hooks: list = []

    def __enter__(self) -> "SuffixCapture":
        modules = dict(self.model.named_modules())
        for name in self.names:
            self._hooks.append(modules[name].register_forward_pre_hook(self._make(name)))
        return self

    def _make(self, name: str):
        def hook(module, args):
            self.inputs[name] = args[0][:, -self.suffix:].detach().clone()
        return hook

    def __exit__(self, *exc) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


class SuffixPatch:
    """Overwrite chosen input channels of the named Linears over the last ``suffix`` positions."""

    def __init__(self, model: torch.nn.Module, sets: dict[str, torch.Tensor],
                 clean: dict[str, torch.Tensor], suffix: int):
        self.model, self.sets, self.clean, self.suffix = model, sets, clean, suffix
        self._hooks: list = []

    def __enter__(self) -> "SuffixPatch":
        modules = dict(self.model.named_modules())
        try:
            for name, idx in self.sets.items():
                if idx.numel():
                    self._hooks.append(
                        modules[name].register_forward_pre_hook(self._make(name, idx)))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _make(self, name: str, idx: torch.Tensor):
        def hook(module, args):
            x = args[0].clone()
            src = self.clean[name]
            x[:, -self.suffix:, idx.to(x.device)] = src[:, :, idx.to(src.device)].to(device=x.device, dtype=x.dtype)
            return (x, *args[1:])
        return hook

    def __exit__(self, *exc) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def layer_of(name: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def ratio_ci(num: torch.Tensor, den: torch.Tensor, n_boot: int, seed: int):
    """Ratio of sums over questions with a percentile bootstrap CI."""
    g = torch.Generator().manual_seed(seed)
    point = float(num.sum() / den.sum())
    n = len(num)
    boots = torch.empty(n_boot, dtype=torch.float64)
    for b in range(n_boot):
        i = torch.randint(n, (n,), generator=g)
        boots[b] = num[i].sum() / den[i].sum()
    lo, hi = torch.quantile(boots, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return point, lo, hi


def pair_patch_probe(
    model: torch.nn.Module,
    tokenizer,
    rows: list[dict],
    *,
    clean_sys: str,
    corrupt_sys: str,
    conditions: dict[str, dict[str, torch.Tensor]],
    style: str = "auto",
    n_boot: int = 1000,
    seed: int = 0,
    keep_per_question: bool = False,
    log=lambda msg: None,
) -> dict:
    """Restoration of the clean-minus-corrupt margin for every named channel-set condition.

    ``rows`` carry ``question``, ``A``, ``B`` and ``matching`` ("A"/"B").
    Questions where either option fails to tokenize as a clean continuation
    are skipped and counted.  ``keep_per_question`` adds the per-question
    margins (``question_index`` says which rows survived), which allows paired
    comparisons across prompt pairs scored on the same questions.  Swapping
    ``clean_sys`` and ``corrupt_sys`` measures *necessity*: the fraction of the
    shift destroyed by patching the other prompt's activations in.
    """
    model.eval()
    device = next(model.parameters()).device
    names = sorted({n for sets in conditions.values() for n in sets})
    clean_m, corrupt_m, kept = [], [], []
    patched_m = {c: [] for c in conditions}
    skipped = 0
    for qi, row in enumerate(rows):
        opts = (row["A"], row["B"]) if row["matching"] == "A" else (row["B"], row["A"])
        pairs = [encode_pair(tokenizer, row["question"], o, clean_sys, corrupt_sys, style) for o in opts]
        if any(p is None for p in pairs):
            skipped += 1
            continue
        lp = {"clean": [], "corrupt": [], **{c: [] for c in conditions}}
        for pair in pairs:
            with SuffixCapture(model, names, pair.suffix) as cap:
                lp["clean"].append(answer_logprob(
                    model(input_ids=pair.ids_clean.to(device)).logits, pair.ids_clean, pair.n_answer))
            ids = pair.ids_corrupt.to(device)
            lp["corrupt"].append(answer_logprob(model(input_ids=ids).logits, ids, pair.n_answer))
            for cond, sets in conditions.items():
                with SuffixPatch(model, sets, cap.inputs, pair.suffix):
                    lp[cond].append(answer_logprob(model(input_ids=ids).logits, ids, pair.n_answer))
        margin = lambda key: lp[key][0] - lp[key][1]
        clean_m.append(margin("clean")); corrupt_m.append(margin("corrupt")); kept.append(qi)
        for cond in conditions:
            patched_m[cond].append(margin(cond))
        if (qi + 1) % 50 == 0:
            log(f"{qi + 1}/{len(rows)} questions")
    clean_t = torch.tensor(clean_m, dtype=torch.float64)
    corrupt_t = torch.tensor(corrupt_m, dtype=torch.float64)
    den = clean_t - corrupt_t
    report = {
        "n_questions": len(clean_m), "n_skipped": skipped,
        "mean_margin": {"clean": float(clean_t.mean()), "corrupt": float(corrupt_t.mean())},
        "conditions": {},
    }
    if keep_per_question:
        report["per_question"] = {"question_index": kept, "clean": clean_m, "corrupt": corrupt_m,
                                  "patched": patched_m}
    for cond in conditions:
        pt = torch.tensor(patched_m[cond], dtype=torch.float64)
        point, lo, hi = _ratio_ci(pt - corrupt_t, den, n_boot, seed)
        report["conditions"][cond] = {
            "restoration": point, "ci95": [lo, hi], "mean_margin": float(pt.mean()),
            "channels": sum(int(i.numel()) for i in conditions[cond].values()),
        }
    return report
