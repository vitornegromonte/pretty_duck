"""Minimal-pair activation patching on the persona answer margin.

A *pair* is one question and answer option scored under two system prompts:
``clean`` (e.g. the seek persona) and ``corrupt`` (e.g. the reject persona).
The question and answer tokens are identical, so the two token sequences share
a common suffix and positions can be aligned from the end.  We run the clean
sequence, record every Linear input over that suffix, then re-run the corrupt
sequence while overwriting chosen input channels with the clean values.

    margin(run) = logp(persona-matching option) - logp(other option)

and the *restoration* of an intervention is

    sum_q [margin_patched - margin_corrupt] / sum_q [margin_clean - margin_corrupt].

Unlike mean ablation, the patched values are real activations of the same
prompt, so the perturbation is matched in size by construction.  Patching
*every* Linear input over the suffix reproduces the clean run exactly (all
residual contributions at those positions then come from clean inputs), which
is the built-in 100% reference.  Forward-only, batch size 1, no padding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from .evaluate import build_prompt


@dataclass
class Pair:
    ids_clean: torch.Tensor      # (1, n_clean)
    ids_corrupt: torch.Tensor    # (1, n_corrupt)
    n_answer: int                # answer tokens (identical in both)
    suffix: int                  # common suffix length, >= n_answer + 1


def common_suffix(a: list[int], b: list[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return n


def _encode(tokenizer, system_prompt: str, question: str, option: str, style: str):
    prompt = build_prompt(system_prompt, f"{question}\n\nAnswer:", tokenizer, style)
    prompt_ids = tokenizer(prompt)["input_ids"]
    full_ids = tokenizer(prompt + " " + option)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
        return None  # the answer does not tokenize as a clean continuation
    return full_ids, len(full_ids) - len(prompt_ids)


def encode_pair(tokenizer, question: str, option: str, clean_sys: str, corrupt_sys: str,
                style: str = "auto") -> Pair | None:
    """Token sequences for one (question, option) under both prompts, or ``None`` to skip."""
    c = _encode(tokenizer, clean_sys, question, option, style)
    k = _encode(tokenizer, corrupt_sys, question, option, style)
    if c is None or k is None or c[1] != k[1]:
        return None
    suffix = common_suffix(c[0], k[0])
    if suffix < c[1] + 1:
        return None
    return Pair(torch.tensor([c[0]]), torch.tensor([k[0]]), c[1], suffix)


def answer_logprob(logits: torch.Tensor, ids: torch.Tensor, n_answer: int) -> float:
    """Summed log-probability of the last ``n_answer`` tokens of ``ids`` (batch size 1)."""
    pred = logits[0, -n_answer - 1:-1].float().log_softmax(-1)
    return float(pred.gather(-1, ids[0, -n_answer:].unsqueeze(-1).to(pred.device)).sum())


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
            x[:, -self.suffix:, idx.to(x.device)] = src[:, :, idx.to(src.device)].to(x.dtype)
            return (x, *args[1:])
        return hook

    def __exit__(self, *exc) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def layer_of(name: str) -> int | None:
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _ratio_ci(num: torch.Tensor, den: torch.Tensor, n_boot: int, seed: int):
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


@torch.no_grad()
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
