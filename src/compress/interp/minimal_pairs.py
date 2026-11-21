"""Entity-swap minimal pairs on templated factual recall.

The clean prompt asks about subject ``s_c`` (answer first token ``a_c``) and the corrupt one about ``s_k``
(``a_k``). Both have the same token length. The metric is the logit difference ``LD = logit[a_c] - logit[a_k]``
at the last position. Only facts that the model answers correctly are kept, so patching measures knowledge
the model has.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


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
