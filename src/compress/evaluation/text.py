"""Next-token negative log-likelihood on plain text (perplexity checks).
"""

from __future__ import annotations

import torch


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


@torch.no_grad()
def token_nll(model: torch.nn.Module, tokenizer, lines: list[str], *, max_len: int = 256, batch_size: int = 8) -> float:
    """Mean next-token NLL over all valid tokens of ``lines`` (token-weighted, unlike ``line_nll``)."""
    model.eval()
    device = next(model.parameters()).device
    total, count = 0.0, 0
    for i in range(0, len(lines), batch_size):
        batch = tokenizer(lines[i:i + batch_size], return_tensors="pt", padding=True, truncation=True,
                          max_length=max_len).to(device)
        logits = model(**batch).logits[:, :-1].float()
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), batch["input_ids"][:, 1:], reduction="none")
        valid = batch["attention_mask"][:, 1:].bool()
        total += float(nll[valid].sum())
        count += int(valid.sum())
    return total / count
