"""Multiple-choice evaluation, same scoring as Persona/contra.py.

Each question offers options A and B; the model picks the option with the
higher log-probability of the answer tokens (prompt tokens excluded from the
loss, matching ``contra.phase3_multiple_choice``).  ``negate`` flips the
correct label, which is how the anti-domain persona ("reject"/"negate") was
scored in the original code.  The prompt uses the tokenizer's chat template
when it has one, so the evaluator works across model families.
"""

from __future__ import annotations

import csv
import time

import torch

from .mmlu import LETTERS, load_mmlu

PROMPT_STYLES = ("auto", "chat", "plain", "llama2")


def build_prompt(
    system_prompt: str, user_prompt: str, tokenizer=None, style: str = "auto"
) -> str:
    """Wrap the question for the model.

    ``auto`` uses the tokenizer's chat template when it has one and plain
    completion otherwise.  Base checkpoints without SFT are better served by
    ``plain``; ``llama2`` reproduces the original Persona/contra.py wrapper.
    """
    if style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt style {style!r}")
    if style == "llama2":
        if system_prompt:
            return f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"
        return f"[INST] {user_prompt} [/INST]"
    template = getattr(tokenizer, "chat_template", None) if tokenizer is not None else None
    if style == "chat" and not template:
        raise ValueError("prompt style 'chat' needs a tokenizer with a chat template")
    if style in ("auto", "chat") and template:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if system_prompt:
        return f"{system_prompt}\n\n{user_prompt}"
    return user_prompt


def option_nll(
    model, tokenizer, system_prompt: str, question: str, option: str, device,
    prompt_style: str = "auto",
) -> tuple[float, int]:
    """Return ``(total NLL, answer token count)`` for ``option``."""
    prompt = build_prompt(system_prompt, f"{question}\n\nAnswer:", tokenizer, prompt_style)
    enc_prompt = tokenizer(prompt, return_tensors="pt").to(device)
    enc_full = tokenizer(prompt + " " + option, return_tensors="pt").to(device)
    input_ids = enc_full["input_ids"]
    labels = input_ids.clone()
    labels[:, : enc_prompt["input_ids"].shape[1]] = -100  # score answer only
    answer_len = int((labels != -100).sum().item())
    if answer_len == 0:
        raise ValueError(f"option {option!r} produced no answer tokens to score")
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=enc_full["attention_mask"], labels=labels)
    return float(out.loss.item()) * answer_len, answer_len


def option_logprob(
    model, tokenizer, system_prompt: str, question: str, option: str, device,
    prompt_style: str = "auto",
) -> float:
    """Log-probability the model assigns to ``option`` given the question."""
    return -option_nll(
        model, tokenizer, system_prompt, question, option, device, prompt_style
    )[0]


def evaluate_mc(
    model,
    tokenizer,
    test_file: str,
    *,
    negate: bool = False,
    sys_prompt: str = "",
    max_questions: int | None = None,
    prompt_style: str = "auto",
) -> dict:
    """Evaluate one ``*_test_infer.csv`` file; returns a metrics dict.

    Accuracy is measured against the ``matching`` column (or its flip when
    ``negate=True``).  Latency is wall-clock per question (two forward passes).
    """
    model.eval()
    device = next(model.parameters()).device
    correct = total = 0
    total_ms = 0.0
    with open(test_file, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if max_questions is not None and total >= max_questions:
                break
            question, a, b = row["question"].strip(), row["A"].strip(), row["B"].strip()
            matching = row["matching"].strip().upper()
            expected = matching if not negate else ("B" if matching == "A" else "A")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logp_a = option_logprob(
                model, tokenizer, sys_prompt, question, a, device, prompt_style
            )
            logp_b = option_logprob(
                model, tokenizer, sys_prompt, question, b, device, prompt_style
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_ms += (time.perf_counter() - t0) * 1000.0

            correct += int(("A" if logp_a >= logp_b else "B") == expected)
            total += 1

    if total == 0:
        raise ValueError(f"no questions evaluated from {test_file}")
    return {
        "test_file": str(test_file),
        "negate": negate,
        "sys_prompt": sys_prompt,
        "prompt_style": prompt_style,
        "n": total,
        "correct": correct,
        "accuracy": 100.0 * correct / total,
        "avg_latency_ms": total_ms / total,
    }


def format_mmlu_question(row: dict) -> str:
    """Lay out an MMLU row with lettered choices, ready for an ``Answer:`` suffix."""
    lines = [row["question"].strip()]
    lines.extend(f"{letter}. {choice.strip()}" for letter, choice in zip(LETTERS, row["choices"]))
    return "\n".join(lines)


def evaluate_mmlu_rows(
    model,
    tokenizer,
    rows: list[dict],
    *,
    sys_prompt: str = "",
    max_questions: int | None = None,
    prompt_style: str = "auto",
) -> dict:
    """Score four-way MMLU rows by letter; NLL is length-normalized.

    Each row is ``{question, choices, answer}`` with ``answer`` an index into
    ``choices``.  The option text is the letter, so accuracy matches the
    standard MMLU protocol rather than comparing choice phrasing.
    """
    model.eval()
    device = next(model.parameters()).device
    correct = total = 0
    total_ms = 0.0
    nll_sum = 0.0
    predictions = []
    for row in rows:
        if max_questions is not None and total >= max_questions:
            break
        question = format_mmlu_question(row)
        expected = LETTERS[row["answer"]]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        option_nlls = {}
        for letter in LETTERS:
            nll, n_tokens = option_nll(
                model, tokenizer, sys_prompt, question, letter, device, prompt_style
            )
            option_nlls[letter] = nll / n_tokens
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_ms += (time.perf_counter() - t0) * 1000.0

        predicted = min(option_nlls, key=option_nlls.get)
        correct += int(predicted == expected)
        nll_sum += option_nlls[expected]
        predictions.append(
            {
                "expected": expected,
                "predicted": predicted,
                "nll": option_nlls[expected],
                "option_nll": option_nlls,
            }
        )
        total += 1

    if total == 0:
        raise ValueError("no MMLU questions evaluated")
    return {
        "sys_prompt": sys_prompt,
        "prompt_style": prompt_style,
        "n": total,
        "correct": correct,
        "accuracy": 100.0 * correct / total,
        "nll_per_token": nll_sum / total,
        "avg_latency_ms": total_ms / total,
        "predictions": predictions,
    }


def evaluate_mmlu(
    model,
    tokenizer,
    subject: str,
    split: str = "test",
    *,
    sys_prompt: str = "",
    max_questions: int | None = None,
    prompt_style: str = "auto",
    cache_dir=None,
) -> dict:
    """Load one MMLU subject/split and score it.  See ``load_mmlu`` for caching."""
    rows = load_mmlu(subject, split, cache_dir)
    result = evaluate_mmlu_rows(
        model,
        tokenizer,
        rows,
        sys_prompt=sys_prompt,
        max_questions=max_questions,
        prompt_style=prompt_style,
    )
    return {"dataset": "cais/mmlu", "subject": subject, "split": split, **result}
