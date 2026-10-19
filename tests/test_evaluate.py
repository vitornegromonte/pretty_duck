"""The answer-only log-probability scorer: mask alignment and empty answers."""

import math

import pytest

from compress.evaluate import (
    build_prompt,
    evaluate_mmlu_rows,
    format_mmlu_question,
    option_logprob,
    option_nll,
)


def _n_tokens(tokenizer, text: str) -> int:
    return tokenizer(text, return_tensors="pt")["input_ids"].shape[1]


def test_answer_mask_starts_exactly_after_the_prompt(tiny_model, tokenizer):
    """The full string must begin with the prompt's own tokens, or the mask
    would hide part of the answer (or score part of the prompt)."""
    prompt = build_prompt("", "w1 w2?\n\nAnswer:")
    n_prompt = _n_tokens(tokenizer, prompt)
    for option in ("w3", "w3 w4", "w5 w1 w2"):
        full = tokenizer(prompt + " " + option, return_tensors="pt")["input_ids"]
        assert full.shape[1] > n_prompt
        assert full[0, :n_prompt].tolist() == tokenizer(
            prompt, return_tensors="pt"
        )["input_ids"][0].tolist()


def test_prompt_uses_tokenizer_chat_template():
    class ChatTokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False and add_generation_prompt is True
            assert messages == [{"role": "user", "content": "w1?"}]
            return "CHAT"

    assert build_prompt("", "w1?", ChatTokenizer()) == "CHAT"


def test_prompt_style_overrides_template():
    class ChatTokenizer:
        chat_template = "template"

        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("plain style must not use the template")

    assert build_prompt("", "w1?", ChatTokenizer(), "plain") == "w1?"
    assert build_prompt("", "w1?", style="llama2") == "[INST] w1? [/INST]"


def test_chat_style_requires_a_template():
    with pytest.raises(ValueError, match="chat template"):
        build_prompt("", "w1?", tokenizer=None, style="chat")


def test_option_logprob_is_finite(tiny_model, tokenizer):
    value = option_logprob(tiny_model, tokenizer, "", "w1 w2?", "w3 w4", "cpu")
    assert math.isfinite(value)


def test_option_logprob_rejects_a_missing_answer(tiny_model, tokenizer):
    prompt = build_prompt("", "w1 w2?\n\nAnswer:")
    if _n_tokens(tokenizer, prompt + " ") > _n_tokens(tokenizer, prompt):
        pytest.skip("this tokenizer appends a token for a trailing space")
    with pytest.raises(ValueError, match="no answer tokens"):
        option_logprob(tiny_model, tokenizer, "", "w1 w2?", "", "cpu")


def test_option_nll_returns_answer_token_count(tiny_model, tokenizer):
    nll, n_tokens = option_nll(tiny_model, tokenizer, "", "w1 w2?", "w3 w4", "cpu")
    assert math.isfinite(nll) and n_tokens > 0
    assert option_logprob(tiny_model, tokenizer, "", "w1 w2?", "w3 w4", "cpu") == -nll


def test_mmlu_question_lists_four_lettered_choices():
    row = {"question": "w1 w2?", "choices": ["w3", "w4", "w5", "w1"], "answer": 0}
    assert format_mmlu_question(row) == "w1 w2?\nA. w3\nB. w4\nC. w5\nD. w1"


def test_evaluate_mmlu_rows_reports_accuracy_and_nll(tiny_model, tokenizer):
    rows = [
        {"question": "w1 w2?", "choices": ["w3", "w4", "w5", "w1"], "answer": 0},
        {"question": "w2 w3?", "choices": ["w1", "w2", "w4", "w5"], "answer": 3},
    ]
    result = evaluate_mmlu_rows(tiny_model, tokenizer, rows)
    assert result["n"] == 2
    assert 0.0 <= result["accuracy"] <= 100.0
    assert math.isfinite(result["nll_per_token"])
    assert len(result["predictions"]) == 2
    assert result["predictions"][0]["expected"] == "A"
    assert set(result["predictions"][0]["option_nll"]) == {"A", "B", "C", "D"}
