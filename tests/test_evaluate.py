"""The answer-only log-probability scorer: mask alignment and empty answers."""

import math

import pytest

from compress.evaluate import build_prompt, option_logprob


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
