"""Shared fixtures: a tiny offline Llama + word-level tokenizer + CSV data.

Everything here is constructed locally so the suite needs no network and
fits comfortably in a few MB of RAM.
"""

import csv

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

VOCAB_SIZE = 64


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizerFast:
    vocab = {f"w{i}": i for i in range(VOCAB_SIZE - 2)}
    vocab["<unk>"] = VOCAB_SIZE - 2
    vocab["<pad>"] = VOCAB_SIZE - 1
    tok = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<unk>"
    )


@pytest.fixture()
def tiny_model(tokenizer) -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        pad_token_id=VOCAB_SIZE - 1,
    )
    return LlamaForCausalLM(config).eval()


CALIB_LINES = [
    "w1 w2 w3 w4 w5",
    "w2 w2 w3",
    "w5 w4 w3 w2 w1 w1",
    "w3 w3 w3 w3",
    "w1 w5 w1 w5",
    "w2 w4 w2 w4 w2",
    "w1 w2",
    "w5 w3 w1 w4 w2 w3",
]

MC_ROWS = [
    {"question": "w1 w2?", "A": "w3 w4", "B": "w5 w1", "matching": "A"},
    {"question": "w3 w4?", "A": "w2 w5", "B": "w1 w3", "matching": "B"},
    {"question": "w5 w1?", "A": "w4 w2", "B": "w3 w5", "matching": "A"},
]


@pytest.fixture()
def calib_file(tmp_path):
    path = tmp_path / "calib.txt"
    path.write_text("\n".join(CALIB_LINES) + "\n")
    return path


@pytest.fixture()
def mc_file(tmp_path):
    path = tmp_path / "test_infer.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "A", "B", "matching"])
        writer.writeheader()
        writer.writerows(MC_ROWS)
    return path
