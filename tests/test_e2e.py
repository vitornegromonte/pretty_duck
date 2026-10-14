"""End-to-end pipeline on a tiny model through the real CLI.

No network: the model and tokenizer are saved to disk by the test first.
Runs every subcommand exactly the way a user would.
"""

import json
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from compress.cli import main
from tests.conftest import VOCAB_SIZE


@pytest.fixture()
def model_dir(tmp_path, tokenizer):
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        pad_token_id=VOCAB_SIZE - 1,
    )
    model = LlamaForCausalLM(config).eval()
    path = tmp_path / "tiny_llama"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def _run(capsys, *argv):
    main([str(a) for a in argv])
    return json.loads(capsys.readouterr().out)


def test_full_pipeline_via_cli(capsys, model_dir, calib_file, mc_file, tmp_path):
    stats = tmp_path / "stats.pt"
    errors = tmp_path / "errors.json"
    plan = tmp_path / "plan.json"
    compressed = tmp_path / "compressed"
    report = tmp_path / "report.json"

    out = _run(capsys, "collect_stats", "--model-path", model_dir,
               "--calib", f"T={calib_file}", "--calib", f"S={calib_file}",
               "--out", stats, "--n-samples", "8", "--batch-size", "3")
    assert out["layers"] > 0 and stats.exists()

    out = _run(capsys, "estimate_quant_error", "--model-path", model_dir,
               "--stats", stats, "--out", errors)
    assert set(json.loads(errors.read_text())["layers"])  # non-empty

    # Generous size budget: compress to ~60% of fp16 payload.
    est = json.loads(errors.read_text())["layers"]
    fp16_bytes = sum(e["nbytes"]["fp16"] for e in est.values())
    out = _run(capsys, "allocate_bits", "--errors", errors,
               "--target", "T", "--secondary", "S",
               "--size-budget", str(int(fp16_bytes * 0.6)),
               "--out", plan)
    assert out["bytes_final"] <= fp16_bytes * 0.6
    assert plan.exists()

    out = _run(capsys, "quantize_model", "--model-path", model_dir,
               "--plan", plan, "--out-dir", compressed)
    assert out["compressed_tensor_bytes"] > 0
    assert (compressed / "manifest.json").exists()

    out = _run(capsys, "evaluate_compression", "--model-path", model_dir,
               "--compressed-dir", compressed,
               "--test", f"T={mc_file}", "--out", report)
    result = out["results"]["T"]
    assert 0.0 <= result["accuracy"] <= 100.0
    assert result["n"] == 3
    assert report.exists()
    # Payload level: packed weights really are smaller than fp16 weights.
    # (On a tiny model the per-file torch.save overhead can outweigh the
    # payload, so the measured on-disk ratio may be < 1 here; on real models
    # the payload dominates.  Both numbers are reported, neither is faked.)
    manifest = json.loads((compressed / "manifest.json").read_text())
    assert manifest["totals"]["payload_nbytes"] < manifest["totals"]["fp16_reference_nbytes"]
    assert out["compression_ratio_vs_fp16_linears"] > 0.0
    assert out["compressed_dir_bytes"] >= out["compressed_tensor_bytes"]

    # Baseline (no compression) runs through the same evaluator.
    out = _run(capsys, "evaluate_compression", "--model-path", model_dir,
               "--test", f"T={mc_file}", "--out", str(tmp_path / "base.json"))
    assert out["compressed_dir"] is None


def test_cli_budget_errors_are_reported(capsys, model_dir, calib_file, tmp_path):
    stats = tmp_path / "stats.pt"
    errors = tmp_path / "errors.json"
    _run(capsys, "collect_stats", "--model-path", model_dir,
         "--calib", f"T={calib_file}", "--out", stats, "--n-samples", "4")
    _run(capsys, "estimate_quant_error", "--model-path", model_dir,
         "--stats", stats, "--out", errors)
    with pytest.raises(SystemExit):  # fp16-only: size budget unreachable
        _run(capsys, "allocate_bits", "--errors", errors, "--target", "T",
             "--formats", "fp16", "--size-budget", "10", "--out", tmp_path / "p.json")
    assert "error:" in capsys.readouterr().err
