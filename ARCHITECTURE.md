# Architecture audit — original Persona code

Written before any structural changes, per REQ §4. Files audited: `Persona/mask.py`,
`Persona/contra.py`, `Persona/README.md`, `Persona/environment.yaml`, `Persona/data/`.

## Entry points

Both scripts are standalone CLIs with argparse subcommands (no package, no tests):

| Script | Subcommand | What it does |
|---|---|---|
| `mask.py` | `phase2_wanda` | Wanda masks: `S = |W| · mean‖h‖₂` per Linear, per-row top-k keep, saves `M_seek_*/M_reject_*.pt` boolean masks |
| `mask.py` | `phase2_sparsegpt` | Named "SparseGPT" but calibration data is loaded and never used — pure per-column magnitude pruning in 128-column blocks |
| `mask.py` | `phase3_mc` | Loads masks, zeroes masked weights in place, evaluates A/B multiple choice by comparing option losses |
| `mask.py` | `phase3_gen` | Loads one mask set, generates a continuation |
| `contra.py` | `phase2_wanda_contrast` | Per-layer mean/var of per-sample input L2 norms for `seek`/`reject`; contrastive score `z=(μ_s−μ_r)/√(σ_s²+σ_r²)`, then same top-k masking |
| `contra.py` | `phase2_sparse_contrast` | Incremental per-channel `sum/sumsq/count` of inputs (proper running stats); score `|μ_s−μ_r|/√(σ_s²+σ_r²)` |
| `contra.py` | `phase3_mc` | Same idea as mask.py but scores options by answer-only log-probability (prompt tokens masked out of labels); flips correct label for the `reject` persona |
| `contra.py` | `phase3_gen` | Generation with anti-repetition defaults and an empty-response retry |

## How activations, importance and masks flow

1. **Activation capture** — `register_forward_hook` on every `nn.Linear`; the hook takes
   `input[0]`, i.e. the pre-activation hidden states of shape `(batch, seq, in_features)`.
   - `mask.py::get_activations`: L2 norm over dims `(0,1)` → one vector per line, averaged over lines.
   - `contra.py::_collect_act_stats`: same norm, then mean/var across lines (variance of norms, not of activations).
   - `contra.py::_collect_input_running_stats`: **the only correct per-channel accumulator** —
     `sum`, `sumsq`, `count` over `(batch·seq, in_features)` → per-channel μ and σ².
2. **Importance / score** — `|W_ij| · score_j` (Wanda-style: weight magnitude × activation
   statistic of input channel `j`). Contrastive variants replace `score` with a
   seek-vs-reject difference (z-score or absolute difference over pooled std).
3. **Masks** — boolean `(out_features, in_features)` tensors, per-row top-k by score.
   Saved with `torch.save`; applied later as `module.weight.data *= mask`
   (**in-place mutation of the loaded model**).
4. **Evaluation** — multiple choice: compare `logprob(answer_A)` vs `logprob(answer_B)`
   under a persona system prompt; generation: standard `model.generate`.

## Architecture / layers supported

- Decoder-only LLaMA-style models (`LlamaForCausalLM` in mask.py, `AutoModelForCausalLM` in contra.py).
- Only `torch.nn.Linear` layers are hooked → q/k/v/o_proj, gate/up/down_proj, `lm_head`.
  `embed_tokens` (an `nn.Embedding`) is never touched.
- `make_layer_skipper` (contra.py) can exclude attention layers, `lm_head`, `embed_tokens`
  by name keywords — a pruning-safety feature, not needed for quantization.

## Model / tokenizer / data loading

- Model: `from_pretrained(path, torch_dtype=fp16|bf16, device_map='auto')`, `.eval()`.
- Tokenizer: `AutoTokenizer`, `pad_token = eos_token` fallback.
- Calibration: plain text files, one line per sample; `get_calibration_data` shuffles
  (seeded via `set_seed` in contra.py only) and pads to `max_length=512`.
- Data (`Persona/data/`): two domain pairs, `matching/not_matching` (calib + raw test)
  and `*_test_infer.csv` with `question,A,B,matching` (multiple-choice eval):
  - honesty pair: `h_train.csv` / `h_test.csv` / `h_test_infer.csv`
  - wealth pair: `wealth_train.csv` / `wealth_test.csv` / `wealth_test_infer.csv`
  - `train.csv` / `test(1).csv` / `test_mc.csv`: a third, MBTI-ish set
    (`compress/` only ever reads the `*_train.csv` and `*_test_infer.csv`
    files; the rest are kept as the original Persona data)

## What the original actually applies to the model

- **Sparse masks only**: weights multiplied by a boolean mask → zeroed weights.
  No quantization, no format conversion, no packed storage. Zeroed weights do not
  reduce model size (explicitly called out as a constraint in TASK.md §2).

## Limitations relevant to the compression work

1. **No quantization backend anywhere** — no int8/int4/NF4 code exists; only
   `bitsandbytes` in `environment.yaml` (CUDA-only; its `Linear4bit` forward
   asserts on CPU with bitsandbytes 0.50.2 — verified locally).
2. **Padding counted as content** — calibration is padded to `max_length`, but hooks
   accumulate over all positions; `attention_mask` is ignored. REQ §5 forbids this.
3. **No batching** — one forward pass per text line (slow; stats collected per-line then stacked).
4. **Stats are the wrong statistic for our formulas** — norms of whole samples, or
   mean/var of those norms; we need per-channel `E|h_j|` and `E[h_j²]` over valid tokens.
5. **Masks mutate weights in place** — evaluation reloads the model each run, so the
   original file is intact, but the object must not be reused across experiments.
6. **`phase2_sparsegpt` ignores its calibration argument** — do not reuse it as a reference.
7. **Environment is CUDA-pinned** — `environment.yaml` targets python 3.8 + cu118;
   this machine has no GPU and 3.7 GB RAM, so all local work uses a tiny model on CPU.
8. **No tests, no packaging** — nothing to run for regression checking; new code ships with pytest coverage.

## What is reused for the compression pipeline

- Hook pattern (`register_forward_hook` on `nn.Linear`, read `input[0]`) — kept, with
  `attention_mask` awareness and batched accumulation added.
- Incremental `sum/sumsq/count` accumulator from `contra.py::_collect_input_running_stats`
  — extended to also accumulate `sum|h|` (for `A = E|h|`).
- MC evaluation protocol from `contra.py::phase3_multiple_choice` — answer-only
  log-probability scoring, label flip for the anti-domain persona, per-question latency.
- argparse-subcommand CLI convention and `set_seed` from `contra.py`.
- Calibration CSVs and the `_test_infer.csv` A/B/matching format.

Original `mask.py` / `contra.py` are left byte-for-byte untouched; the new pipeline is
a separate root-level `compress/` package.
