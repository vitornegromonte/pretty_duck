# Domain-aware weight compression (built on Persona)

Implements the pipeline described in `REQ.md` / `TASK.md`: per-domain
activation statistics, diagonal quantization-error estimation, greedy
heterogeneous bit allocation per Linear block, real packed storage, and
multiple-choice evaluation.  The original Persona code lives untouched in
`Persona/` (see `ARCHITECTURE.md` for the audit that maps what it does and
what is reused).  Design and honesty rules are in the module docstrings of
`compress/`.

## Install

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
```

`bitsandbytes` is optional (only the `bnb_nf4` format needs it; its
functional API runs on CPU, its `nn.Module` layers need CUDA).

## Pipeline

```bash
python -m compress collect_stats \
    --model-path HuggingFaceTB/SmolLM-135M \
    --calib wealth=Persona/data/wealth_train.csv \
    --calib honesty=Persona/data/h_train.csv \
    --out stats.pt --n-samples 128 --max-len 256 --seed 42

python -m compress estimate_quant_error \
    --model-path HuggingFaceTB/SmolLM-135M --stats stats.pt \
    --formats fp16,int8,int4,nf4 --out errors.json

python -m compress allocate_bits \
    --errors errors.json --target wealth --secondary honesty \
    --formats fp16,int8,int4,nf4 --size-budget 70000000 \
    --signal contrastive --lam 0.5 --out plan.json
# or: --error-budget <float> instead of --size-budget (exactly one is required)

python -m compress quantize_model \
    --model-path HuggingFaceTB/SmolLM-135M --plan plan.json --out-dir compressed/

python -m compress evaluate_compression \
    --model-path HuggingFaceTB/SmolLM-135M --compressed-dir compressed/ \
    --test wealth=Persona/data/wealth_test_infer.csv \
    --test honesty=Persona/data/h_test_infer.csv \
    --out report.json

# Four-way MMLU, scored by answer letter (NLL per token + per-question predictions).
# First run caches the canonical 160 MB CSV tarball under .cache/mmlu.
python -m compress evaluate_compression \
    --model-path Qwen/Qwen3.5-0.8B --mmlu law=professional_law:test \
    --out mmlu_report.json
```

`--target` accepts any domain present in the stats file: the pipeline is
domain-agnostic.  `--signal contrastive` prioritizes blocks using
`C_g = I_T - lam * mean_S I_S` (a hypothesis, validated only by the
evaluation step, never assumed).

One end-to-end run on a real trained model:

```bash
python experiments/run_smollm.py   # SmolLM-135M, wealth target, honesty secondary
```

## Formats

| name | bits/weight | granularity | storage |
|---|---|---|---|
| `fp16` | 16 | - | half tensor (baseline) |
| `int8` | 8 | per output row | int8 codes + fp16 scales |
| `int4` | 4 | per output row | packed nibbles + fp16 scales |
| `nf4` | 4 | per 64-weight row block | packed nibbles (NF4 codebook) + fp16 absmax |
| `bnb_nf4` | 4 | bitsandbytes blocksize 64 | bitsandbytes' own packed codes + QuantState |

## What is measured vs estimated (read before quoting numbers)

* **Measured**: file sizes on disk (`compressed_tensor_bytes`,
  `compressed_dir_bytes`), evaluation accuracy, per-question latency,
  wall-clock times.
* **Estimated (local, diagonal)**: `D_{g,d}(b) = sum_ij E_ij^2 V_j` — ignores
  cross-channel correlation; a prioritization signal, not a predicted final
  loss.
* **Simulated**: quantization/dequantization happens in float; inference runs
  on dequantized weights.  Storage really is 4/8-bit packed, but there are no
  low-bit kernels here — no inference speedup is claimed.
* Nominal bits-per-parameter is never used as a size; the allocator uses
  exact payload formulas (asserted against packed tensors in the tests), the
  reports use measured bytes.

## Limitations

* `estimate_quant_error` and `quantize_model` stream weights straight from
  the checkpoint's `model.safetensors` (one matrix in memory at a time) when
  a single-file checkpoint is present, and fall back to a full model load
  otherwise. Multi-file sharded checkpoints use the fallback.
* `evaluate_compression` still loads the whole model: fp16 on CUDA, **bf16
  on CPU** (half the RAM of fp32; recorded as `dtype` in every report, and
  identical across compared configurations).
* Only `nn.Linear` weights are compressed; embeddings/norms reload from the
  base model at their original precision.  A `lm_head` tied to the embeddings
  is skipped: rewriting it would rewrite the embeddings too.
* Multimodal checkpoints (e.g. Qwen3.5-Base) are evaluated on their text
  tower only; the vision tower is neither collected nor compressed.
* Prompts follow the tokenizer's chat template by default (`--prompt-style
  auto`).  Use `plain` for base checkpoints that were never chat-tuned, or
  `llama2` to reproduce the original Persona protocol.
* The allocator accepts portable formats only (`fp16/int8/int4/nf4`);
  `bnb_nf4` can be used for uniform experiments via `estimate_quant_error`
  and direct quantization, but has no exact byte formula in the allocator.
* Local development/tests run on CPU with tiny models; real-model runs need
  a machine with enough RAM (the SmolLM-135M experiment needs ~2 GB free).
* Persona's sparse-mask method (weight zeroing) is a different mechanism from
  quantization and is preserved untouched in `Persona/`.

## Run on Colab

Use a GPU runtime (T4 is plenty). In three cells:

```python
# cell 1
!git clone -b feature/domain-compression https://github.com/vitornegromonte/pretty_duck.git
%cd pretty_duck
```

```python
# cell 2 — full pipeline (SmolLM-360M default; ~10 min on T4)
!bash script/run_all.sh
# or: !bash script/run_all.sh HuggingFaceTB/SmolLM-135M
```

```python
# cell 3 — results table
import json, pandas as pd
r = json.load(open("outputs/smollm-360m/report.json"))
rows = [{"config": k, **{k2: v for k2, v in c.items() if not isinstance(v, str)}}
        for k, c in r["configs"].items()]
pd.DataFrame(rows).set_index("config")
```

On GPU the CLI automatically evaluates in fp16 (recorded as `dtype` in the
reports); `bnb_nf4` runs through bitsandbytes' CUDA path. Interrupted runs
resume: each stage is skipped when its artifact already exists under
`outputs/<model>/`.

## Tests

```bash
python -m pytest tests/ -q    # offline, CPU, < 3 GB RAM
```
