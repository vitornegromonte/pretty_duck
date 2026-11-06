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
# Qwen3.5-Base: use --prompt-style plain; "auto" applies the chat template's
# empty <think> block, which this base model does not follow.
python -m compress evaluate_compression \
    --model-path Qwen/Qwen3.5-0.8B --prompt-style plain \
    --mmlu law=professional_law:test --out mmlu_report.json

# Domain sub-network probe: zero weights by the math-vs-law activation contrast,
# then score both domains under the same mask.
python -m compress evaluate_compression \
    --model-path Qwen/Qwen3.5-0.8B --mask-stats stats_math_law.pt \
    --mask-kind contrast --mask-seek math --mask-reject law --mask-sparsity 0.3 \
    --prompt-style plain \
    --mmlu law=professional_law:test --mmlu math=college_mathematics:test \
    --out masked.json
```

`--mask-kind` is one of `contrast` (signed `|W|*relu(z)`), `contrast_abs`
(`|W|*|z|`), `inverse` (`|W|*relu(-z)`, the anti-domain control), `magnitude`,
`wanda` (domain-agnostic activation norm), `random`, or `none`.
`contrast`/`inverse`/`contrast_abs` also need `--mask-seek` and `--mask-reject`
from the stats file.  `--mask-only-mlp` restricts masking to the MLP projections
(Persona's `prune_only_mlp`), leaving attention and the head intact.
`--mask-z-clip` bounds `|z|` so a few near-zero-variance channels cannot
dominate the per-row ranking.  The mask is applied after any `--compressed-dir`
weights are loaded, per output row, in place; the report's `mask` block records
what was actually zeroed.

`--target` accepts any domain present in the stats file: the pipeline is
domain-agnostic.  `--signal contrastive` prioritizes blocks using
`C_g = I_T - lam * mean_S I_S` (a hypothesis, validated only by the
evaluation step, never assumed).

One end-to-end run on a real trained model:

```bash
python experiments/run_smollm.py   # SmolLM-135M, wealth target, honesty secondary
```

### Finer allocation units and the target-anchored signal

```bash
# split every Linear into units of 64 consecutive output rows ("layer@r0:r1")
python -m compress estimate_quant_error \
    --model-path HuggingFaceTB/SmolLM-135M --stats stats.pt \
    --formats fp16,int8,int4,nf4 --row-group 64 --out errors_rg64.json

# target-anchored relative contrast: S_T = R_T - median_{d != T} R_d
python -m compress allocate_bits --errors errors_rg64.json --target math \
    --signal tarc --lam 10 --size-budget 117000000 --out plan.json
```

Every format quantizes output rows independently, so a row group stored on its
own is bit-identical to the same rows in the full matrix and its byte cost is
exact. `quantize_model` regroups units into one file per layer (mixed formats
per layer are fine) and refuses plans whose groups do not tile a layer.
`--signal tarc` ranks each domain's importance across units, subtracts the
median rank of the other domains (all non-target domains in the errors file
unless `--secondary` is given) and multiplies the cost by
`1 + lam * max(S_T, 0)`; here `--lam` is only the penalty strength. The greedy
re-scans all candidates at every step: about 24 s for 2.4k units on SmolLM-135M,
so per-row units (about 30k) would need a priority queue.

### Causal probes

Statistics say a channel is more *active* on a domain, not that the model
*needs* it. These probes intervene on Linear inputs (forward-only, no
gradients, weights never modified).

```bash
# held-out evaluation text (MMLU test) and calibration text (MMLU val)
python scripts/make_mmlu_calib.py --split test --out-dir outputs/eval \
    math=college_mathematics,elementary_mathematics law=professional_law
python scripts/make_mmlu_calib.py --out-dir outputs/calib \
    math=college_mathematics,elementary_mathematics law=professional_law

# mean ablation of candidate channel sets vs size- and dose-matched controls
python -m compress causal_probe --model-path HuggingFaceTB/SmolLM-135M \
    --stats stats.pt --target math --others law \
    --eval-target math=outputs/eval/math.txt --eval-other law=outputs/eval/law.txt \
    --k-frac 0.01 --k-frac 0.02 --n-random 5 --out probe.json
```

Channel sets: `tarc`, `zdiff` (standardized mean difference), `importance`
(top `A_T`), `anti`, `random` (matched on `V` rank), `energy_random` (matched on
the perturbation energy the patch inflicts). Selectivity is the NLL increase on
the target minus the mean increase on the other domains, with a bootstrap CI.

Persona experiments (minimal-pair patching). The scripts need `PYTHONPATH=.`:

```bash
python scripts/make_persona_calib.py Persona/data/wealth_train.csv \
    --out-dir outputs/persona_calib        # seek.txt / reject.txt, parsed with csv
python -m compress collect_stats --model-path Qwen/Qwen3.5-0.8B \
    --calib seek=outputs/persona_calib/seek.txt \
    --calib reject=outputs/persona_calib/reject.txt --out stats_persona.pt

T=Persona/data/wealth_test_infer.csv
# does the persona prompt move the margin? includes shuffled/irrelevant controls
python scripts/persona_gate.py --model-path Qwen/Qwen3.5-0.8B --test $T --out gate.json
# sufficiency; add --reverse for necessity (patch the other prompt's activations in)
python scripts/persona_patch.py --model-path Qwen/Qwen3.5-0.8B \
    --stats stats_persona.pt --test $T --k-frac 0.05 --per-question --out patch.json
# quantize only the selected weight columns; how much of the shift survives?
python scripts/persona_quant.py --model-path Qwen/Qwen3.5-0.8B \
    --stats stats_persona.pt --test $T --out quant.json
```

`persona_patch.py` runs three prompt pairs on the same questions (real
seek/reject, word-shuffled, unrelated). Patching every Linear input over the
aligned suffix reproduces the clean run exactly (restoration 1.000), which the
tests assert and every run reports. `persona_quant.py` uses bit widths below 4
only as stronger interventions; nothing is stored in those formats.

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

## Findings so far (measured; see `docs/method.pdf`)

All of this is one model per experiment (SmolLM-135M for statistics, plans and
mean ablation; Qwen3.5-0.8B for the persona experiments), one trait, one seed.

* **Block and row-group signals are too weak to change a plan.** Math-vs-law
  error and importance rank correlations over blocks are 0.995-0.996; target-anchored
  `S_T` is reproducible across calibration halves (split-half 0.91-0.93) but its
  spread is small (std 0.02-0.03), and error-only plans for a math vs a law
  target differ in 2/210 blocks (22/2430 row groups).
* **Per input channel the signal is wider** (std 0.20, 20% of channels with
  |S_T| > 0.25, split-half 0.84).
* **Mean-ablation selectivity is explained by dose.** A channel set chosen by
  contrast looks selective (damage ratio 1.6-2.5x), but sets matched on the
  perturbation energy reproduce it (gap within +-0.012 nats, inside the control spread).
* **Persona prompts move the answer margin, mostly through word overlap.**
  The seek-minus-reject margin shift on the 200 wealth questions is +2.82 nats
  for Qwen3.5-0.8B; shuffled-word prompts give +1.95 and unrelated prompts +0.91.
  SmolLM-135M responds only to word overlap (shuffled +0.97 vs real +0.83).
  The answer *choices* stay near chance under every prompt.
* **A standardized-mean-difference selector (`zdiff`) finds channels that are
  sufficient and necessary for the shift** (5% of channels: restores 0.84 vs 0.18
  for random, destroys 0.80 vs 0.15), at random level on the unrelated pair, and
  with a paired real-minus-shuffled gain of +1.11 [0.85, 1.37] nats. The rank
  contrast `tarc` also restores the unrelated pair, so it is not persona-specific.
* **That does not become a quantization signal.** Quantizing the selected columns
  at 3-4 bits leaves the persona shift intact, like random columns of the same
  dose; only 2-bit shows a hint, inside the noise.

Not shown: factual-knowledge sub-networks, other traits (the honesty split has no
defined prompts), other models or seeds. Patching covers the question and answer
tokens, so it shows where an effect is expressed, not where it is stored.

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
* The Persona CSVs contain multi-line fields; read them with `csv` (as
  `scripts/make_persona_calib.py` does), not line by line, or headers and split
  rows end up in the calibration text.
* The minimal-pair and column-quantization probes score the wealth persona only;
  `h_test_infer.csv` has no defined honesty prompts and its `matching` label appears
  to mark the dishonest answer, so it is not used.

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
