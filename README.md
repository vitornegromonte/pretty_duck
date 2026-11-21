# pretty_duck

Domain-aware low-bit quantization of language models, with interpretability tests that find which components
(attention heads, MLP neurons) a body of knowledge depends on. The code studies whether protecting those components at
full precision, or other signals, help when the rest of the model is quantized at a low bit width.

> Find a document: start at `docs/index.md`. Rules for contributors and for Claude Code: `CLAUDE.md` and `.claude/rules/`.

## Install

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[dev]"
```

`bitsandbytes` is optional (`pip install -e ".[gpu]"`): only the `bnb_nf4` format needs it.
Without an editable install, run scripts with `PYTHONPATH=src` (the Makefile does this).

## Layout

| Path | Content |
|---|---|
| `src/compress/quant/` | formats and exact sizes, fake quantization, GPTQ, protected columns, activation statistics, per-layer sensitivity |
| `src/compress/interp/` | units, minimal pairs, activation patching, circuit ranking, quantization damage of a unit set |
| `src/compress/evaluation/` | multiple-choice scoring and text perplexity |
| `src/compress/data/` | MMLU questions, templated fact domains (capitals, elements, french), calibration text |
| `src/compress/experiments/` | experiment setup and the suite of quantization conditions |
| `scripts/` | command-line drivers; each one parses arguments and calls the library |
| `tests/` | offline CPU tests, in folders that mirror `src/compress/` |
| `docs/` | paper (`paper/`), decisions and experiment record (`adr/`), advisor review (`reviews/`), guides |

## Run

```bash
make test                                   # offline CPU tests
PYTHONPATH=src python scripts/make_mmlu_calib.py --help      # calibration and evaluation text from MMLU
PYTHONPATH=src python scripts/circuit_probe.py --help        # find circuits by activation patching
PYTHONPATH=src python scripts/entanglement.py --help         # are the circuits of two domains shared?
PYTHONPATH=src python scripts/neuron_scan.py --help          # neuron-level scan
PYTHONPATH=src python scripts/sensitivity.py --help          # measured damage of quantizing one layer at a time
PYTHONPATH=src python scripts/quant_suite.py --help          # quantization conditions with a protected circuit
```

`scripts/quant_suite.py` runs the suites defined in `src/compress/experiments/suite.py` (grid, calibration, protection,
soft weights, mixed bits, low rank, reference). It writes after every condition and resumes where it stopped.
Every result file stores its configuration and a provenance record (commit, software, GPUs, dtype).

## What is measured and what is simulated

All quantization is **simulated** on dequantized weights: no packed model is run and no speedup is claimed. Sizes are counted with
exact formulas (`quant/formats.py`, `quant/protect.py::plan_nbytes`), never with nominal bits. A number is marked as measured,
estimated, or simulated wherever it is quoted.

## Findings so far

Details, numbers, and caveats: `docs/adr/0001-domain-aware-quantization-experiment-record.md` and `docs/paper/method.pdf`.

* **Works.** Minimal-pair patching finds circuits of about 1% of the units that restore, and when removed destroy, most of a
  fact's answer logit difference on held-out facts. At neuron level the MLP part of a circuit is specific to its domain;
  attention heads are shared. A directly measured per-layer sensitivity table is cheap, roughly additive, and the best
  allocation signal found. Clipping the group range is the largest quantizer-side gain (GPTQ at int3: the NLL gap to fp16
  goes from 0.18 to 0.07).
* **Did not hold up.** Protecting a circuit helped under round-to-nearest in one realization, but not under GPTQ, and not
  specifically for the circuit's domain. Letter accuracy of equivalent quantized models varies by about 6 points (the same GPTQ gave
  49.5% and 40.7% on two GPUs), so judge damage with continuous metrics and several realizations.
* **Negative results.** A domain-contrast score does not change a bit allocation; mean-ablation selectivity is explained by
  perturbation dose; the diagonal error `D` is uncorrelated with the measured damage per layer.
* **Limits.** One model family, one rounding realization per reported condition, small fact sets, simulated quantization.

## History

The experiments E1 to E23 were run with an earlier code base that included a diagonal-error bit-allocation planner, packed
storage, persona experiments, and Brazilian-law benchmarks. That code was removed because the paper studies circuits in general.
It is available with `git checkout v0-pre-refactor`. The results stay in the ADR and the paper. The original task
specification and audit are in `docs/history/`. Result files written by the old scripts keep their old keys
(for example `oab_acc`, `law_logp`); the new suite writes `target_acc`, `control_acc`, `fact_logp`.
