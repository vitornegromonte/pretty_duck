#!/usr/bin/env bash
# One-shot Colab runner. In a Colab cell (GPU runtime recommended):
#   !git clone -b feature/domain-compression https://github.com/vitornegromonte/pretty_duck.git
#   %cd pretty_duck
#   !bash script/run_all.sh                       # default: SmolLM-360M
#   !bash script/run_all.sh HF/user-model         # any single-file HF checkpoint
set -euo pipefail

MODEL="${1:-HuggingFaceTB/SmolLM-360M}"

pip install -q transformers bitsandbytes safetensors

# Collects stats, estimates errors, allocates, quantizes, evaluates, and
# prints the summary table. Skips any stage whose artifact already exists.
python experiments/run_smollm.py "$MODEL"
