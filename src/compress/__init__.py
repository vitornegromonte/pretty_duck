"""Domain-aware low-bit quantization of language models, with interpretability tests.

Subpackages:
    quant        -- how weights are quantized: formats, fake quantization, GPTQ, protected columns, statistics
    interp       -- which components matter: units, minimal pairs, activation patching, circuits, damage
    evaluation   -- scoring: multiple-choice questions and text perplexity
    data         -- inputs: MMLU questions, templated fact domains, calibration text
    experiments  -- experiment setup and the suite of quantization conditions

Top-level modules:
    models       -- load a causal LM and its tokenizer the same way everywhere
    provenance   -- code, software, and hardware record stored in result files
"""
