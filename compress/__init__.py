"""Domain-aware low-bit quantization of language models, with interpretability tests.

Modules:
    models       -- load a causal LM and its tokenizer the same way everywhere
    mmlu         -- cached MMLU question files
    facts        -- templated factual-recall domains (capitals, elements, french)
    evaluate     -- multiple-choice and next-token scoring
    stats        -- per-channel activation statistics from calibration text
    quant        -- quantization formats and exact storage sizes
    quant_probe  -- fake quantization (whole matrices, groups, chosen columns)
    gptq         -- GPTQ with protected columns, clipping, and a low-rank correction
    protect      -- whole-model quantization with columns kept at full precision
    sensitivity  -- measured damage of quantizing one layer at a time
    pair_patching, circuit -- activation patching and circuit discovery
    provenance   -- code, software, and hardware record stored in result files
"""
