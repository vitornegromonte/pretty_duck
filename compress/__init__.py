"""Domain-aware weight compression pipeline built on top of the Persona code.

Modules:
    stats      -- per-domain, per-channel activation statistics (E|h|, E[h^2])
    quant      -- quantization formats (simulated + packed, optional bitsandbytes)
    allocate   -- greedy heterogeneous bit allocation per Linear block
    serialize  -- materialize a plan to disk and measure real artifact bytes
    evaluate   -- multiple-choice evaluation (same protocol as Persona/contra.py)
    cli        -- collect_stats / estimate_quant_error / allocate_bits /
                  quantize_model / evaluate_compression
"""

__version__ = "0.1.0"
