"""Materialize an allocation plan to disk and measure the real artifact.

Only ``nn.Linear`` weights are rewritten; embeddings and norms stay at the
base model's precision and are reloaded from the original model at inference
time.  Each block becomes one ``torch.save`` payload file; the manifest
records both the analytic payload size and the actual file size on disk
(torch serialization overhead included), so reports never rely on
nominal bits-per-parameter.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .quant import quantize_weight, dequantize_weight, storage_nbytes


def save_compressed(
    model: torch.nn.Module, plan: dict, out_dir: str | Path
) -> dict:
    """Quantize every planned block and write tensors + manifest under ``out_dir``.

    Returns the manifest dict (also written to ``out_dir/manifest.json``).
    The model itself is not modified.
    """
    out_dir = Path(out_dir)
    tensors_dir = out_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    named = dict(model.named_modules())

    manifest: dict = {
        "version": 1,
        "model": plan.get("model", ""),
        "target_domain": plan.get("target_domain", ""),
        "blocks": {},
        "totals": {"payload_nbytes": 0, "file_nbytes": 0, "fp16_reference_nbytes": 0},
    }
    for name, spec in plan["blocks"].items():
        module = named.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise KeyError(f"plan block {name!r} is not an nn.Linear in the model")
        fmt = spec["format"]
        W = module.weight.detach()
        q = quantize_weight(W, fmt)
        file_path = tensors_dir / f"{name}.pt"
        payload = {"format": q.format, "shape": tuple(W.shape), "codes": q.codes, "scale": q.scale}
        if q.meta is not None:
            payload.update(q.meta)  # bitsandbytes QuantState etc.
        torch.save(payload, file_path)
        file_nbytes = file_path.stat().st_size
        manifest["blocks"][name] = {
            "file": str(file_path.relative_to(out_dir)),
            "format": fmt,
            "shape": list(W.shape),
            "payload_nbytes": q.nbytes,
            "file_nbytes": file_nbytes,
            "analytic_nbytes": storage_nbytes(tuple(W.shape), fmt),
        }
        manifest["totals"]["payload_nbytes"] += q.nbytes
        manifest["totals"]["file_nbytes"] += file_nbytes
        manifest["totals"]["fp16_reference_nbytes"] += W.numel() * 2

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def load_block(manifest: dict, out_dir: str | Path, name: str) -> torch.Tensor:
    """Read one stored block and reconstruct its float32 weight."""
    info = manifest["blocks"][name]
    payload = torch.load(Path(out_dir) / info["file"], map_location="cpu", weights_only=False)
    return dequantize_weight(
        payload["codes"], payload.get("scale", payload.get("state")), payload["format"],
        tuple(payload["shape"]),
    )


def apply_to_model(model: torch.nn.Module, out_dir: str | Path) -> int:
    """Copy every stored block into ``model`` (in-place) and return the count.

    Mutates ``model``: reload the model from its original source before
    running a different configuration.
    """
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    named = dict(model.named_modules())
    for name in manifest["blocks"]:
        module = named[name]
        weight = load_block(manifest, out_dir, name)
        if tuple(weight.shape) != tuple(module.weight.shape):
            raise ValueError(
                f"shape mismatch for {name}: stored {tuple(weight.shape)} vs "
                f"model {tuple(module.weight.shape)}"
            )
        module.weight.data.copy_(weight.to(module.weight.device))
    return len(manifest["blocks"])


def measure_bytes(out_dir: str | Path) -> int:
    """Total bytes of every file under ``out_dir`` (manifest included)."""
    return sum(p.stat().st_size for p in Path(out_dir).rglob("*") if p.is_file())
