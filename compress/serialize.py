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
from .stats import tied_weight_ptrs


def save_compressed(
    model: torch.nn.Module, plan: dict, out_dir: str | Path
) -> dict:
    """Quantize every planned block of an in-memory model (see streamed variant)."""
    named = dict(model.named_modules())

    def get_weight(name: str) -> torch.Tensor:
        module = named.get(name)
        if not isinstance(module, torch.nn.Linear):
            raise KeyError(f"plan block {name!r} is not an nn.Linear in the model")
        return module.weight.detach()

    return save_compressed_streamed(get_weight, plan, out_dir)


def save_compressed_streamed(
    get_weight, plan: dict, out_dir: str | Path
) -> dict:
    """Quantize every planned block and write tensors + manifest under ``out_dir``.

    ``get_weight(name) -> Tensor`` may read one layer at a time (e.g. from a
    memory-mapped safetensors file), so peak memory stays at one weight
    matrix.  Returns the manifest dict (also written to
    ``out_dir/manifest.json``).  No model object is modified.
    """
    out_dir = Path(out_dir)
    tensors_dir = out_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "version": 1,
        "model": plan.get("model", ""),
        "target_domain": plan.get("target_domain", ""),
        "blocks": {},
        "totals": {"payload_nbytes": 0, "file_nbytes": 0, "fp16_reference_nbytes": 0},
    }
    for name, spec in plan["blocks"].items():
        fmt = spec["format"]
        W = get_weight(name)
        if W.dim() != 2:
            raise ValueError(f"{name}: expected a 2-D Linear weight, got {W.dim()}-D")
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

    # Blocks a previous plan wrote but this one does not are still counted by
    # measure_bytes; drop them once the new ones are safely on disk.
    written = {tensors_dir / f"{name}.pt" for name in manifest["blocks"]}
    for stale in tensors_dir.glob("*.pt"):
        if stale not in written:
            stale.unlink()

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


def load_block(manifest: dict, out_dir: str | Path, name: str) -> torch.Tensor:
    """Read one stored block and reconstruct its float32 weight."""
    info = manifest["blocks"][name]
    # bnb_nf4 stores a pickled bitsandbytes QuantState, which weights_only rejects.
    payload = torch.load(
        Path(out_dir) / info["file"], map_location="cpu",
        weights_only=info["format"] != "bnb_nf4",
    )
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
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found; quantize_model did not finish writing this artifact"
        )
    manifest = json.loads(manifest_path.read_text())
    named = dict(model.named_modules())
    tied = tied_weight_ptrs(model)  # e.g. a lm_head tied to the embeddings
    applied = 0
    for name in manifest["blocks"]:
        module = named[name]
        if module.weight.data_ptr() in tied:
            continue  # storing it would rewrite the tied module as well
        weight = load_block(manifest, out_dir, name)
        if tuple(weight.shape) != tuple(module.weight.shape):
            raise ValueError(
                f"shape mismatch for {name}: stored {tuple(weight.shape)} vs "
                f"model {tuple(module.weight.shape)}"
            )
        # copy_ casts and moves CPU->GPU without a full-size GPU temporary,
        # which matters for an untied lm_head on a large model.
        module.weight.data.copy_(weight)
        applied += 1
    return applied


def measure_bytes(out_dir: str | Path) -> int:
    """Total bytes of every file under ``out_dir`` (manifest included)."""
    return sum(p.stat().st_size for p in Path(out_dir).rglob("*") if p.is_file())
