"""Command-line interface for the compression pipeline.

    python -m compress collect_stats       --model-path M --calib wealth=data/wealth_train.csv --out stats.pt
    python -m compress estimate_quant_error --model-path M --stats stats.pt --out errors.json
    python -m compress allocate_bits       --errors errors.json --target wealth --secondary honesty \
                                           --error-budget 1e-3 --out plan.json
    python -m compress quantize_model      --model-path M --plan plan.json --out-dir out/compressed
    python -m compress evaluate_compression --model-path M [--compressed-dir out/compressed] \
                                            --test wealth=Persona/data/wealth_test_infer.csv --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from . import stats as stats_mod
from . import serialize
from .allocate import estimate_blocks, estimate_model_blocks, build_plan, load_plan, save_plan
from .contrast import MASK_KINDS, apply_domain_mask
from .evaluate import PROMPT_STYLES, evaluate_mc, evaluate_mmlu

DEFAULT_FORMATS = "fp16,int8,int4,nf4"


def _safetensors_path(model_path: str) -> Path | None:
    """Single-file safetensors checkpoint (the streamed, low-RAM path)."""
    for name in ("model.safetensors", "model.safetensors-00001-of-00001.safetensors"):
        p = Path(model_path) / name
        if p.exists():
            return p
    return None


@contextmanager
def _open_safetensors(path: Path):
    """Open a checkpoint once: safe_open re-reads the header on every call."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        yield f, set(f.keys())


def _weight_from(handle, keys: set[str], name: str):
    """Read one Linear weight from an open checkpoint.

    Falls back to the embedding matrix for ``lm_head`` when embeddings are
    tied and the head is therefore not stored separately.
    """
    key = f"{name}.weight"
    if key in keys:
        return handle.get_tensor(key)
    if name == "lm_head" and "model.embed_tokens.weight" in keys:
        return handle.get_tensor("model.embed_tokens.weight")
    raise KeyError(key)


def _stream_weights(path: Path, layer_names):
    """Yield ``(name, weight)`` for the requested layers, one tensor at a time.

    Only one weight matrix is resident at any moment.
    """
    with _open_safetensors(path) as (f, keys):
        for name in sorted(layer_names):
            yield name, _weight_from(f, keys, name)


def _parse_pairs(values: list[str], what: str) -> dict[str, str]:
    """Parse repeated ``name=path`` arguments into a dict."""
    out = {}
    for v in values:
        if "=" not in v:
            raise SystemExit(f"--{what} expects name=path, got {v!r}")
        name, path = v.split("=", 1)
        if name in out:
            raise SystemExit(f"--{what} {name!r} given more than once")
        out[name] = path
    return out


def _parse_mmlu(values: list[str]) -> dict[str, tuple[str, str]]:
    """Parse repeated ``name=subject[:split]`` arguments into ``name -> (subject, split)``."""
    out = {}
    for v in values:
        if "=" not in v:
            raise SystemExit(f"--mmlu expects name=subject[:split], got {v!r}")
        name, spec = v.split("=", 1)
        subject, _, split = spec.partition(":")
        if not subject:
            raise SystemExit(f"--mmlu {name!r} has an empty subject")
        if name in out:
            raise SystemExit(f"--mmlu {name!r} given more than once")
        out[name] = (subject, split or "test")
    return out


def _parse_formats(value: str) -> list[str]:
    return [f.strip() for f in value.split(",") if f.strip()]


def _load_model(path: str):
    # transformers renamed torch_dtype -> dtype in v5; support both pins.
    import transformers

    kwarg = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    # CPU runs use bf16: half the resident memory of fp32, and every compared
    # configuration is evaluated in the same dtype so comparisons stay fair.
    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.bfloat16
    config = AutoConfig.from_pretrained(path)
    if getattr(config, "vision_config", None) is not None:
        # Multimodal checkpoints (e.g. Qwen3.5-Base) are not CausalLM classes:
        # AutoModelForCausalLM would silently skip their text weights.
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(path, **{kwarg: dtype})
        # Text-only pipeline: the vision tower is never used and costs 1-2 GB
        # of VRAM (which the 9B model needs during quantization).
        if hasattr(model.model, "visual"):
            del model.model.visual
    else:
        model = AutoModelForCausalLM.from_pretrained(path, **{kwarg: dtype})
    if cuda:
        model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model.eval(), tokenizer


def cmd_collect_stats(args) -> dict:
    model, tokenizer = _load_model(args.model_path)
    calibs = _parse_pairs(args.calib, "calib")
    if not calibs:
        raise SystemExit("provide at least one --calib domain=file.csv")
    domains = {
        domain: stats_mod.collect_domain(
            model, tokenizer, Path(path).read_text(encoding="utf-8").splitlines(),
            batch_size=args.batch_size, max_len=args.max_len,
            n_samples=args.n_samples, seed=args.seed,
        )
        for domain, path in calibs.items()
    }
    meta = {
        "model": args.model_path, "seed": args.seed, "batch_size": args.batch_size,
        "max_len": args.max_len, "n_samples": args.n_samples,
        "calib_files": calibs,
    }
    stats_mod.save_stats(domains, args.out, meta)
    n_tokens = {d: next(iter(v.values()))["count"] for d, v in domains.items()}
    return {"out": args.out, "layers": len(next(iter(domains.values()))), "valid_tokens": n_tokens}


def cmd_estimate_quant_error(args) -> dict:
    domains, meta = stats_mod.load_stats(args.stats)
    formats = _parse_formats(args.formats)
    names = sorted(next(iter(domains.values())))

    estimates, source = None, "model-load"
    st = _safetensors_path(args.model_path)
    if st is not None:
        try:  # low-RAM path: stream one weight at a time from the checkpoint
            estimates = estimate_blocks(_stream_weights(st, names), domains, formats)
            source = "safetensors"
        except KeyError:  # stats layer missing from the checkpoint -> full load
            estimates = None
    if estimates is None:
        model, _ = _load_model(args.model_path)
        estimates = estimate_model_blocks(model, domains, formats)

    # JSON needs lists, not tuples.
    for e in estimates.values():
        e["shape"] = list(e["shape"])
    blob = {
        "meta": {"model": args.model_path, "stats_file": str(args.stats),
                 "stats_meta": meta, "formats": formats, "weights_source": source},
        "layers": estimates,
    }
    Path(args.out).write_text(json.dumps(blob, indent=1, sort_keys=True))
    return {"out": args.out, "layers": len(estimates), "formats": formats, "weights_source": source}


def cmd_allocate_bits(args) -> dict:
    blob = json.loads(Path(args.errors).read_text())
    estimates = blob["layers"]
    for e in estimates.values():
        e["shape"] = tuple(e["shape"])
    formats = _parse_formats(args.formats)
    plan = build_plan(
        estimates,
        target=args.target,
        formats=formats,
        error_budget=args.error_budget,
        size_budget=args.size_budget,
        secondary=args.secondary or [],
        lam=args.lam,
        signal=args.signal,
        model=blob["meta"].get("model", ""),
    )
    save_plan(plan, args.out)
    t = plan["totals"]
    fmt_counts: dict[str, int] = {}
    for spec in plan["blocks"].values():
        fmt_counts[spec["format"]] = fmt_counts.get(spec["format"], 0) + 1
    return {
        "out": args.out, "format_counts": fmt_counts,
        "bytes_fp16": t["bytes_fp16"], "bytes_final": t["bytes_final"],
        "estimated_error_target": t["estimated_error_target"],
        "steps": len(plan["decisions"]),
    }


def cmd_quantize_model(args) -> dict:
    plan = load_plan(args.plan)
    manifest, source = None, "model-load"
    st = _safetensors_path(args.model_path)
    if st is not None:
        try:  # low-RAM path: quantize straight from the memory-mapped checkpoint
            with _open_safetensors(st) as (f, keys):
                manifest = serialize.save_compressed_streamed(
                    lambda name: _weight_from(f, keys, name), plan, args.out_dir
                )
            source = "safetensors"
        except KeyError:
            manifest = None
    if manifest is None:
        model, _ = _load_model(args.model_path)
        manifest = serialize.save_compressed(model, plan, args.out_dir)
    return {
        "out_dir": str(args.out_dir),
        "blocks": len(manifest["blocks"]),
        "compressed_tensor_bytes": manifest["totals"]["file_nbytes"],
        "fp16_reference_bytes": manifest["totals"]["fp16_reference_nbytes"],
        "dir_bytes": serialize.measure_bytes(args.out_dir),
        "weights_source": source,
    }


def cmd_evaluate_compression(args) -> dict:
    model, tokenizer = _load_model(args.model_path)
    if args.compressed_dir:
        n = serialize.apply_to_model(model, args.compressed_dir)
        manifest = json.loads((Path(args.compressed_dir) / "manifest.json").read_text())
    else:
        n, manifest = 0, None
    mask = None
    if args.mask_kind != "none":
        if not (args.mask_stats and args.mask_seek and args.mask_reject):
            raise SystemExit(
                "--mask-kind requires --mask-stats, --mask-seek and --mask-reject"
            )
        domains, _ = stats_mod.load_stats(args.mask_stats)
        mask = apply_domain_mask(
            model, domains, kind=args.mask_kind, seek=args.mask_seek,
            reject=args.mask_reject, sparsity=args.mask_sparsity, seed=args.mask_seed,
            only_mlp=args.mask_only_mlp,
        )
    tests = _parse_pairs(args.test, "test")
    mmlu = _parse_mmlu(args.mmlu)
    if not tests and not mmlu:
        raise SystemExit("provide at least one --test name=file.csv or --mmlu name=subject[:split]")
    if set(tests) & set(mmlu):
        raise SystemExit(f"names given to both --test and --mmlu: {sorted(set(tests) & set(mmlu))}")
    results = {
        name: evaluate_mc(
            model, tokenizer, path, negate=args.negate, sys_prompt=args.sys_prompt,
            max_questions=args.max_questions, prompt_style=args.prompt_style,
        )
        for name, path in tests.items()
    }
    for name, (subject, split) in mmlu.items():
        results[name] = evaluate_mmlu(
            model, tokenizer, subject, split, sys_prompt=args.sys_prompt,
            max_questions=args.max_questions, prompt_style=args.prompt_style,
            cache_dir=args.mmlu_cache,
        )
    report = {
        "model": args.model_path,
        "dtype": str(next(model.parameters()).dtype),  # bf16 on CPU, fp16 on CUDA
        "prompt_style": args.prompt_style,
        "compressed_dir": str(args.compressed_dir) if args.compressed_dir else None,
        "compressed_blocks": n,
        "model_param_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "mask": mask,
        "results": results,
    }
    if manifest is not None:
        report["linear_fp16_reference_bytes"] = manifest["totals"]["fp16_reference_nbytes"]
        report["compressed_tensor_bytes"] = manifest["totals"]["file_nbytes"]
        report["compression_ratio_vs_fp16_linears"] = (
            manifest["totals"]["fp16_reference_nbytes"] / manifest["totals"]["file_nbytes"]
        )
        report["compressed_dir_bytes"] = serialize.measure_bytes(args.compressed_dir)
    Path(args.out).write_text(json.dumps(report, indent=1, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="compress", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("collect_stats", help="per-domain activation statistics")
    p.add_argument("--model-path", required=True)
    p.add_argument("--calib", action="append", default=[], metavar="DOMAIN=FILE")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--n-samples", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_collect_stats)

    p = sub.add_parser("estimate_quant_error", help="per-block diagonal quant error")
    p.add_argument("--model-path", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--formats", default=DEFAULT_FORMATS)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_estimate_quant_error)

    p = sub.add_parser("allocate_bits", help="greedy heterogeneous allocation")
    p.add_argument("--errors", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--secondary", action="append", default=[])
    p.add_argument("--formats", default=DEFAULT_FORMATS)
    p.add_argument("--error-budget", type=float, default=None)
    p.add_argument("--size-budget", type=int, default=None)
    p.add_argument("--signal", choices=["error", "contrastive"], default="error")
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_allocate_bits)

    p = sub.add_parser("quantize_model", help="materialize a plan to disk")
    p.add_argument("--model-path", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--out-dir", required=True)
    p.set_defaults(func=cmd_quantize_model)

    p = sub.add_parser("evaluate_compression", help="MC evaluation + artifact sizes")
    p.add_argument("--model-path", required=True)
    p.add_argument("--compressed-dir", default=None)
    p.add_argument("--test", action="append", default=[], metavar="NAME=FILE")
    p.add_argument("--mmlu", action="append", default=[], metavar="NAME=SUBJECT[:SPLIT]")
    p.add_argument("--mmlu-cache", default=None)
    p.add_argument("--mask-kind", choices=MASK_KINDS, default="none")
    p.add_argument("--mask-stats", default=None)
    p.add_argument("--mask-seek", default=None)
    p.add_argument("--mask-reject", default=None)
    p.add_argument("--mask-sparsity", type=float, default=0.3)
    p.add_argument("--mask-seed", type=int, default=0)
    p.add_argument("--mask-only-mlp", action="store_true")
    p.add_argument("--negate", action="store_true")
    p.add_argument("--sys-prompt", default="")
    p.add_argument("--prompt-style", choices=PROMPT_STYLES, default="auto")
    p.add_argument("--max-questions", type=int, default=None)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_evaluate_compression)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        summary = args.func(args)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
