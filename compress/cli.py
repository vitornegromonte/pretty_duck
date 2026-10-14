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
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import stats as stats_mod
from . import serialize
from .allocate import estimate_blocks, build_plan, load_plan, save_plan
from .evaluate import evaluate_mc

DEFAULT_FORMATS = "fp16,int8,int4,nf4"


def _parse_pairs(values: list[str], what: str) -> dict[str, str]:
    """Parse repeated ``name=path`` arguments into a dict."""
    out = {}
    for v in values:
        if "=" not in v:
            raise SystemExit(f"--{what} expects name=path, got {v!r}")
        name, path = v.split("=", 1)
        out[name] = path
    return out


def _load_model(path: str):
    # transformers renamed torch_dtype -> dtype in v5; support both pins.
    import transformers

    kwarg = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, **{kwarg: dtype})
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
    model, _ = _load_model(args.model_path)
    domains, meta = stats_mod.load_stats(args.stats)
    formats = [f for f in args.formats.split(",") if f]
    estimates = estimate_blocks(model, domains, formats)
    # JSON needs lists, not tuples.
    for e in estimates.values():
        e["shape"] = list(e["shape"])
    blob = {
        "meta": {"model": args.model_path, "stats_file": str(args.stats),
                 "stats_meta": meta, "formats": formats},
        "layers": estimates,
    }
    Path(args.out).write_text(json.dumps(blob, indent=1, sort_keys=True))
    return {"out": args.out, "layers": len(estimates), "formats": formats}


def cmd_allocate_bits(args) -> dict:
    blob = json.loads(Path(args.errors).read_text())
    estimates = blob["layers"]
    for e in estimates.values():
        e["shape"] = tuple(e["shape"])
    formats = [f for f in args.formats.split(",") if f]
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
    model, _ = _load_model(args.model_path)
    plan = load_plan(args.plan)
    manifest = serialize.save_compressed(model, plan, args.out_dir)
    return {
        "out_dir": str(args.out_dir),
        "blocks": len(manifest["blocks"]),
        "compressed_tensor_bytes": manifest["totals"]["file_nbytes"],
        "fp16_reference_bytes": manifest["totals"]["fp16_reference_nbytes"],
        "dir_bytes": serialize.measure_bytes(args.out_dir),
    }


def cmd_evaluate_compression(args) -> dict:
    model, tokenizer = _load_model(args.model_path)
    if args.compressed_dir:
        n = serialize.apply_to_model(model, args.compressed_dir)
        manifest = json.loads((Path(args.compressed_dir) / "manifest.json").read_text())
    else:
        n, manifest = 0, None
    tests = _parse_pairs(args.test, "test")
    if not tests:
        raise SystemExit("provide at least one --test name=file.csv")
    report = {
        "model": args.model_path,
        "compressed_dir": str(args.compressed_dir) if args.compressed_dir else None,
        "compressed_blocks": n,
        "model_param_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "results": {
            name: evaluate_mc(
                model, tokenizer, path, negate=args.negate, sys_prompt=args.sys_prompt,
                max_questions=args.max_questions,
            )
            for name, path in tests.items()
        },
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
    p.add_argument("--negate", action="store_true")
    p.add_argument("--sys-prompt", default="")
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
