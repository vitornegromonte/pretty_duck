"""End-to-end experiment on a real, trained SmolLM model.

Usage (repo root):  python experiments/run_smollm.py [HF_MODEL_ID]
Default model: HuggingFaceTB/SmolLM-135M; artifacts go to
outputs/<model-short-name>/ (so different models never clobber each other
and interrupted runs resume from the first missing artifact).

Compares, at comparable storage budgets:
    baseline            -- no compression (original weights)
    uniform_int8 / _int4 / _nf4
    hetero_error        -- heterogeneous allocation on estimated target error
    hetero_contrastive  -- same, with contrastive importance as auxiliary signal

Target domain = wealth (Persona/data/wealth_*), secondary = honesty (h_*).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from compress.cli import main as cli

MODEL = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM-135M"
OUT = Path("outputs") / MODEL.split("/")[-1].lower()
DATA = Path("Persona/data")
SIZE_FRACTION = 0.5  # heterogeneous runs target 50% of the fp16 Linear payload
LAM = 0.5


def run(*argv) -> float:
    """Invoke a pipeline command in-process; returns wall seconds."""
    t0 = time.perf_counter()
    cli([str(a) for a in argv])
    return time.perf_counter() - t0


def write_uniform_plans(est: dict, fp16_bytes: int) -> dict[str, Path]:
    plans = {}
    for fmt in ("int8", "int4", "nf4"):
        plan = {
            "version": 1, "model": MODEL, "target_domain": "wealth",
            "secondary_domains": ["honesty"], "formats": ["fp16", fmt],
            "signal": "uniform", "lambda": 0.0,
            "budget": {"mode": "uniform", "value": fmt},
            "blocks": {name: {"format": fmt} for name in est},
            "decisions": [],
            "totals": {
                "bytes_fp16": fp16_bytes,
                "bytes_final": sum(e["nbytes"][fmt] for e in est.values()),
                "estimated_error_target": sum(e["error"]["wealth"][fmt] for e in est.values()),
            },
        }
        path = OUT / f"plan_uniform_{fmt}.json"
        path.write_text(json.dumps(plan, indent=1, sort_keys=True))
        plans[f"uniform_{fmt}"] = path
    return plans


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats, errors = OUT / "stats.pt", OUT / "errors.json"

    timings: dict[str, float] = {}
    # Cheap resume: completed stages leave their artifact behind; skip them.
    if not stats.exists():
        timings["collect_stats"] = run(
            "collect_stats", "--model-path", MODEL,
            "--calib", f"wealth={DATA / 'wealth_train.csv'}",
            "--calib", f"honesty={DATA / 'h_train.csv'}",
            "--out", stats, "--n-samples", "128", "--max-len", "256",
            "--batch-size", "4", "--seed", "42",
        )
    if not errors.exists():
        timings["estimate_quant_error"] = run(
            "estimate_quant_error", "--model-path", MODEL,
            "--stats", stats, "--formats", "fp16,int8,int4,nf4,bnb_nf4", "--out", errors,
        )

    est = json.loads(errors.read_text())["layers"]
    fp16_bytes = sum(e["nbytes"]["fp16"] for e in est.values())
    size_budget = int(fp16_bytes * SIZE_FRACTION)

    plans = write_uniform_plans(est, fp16_bytes)
    for name, signal in (("hetero_error", "error"), ("hetero_contrastive", "contrastive")):
        path = OUT / f"plan_{name}.json"
        timings[f"allocate_{name}"] = run(
            "allocate_bits", "--errors", errors,
            "--target", "wealth", "--secondary", "honesty",
            "--formats", "fp16,int8,int4,nf4",
            "--size-budget", str(size_budget), "--signal", signal,
            "--lam", str(LAM), "--out", path,
        )
        plans[name] = path

    configs: list[tuple[str, Path | None]] = [("baseline", None)] + list(plans.items())
    report: dict = {"model": MODEL, "target": "wealth", "secondary": "honesty",
                    "size_budget_bytes": size_budget, "fp16_linear_bytes": fp16_bytes,
                    "timings_s": timings, "configs": {}}
    for name, plan_path in configs:
        compressed = OUT / f"compressed_{name}"
        argv = ["evaluate_compression", "--model-path", MODEL,
                "--test", f"wealth={DATA / 'wealth_test_infer.csv'}",
                "--test", f"honesty={DATA / 'h_test_infer.csv'}",
                "--out", OUT / f"report_{name}.json"]
        if plan_path is not None:
            timings[f"quantize_{name}"] = run(
                "quantize_model", "--model-path", MODEL,
                "--plan", plan_path, "--out-dir", compressed,
            )
            argv += ["--compressed-dir", compressed]
        timings[f"evaluate_{name}"] = run(*argv)
        run_report = json.loads((OUT / f"report_{name}.json").read_text())
        report["configs"][name] = {
            "plan": str(plan_path) if plan_path else None,
            "acc_wealth": run_report["results"]["wealth"]["accuracy"],
            "acc_honesty": run_report["results"]["honesty"]["accuracy"],
            "avg_latency_ms_wealth": run_report["results"]["wealth"]["avg_latency_ms"],
            "compressed_tensor_bytes": run_report.get("compressed_tensor_bytes"),
            "compressed_dir_bytes": run_report.get("compressed_dir_bytes"),
            "compression_ratio_vs_fp16_linears": run_report.get("compression_ratio_vs_fp16_linears"),
            "estimated_error_target": (
                json.loads(plan_path.read_text())["totals"]["estimated_error_target"]
                if plan_path else 0.0
            ),
        }

    (OUT / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True))

    header = f"{'config':<20} {'acc_w%':>7} {'acc_h%':>7} {'tensor_B':>10} {'ratio':>6} {'ms/q':>7}"
    print(f"\n{header}\n{'-' * len(header)}")
    for name, row in report["configs"].items():
        tensor_b = row["compressed_tensor_bytes"] or report["fp16_linear_bytes"]
        ratio = row["compression_ratio_vs_fp16_linears"] or 1.0
        print(f"{name:<20} {row['acc_wealth']:>7.2f} {row['acc_honesty']:>7.2f} "
              f"{tensor_b:>10} {ratio:>6.2f} {row['avg_latency_ms_wealth']:>7.1f}")
    print(f"\nreport: {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
