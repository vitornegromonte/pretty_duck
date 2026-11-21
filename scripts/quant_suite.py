"""Run a suite of quantization conditions on one model (see ``compress/suite.py`` for the conditions).

Evaluation uses two groups of MMLU subjects: ``target`` and ``control``. The protected circuit comes from a
``circuit_probe.py`` result file for ``--domain``. Results are written after every condition, and a restart skips the
conditions that are already done.

    PYTHONPATH=. python scripts/quant_suite.py --model-path Qwen/Qwen3.5-4B \
        --probe outputs/circuit_probe.json --stats outputs/stats.pt --calib-dir outputs/calib2 \
        --neutral outputs/eval/hum.txt --domain capitals \
        --target-subjects geography --control-subjects high_school_mathematics \
        --suites grid,protect --out outputs/quant_suite.json
"""

from __future__ import annotations

import argparse
import logging
import sys

from compress.experiment import load_context
from compress.suite import SUITES, SYS_PROMPT, QuantSuite, SuiteConfig


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, SuiteConfig]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True)
    p.add_argument("--probe", required=True, help="circuit_probe.py result file")
    p.add_argument("--stats", required=True, help="stats file with per-channel activation moments")
    p.add_argument("--calib-dir", required=True, help="folder with bio.txt, cs.txt, hum.txt, law.txt, math.txt")
    p.add_argument("--neutral", required=True, help="text file for the perplexity check")
    p.add_argument("--domain", default="capitals", help="fact domain whose circuit is protected")
    p.add_argument("--target-subjects", required=True, help="comma-separated MMLU subjects of the target group")
    p.add_argument("--control-subjects", required=True, help="comma-separated MMLU subjects of the control group")
    p.add_argument("--suites", default=",".join(SUITES[:-1]), help=f"comma-separated, from {', '.join(SUITES)}")
    p.add_argument("--only", default="", help="comma-separated condition names (default: all of the suites)")
    p.add_argument("--bits", type=int, default=3)
    p.add_argument("--group", type=int, default=64)
    p.add_argument("--ks", default="128", help="circuit sizes in units, comma-separated")
    p.add_argument("--n-random", type=int, default=2)
    p.add_argument("--reps", type=int, default=3, help="calibration replicates of each GPTQ protect condition")
    p.add_argument("--roundings", default="gptq", help="rounding of the protect suite: rtn, gptq, comma-separated")
    p.add_argument("--keep-blocks", action="store_true", help="RTN with protected columns keeps the 64-column blocks")
    p.add_argument("--n-eval", type=int, default=300, help="evaluation questions per group")
    p.add_argument("--n-calib", type=int, default=160)
    p.add_argument("--sys-prompt", default=SYS_PROMPT)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    suites = tuple(s for s in args.suites.split(",") if s)
    unknown = set(suites) - set(SUITES)
    if unknown:
        p.error(f"unknown suites {sorted(unknown)}")
    cfg = SuiteConfig(
        out=args.out, calib_dir=args.calib_dir, neutral=args.neutral,
        target_subjects=args.target_subjects.split(","), control_subjects=args.control_subjects.split(","),
        suites=suites, only=frozenset(s for s in args.only.split(",") if s), bits=args.bits, group=args.group,
        ks=tuple(int(k) for k in args.ks.split(",")), n_random=args.n_random, reps=args.reps,
        roundings=tuple(args.roundings.split(",")), keep_blocks=args.keep_blocks, n_eval=args.n_eval,
        n_calib=args.n_calib, sys_prompt=args.sys_prompt)
    return args, cfg


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(message)s")
    args, cfg = parse_args()
    ctx = load_context(args.model_path, args.probe, args.stats, args.domain)
    QuantSuite(ctx, cfg).run()


if __name__ == "__main__":
    main()
