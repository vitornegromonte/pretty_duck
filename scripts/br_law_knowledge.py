"""How much Brazilian law does a model know?  Two measurements.

1. OAB questions scored by answer letter (chance 25%), overall and per area.
2. Cloze facts from the review sheet: how many survive tokenization, share one
   prompt length (the probe's requirement) and are answered correctly (first
   token), per relation.  Facts a model does not know cannot carry a circuit.

    PYTHONPATH=. python scripts/br_law_knowledge.py --model-path Qwen/Qwen3.5-0.8B \
        --oab outputs/data/oab_exams/oab_exams.jsonl \
        --facts data/legal_facts/fatos_direito_br_rascunho.csv --out outputs/br_law_<model>.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

from compress.circuit import build_facts
from compress.cli import _load_model
from compress.evaluate import evaluate_mmlu_rows
from compress.oab import FEW_SHOT, load_fact_csv, load_oab


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--oab", required=True)
    p.add_argument("--facts", required=True)
    p.add_argument("--max-questions", type=int, default=800, help="seeded sample of the OAB questions")
    p.add_argument("--prompt-style", default="plain")
    p.add_argument("--sys-prompt", default="As questões a seguir são de provas da OAB. Responda com a letra da alternativa correta.")
    p.add_argument("--few-shot", action="store_true", help="prefix each relation's template with fixed examples")
    p.add_argument("--skip-oab", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    model, tokenizer = _load_model(args.model_path)
    oab = None
    if not args.skip_oab:
        rows = load_oab(args.oab)
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_questions]
        res = evaluate_mmlu_rows(model, tokenizer, rows, sys_prompt=args.sys_prompt, prompt_style=args.prompt_style)
        by_area = defaultdict(list)
        for r, pred in zip(rows, res["predictions"]):
            by_area[r["area"]].append(pred["predicted"] == pred["expected"])
        oab = {"n": res["n"], "accuracy": res["accuracy"],
               "by_area": {a: {"n": len(v), "accuracy": 100 * st.mean(v)} for a, v in sorted(by_area.items())}}
        print(f"OAB: {oab['accuracy']:.1f}% on {oab['n']} questions (chance 25%)")
        for a, v in oab["by_area"].items():
            print(f"   {a:18s} n={v['n']:4d}  {v['accuracy']:.1f}%")

    facts = {}
    for rel, spec in load_fact_csv(args.facts).items():
        listed = len(spec["pairs"])
        template = (FEW_SHOT.get(rel, "") + spec["template"]) if args.few_shot else spec["template"]
        clean = build_facts(model, tokenizer, template, spec["pairs"], require_correct=False)
        known = build_facts(model, tokenizer, template, spec["pairs"])
        facts[rel] = {"listed": listed, "same_length_clean": len(clean), "known": len(known),
                      "known_subjects": [f.subject for f in known]}
        print(f"facts {rel:14s} listed {listed:3d}  same-length+clean {len(clean):3d}  known {len(known):3d}")
    Path(args.out).write_text(json.dumps({"model": args.model_path, "oab": oab, "facts": facts,
                                          "prompt_style": args.prompt_style, "few_shot": args.few_shot}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
