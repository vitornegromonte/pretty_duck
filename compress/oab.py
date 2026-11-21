"""Brazilian-law data: OAB exam questions and reviewable cloze facts.

``eduagarcia/oab_exams`` (four-option questions from the Bar exams) is converted
once to JSONL by ``scripts/oab_prepare.py``; ``load_oab`` returns rows in the
MMLU layout (``question``, ``choices``, ``answer`` index) plus ``area``, so
``evaluate_mmlu_rows`` scores them unchanged.  ``load_fact_csv`` reads the
cloze-fact sheet a jurist reviews (``data/legal_facts/``).
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

LETTERS = "ABCD"


def load_oab(path: str | Path, *, drop_nullified: bool = True) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if drop_nullified and str(r.get("nullified")).lower() == "true":
            continue
        choices = r["choices"]["text"] if isinstance(r["choices"], dict) else r["choices"]
        if len(choices) != 4 or r["answerKey"] not in LETTERS:
            continue
        rows.append({
            "question": r["question"], "choices": list(choices), "answer": LETTERS.index(r["answerKey"]),
            "area": r.get("question_type") or "SEM-AREA", "exam": r.get("exam_id"), "year": r.get("exam_year"),
        })
    return rows


# Fact sheets moved from docs/ to data/. Result files written before the move store the old paths.
LEGACY_DATA_PREFIXES = {"docs/revisao_juridica/": "data/legal_facts/", "docs/controles/": "data/controls/"}


def resolve_data_path(path: str | Path) -> Path:
    """Return ``path`` if it exists, else its new location when it uses a legacy data prefix."""
    p = Path(path)
    if p.exists():
        return p
    text = str(path).replace("\\", "/")
    for old, new in LEGACY_DATA_PREFIXES.items():
        if text.startswith(old) and Path(new + text[len(old):]).exists():
            return Path(new + text[len(old):])
    return p


def load_fact_csv(path: str | Path, *, drop=("remover",)) -> dict[str, dict]:
    """``{relation: {"template": str, "pairs": [(subject, answer), ...]}}`` from the review sheet.

    Rows whose ``status`` is in ``drop`` are skipped; ``resposta_completa`` is the answer
    (the probe uses its first token).  All rows of a relation must share one template.
    """
    out: dict[str, dict] = {}
    with open(resolve_data_path(path), encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["status"].strip().lower() in drop:
                continue
            rel = out.setdefault(r["relacao"], {"template": r["modelo_de_frase"], "pairs": []})
            if rel["template"] != r["modelo_de_frase"]:
                raise ValueError(f"relation {r['relacao']!r} has more than one template")
            if "{}" not in rel["template"]:
                raise ValueError(f"template of {r['relacao']!r} has no '{{}}' placeholder")
            rel["pairs"].append((r["sujeito"], r["resposta_completa"]))
    return out


# Few-shot prefixes (constant; their subjects are not in the sheet) so the model's next token is the
# answer, not a format guess.  Kept in code so the review sheet stays a plain one-sentence template.
FEW_SHOT = {
    "siglas": ("Em direito brasileiro, a sigla MPU significa Ministério Público da União.\n"
               "Em direito brasileiro, a sigla TJDFT significa Tribunal de Justiça do Distrito Federal e dos Territórios.\n"),
    # non-legal Portuguese control for the sigla task (data/controls): same language and format, no law
    "siglas_gerais": ("Em português, a sigla IFPE significa Instituto Federal de Pernambuco.\n"
                      "Em português, a sigla FINEP significa Financiadora de Estudos e Projetos.\n"),
    "sedes_trf": ("A sede do Supremo Tribunal Federal fica em Brasília.\n"
                  "A sede do Superior Tribunal de Justiça fica em Brasília.\n"),
}


def register_csv_domains(csv_path, specs, facts: dict, templates: dict) -> None:
    """Add ``name=relation`` domains from the review sheet to ``FACTS`` / ``TEMPLATES`` (few-shot prefixed)."""
    sheet = load_fact_csv(csv_path)
    for spec in specs:
        name, rel = spec.split("=", 1)
        facts[name] = sheet[rel]["pairs"]
        templates[name] = FEW_SHOT.get(rel, "") + sheet[rel]["template"]


LAW_SUBJECTS = ("professional_law", "jurisprudence", "international_law")


def load_mmmlu(path: str | Path, *, law_subjects=LAW_SUBJECTS, n_law: int = 500, n_other: int = 500,
               seed: int = 42) -> list[dict]:
    """Sample law and non-law questions from the Portuguese MMMLU csv (``openai/MMMLU``, ``mmlu_PT-BR.csv``).

    Rows follow the MMLU layout (``question``, ``choices``, ``answer`` index) plus ``area`` ("law" or
    "other") and ``subject``.  The first ``n_law`` rows are law, the rest are not, so per-group results
    can be split by position.  The legal subjects are mostly US law translated to Portuguese.
    """
    law, other = [], []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["Answer"].strip() not in LETTERS:
                continue
            row = {"question": r["Question"], "choices": [r["A"], r["B"], r["C"], r["D"]],
                   "answer": LETTERS.index(r["Answer"].strip()), "subject": r["Subject"]}
            (law if r["Subject"] in law_subjects else other).append(row)
    rng = random.Random(seed)
    rng.shuffle(law)
    rng.shuffle(other)
    return [{**r, "area": "law"} for r in law[:n_law]] + [{**r, "area": "other"} for r in other[:n_other]]
