"""Brazilian-law data: OAB exam questions and reviewable cloze facts.

``eduagarcia/oab_exams`` (four-option questions from the Bar exams) is converted
once to JSONL by ``scripts/oab_prepare.py``; ``load_oab`` returns rows in the
MMLU layout (``question``, ``choices``, ``answer`` index) plus ``area``, so
``evaluate_mmlu_rows`` scores them unchanged.  ``load_fact_csv`` reads the
cloze-fact sheet a jurist reviews (``docs/revisao_juridica``).
"""

from __future__ import annotations

import csv
import json
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


def load_fact_csv(path: str | Path, *, drop=("remover",)) -> dict[str, dict]:
    """``{relation: {"template": str, "pairs": [(subject, answer), ...]}}`` from the review sheet.

    Rows whose ``status`` is in ``drop`` are skipped; ``resposta_completa`` is the answer
    (the probe uses its first token).  All rows of a relation must share one template.
    """
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8", newline="") as f:
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
    # non-legal Portuguese control for the sigla task (docs/controles): same language and format, no law
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
