import json

import pytest

from compress.oab import load_fact_csv, load_oab


def _row(i, key="A", nullified="False", choices=4, qtype="CIVIL"):
    return {"id": f"x_{i}", "question": f"q{i}", "choices": {"text": [f"c{j}" for j in range(choices)],
            "label": list("ABCD"[:choices])}, "answerKey": key, "nullified": nullified,
            "question_type": qtype, "exam_id": "2010-01", "exam_year": "2010"}


def test_load_oab_converts_to_mmlu_rows_and_drops_bad_ones(tmp_path):
    rows = [_row(0, "C"), _row(1, nullified="True"), _row(2, choices=3), _row(3, qtype=None), _row(4, key="E")]
    path = tmp_path / "o.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf-8")
    got = load_oab(path)
    assert [r["question"] for r in got] == ["q0", "q3"]
    assert got[0]["answer"] == 2 and got[0]["choices"] == ["c0", "c1", "c2", "c3"] and got[0]["area"] == "CIVIL"
    assert got[1]["area"] == "SEM-AREA"
    assert len(load_oab(path, drop_nullified=False)) == 3


def test_load_fact_csv_groups_by_relation_and_honours_status(tmp_path):
    head = "id,relacao,sujeito,resposta_primeira_palavra,resposta_completa,modelo_de_frase,verificacao,status\n"
    body = ("1,siglas,STF,Supremo,Supremo Tribunal Federal,A sigla {} significa,x,rascunho\n"
            "2,siglas,STJ,Superior,Superior Tribunal de Justiça,A sigla {} significa,x,ok\n"
            "3,siglas,XXX,Errado,Errado,A sigla {} significa,x,remover\n"
            "4,sedes,1ª Região,Brasília,Brasília,Sede da {} fica em,x,rascunho\n")
    path = tmp_path / "f.csv"
    path.write_text(head + body, encoding="utf-8")
    got = load_fact_csv(path)
    assert set(got) == {"siglas", "sedes"}
    assert got["siglas"]["pairs"] == [("STF", "Supremo Tribunal Federal"), ("STJ", "Superior Tribunal de Justiça")]
    assert got["sedes"]["template"] == "Sede da {} fica em"
    bad = head + "1,r,a,b,b,sem marcador,x,rascunho\n"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="placeholder"):
        load_fact_csv(path)
    mixed = head + "1,r,a,b,b,T1 {},x,rascunho\n2,r,c,d,d,T2 {},x,rascunho\n"
    path.write_text(mixed, encoding="utf-8")
    with pytest.raises(ValueError, match="more than one template"):
        load_fact_csv(path)


def test_register_csv_domains_adds_few_shot_prefixed_templates(tmp_path):
    from compress.oab import FEW_SHOT, register_csv_domains
    head = "id,relacao,sujeito,resposta_primeira_palavra,resposta_completa,modelo_de_frase,verificacao,status\n"
    path = tmp_path / "f.csv"
    path.write_text(head + "1,siglas,STF,Supremo,Supremo Tribunal Federal,A sigla {} significa,x,rascunho\n"
                    "2,outra,A,B,B,Frase {} aqui,x,rascunho\n", encoding="utf-8")
    facts, templates = {}, {}
    register_csv_domains(path, ["direito_siglas=siglas", "extra=outra"], facts, templates)
    assert facts == {"direito_siglas": [("STF", "Supremo Tribunal Federal")], "extra": [("A", "B")]}
    assert templates["direito_siglas"] == FEW_SHOT["siglas"] + "A sigla {} significa"
    assert templates["extra"] == "Frase {} aqui"  # no prefix defined for this relation
