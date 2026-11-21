import json
import re
import subprocess

from compress import provenance as prov


def test_provenance_is_json_serializable_and_has_the_fields_a_comparison_needs():
    record = prov.provenance(["python", "scripts/x.py", "--seed", "1"])
    json.dumps(record)
    for key in ("timestamp_utc", "command", "git_commit", "git_dirty", "torch", "transformers", "gpus",
                "default_dtype", "tf32_matmul", "deterministic_algorithms"):
        assert key in record
    assert record["command"] == "python scripts/x.py --seed 1"
    assert record["git_commit"] is None or re.fullmatch(r"[0-9a-f]{40}", record["git_commit"])
    assert isinstance(record["gpus"], list)


def test_provenance_survives_a_missing_git(monkeypatch):
    def broken(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", broken)
    record = prov.provenance([])
    assert record["git_commit"] is None and record["git_dirty"] is None


def test_provenance_stores_no_personal_data():
    text = json.dumps(prov.provenance(["python", "a.py"]))
    for forbidden in ("/home/", "@"):
        assert forbidden not in text
