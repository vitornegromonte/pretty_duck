"""MMLU CSV loader: tarball extraction, split aliases, malformed rows."""

import io
import tarfile

import pytest

from compress.mmlu import load_mmlu

ROWS = [
    ["w1 w2?", "w3", "w4", "w5", "w1", "B"],
    ["w2 w3?", "w1", "w2", "w4", "w5", "D"],
]


def _tar(path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


def _csv(rows) -> bytes:
    # MMLU CSVs are headerless: question, A, B, C, D, answer.
    return ("\n".join(",".join(f'"{c}"' for c in row) for row in rows) + "\n").encode()


def test_loads_rows_from_the_cached_tarball(tmp_path):
    _tar(tmp_path / "data.tar", {"data/test/fake_subject_test.csv": _csv(ROWS)})
    rows = load_mmlu("fake_subject", "test", cache_dir=tmp_path)
    assert [r["answer"] for r in rows] == [1, 3]
    assert rows[0]["question"] == "w1 w2?"
    assert rows[0]["choices"] == ["w3", "w4", "w5", "w1"]
    assert (tmp_path / "data/test/fake_subject_test.csv").exists()


def test_validation_is_an_alias_for_val(tmp_path):
    _tar(tmp_path / "data.tar", {"data/val/fake_subject_val.csv": _csv(ROWS)})
    assert len(load_mmlu("fake_subject", "validation", cache_dir=tmp_path)) == 2


def test_unknown_subject_raises(tmp_path):
    _tar(tmp_path / "data.tar", {"data/test/fake_subject_test.csv": _csv(ROWS)})
    with pytest.raises(KeyError, match="not found"):
        load_mmlu("other_subject", "test", cache_dir=tmp_path)


def test_rejects_path_like_subject(tmp_path):
    with pytest.raises(ValueError, match="invalid MMLU subject"):
        load_mmlu("../escape", "test", cache_dir=tmp_path)


def test_rejects_malformed_rows(tmp_path):
    _tar(tmp_path / "data.tar", {"data/test/fake_subject_test.csv": b'"only", "three"\n'})
    with pytest.raises(ValueError, match="expected 6 columns"):
        load_mmlu("fake_subject", "test", cache_dir=tmp_path)
