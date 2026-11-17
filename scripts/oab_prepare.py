"""Convert the downloaded ``eduagarcia/oab_exams`` parquet to JSONL (needs pyarrow, once).

    python scripts/oab_prepare.py outputs/data/oab_exams/train.parquet outputs/data/oab_exams/oab_exams.jsonl
"""

import json
import sys

import pyarrow.parquet as pq

src, dst = sys.argv[1], sys.argv[2]
rows = pq.read_table(src).to_pylist()
with open(dst, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(rows)} rows -> {dst}")
