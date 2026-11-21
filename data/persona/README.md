# Persona data

CSV files from the Persona project ("Your Language Model Secretly Contains Personality Subnetworks", Ye et al., 2026, arXiv:2602.07164), moved here from `Persona/data/`.
They are used only by the persona experiments (`scripts/persona_*.py`, `scripts/make_persona_calib.py`, `experiments/run_smollm.py`).

- The files contain multi-line fields. Read them with `csv`, not line by line.
- The copy had no license file. Check the license of the source before you redistribute these files.
- Used files: `wealth_train.csv`, `wealth_test_infer.csv`, `h_train.csv`, `h_test_infer.csv`. The others are unused.
