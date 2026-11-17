# ADR 0001: Domain-aware heterogeneous quantization — what was tested and what we decided

- **Status:** Accepted (decisions D1–D10 stand until new evidence overturns them; section 7 lists what is still open)
- **Date:** 2026-11-10 (updated 2026-11-17 with E15 to E23)
- **Branch:** `feature/domain-compression`
- **Related:** `docs/method.pdf` (method and Appendix B), `README.md` ("Findings so far"), `REQ.md` / `TASK.md` (original spec)

## 1. Context

The project builds on the Persona code (activation-contrast masks that zero weights to shift personality). The goal
(`TASK.md`) is a **domain-aware, heterogeneous quantization** of a language model: preserve a target domain's
behaviour while compressing harder in components that tolerate it, with real stored bytes.

The working hypothesis, **H**, was: *a domain-specialization score (activation contrast between a target domain and
others) identifies components that matter for that domain, and can steer per-block bit allocation.*

Constraints that shaped every experiment: no gradients for the initial sensitivity, no fine-tuning, no auxiliary
network, forward-only interventions with weights restored afterwards, sizes from exact byte formulas (never nominal
bits), results labelled as measured / estimated / simulated.

Environment: SmolLM-135M on CPU for statistics, plans and mean ablation; Qwen3.5-0.8B (24 layers, 18 linear-attention
and 6 full-attention) on an RTX 3090 (`maidai`) for the persona, circuit and allocation experiments. Seed 42 throughout.

## 2. Decisions

| # | Decision | Evidence |
|---|---|---|
| D1 | **Do not use a domain-contrast score (`C_g`, TARC) as an allocation signal.** Keep the code, label it a negative result. | E1–E3, E8, E13 |
| D2 | **Keep specialization and quantization sensitivity as two separate scores** and measure whether the first adds predictive information about the second before combining them. | E7, E11, E12 |
| D3 | **Rank layers by measured per-layer ΔL, not by the diagonal `D`.** `D` remains useful at column level (dose) and within a layer. | E8, E12, E13 |
| D4 | **Causal claims need minimal pairs plus controls**: dose-matched random sets, a lexical (shuffled-prompt) control, an unrelated-prompt control, and held-out items. Mean ablation alone is not accepted evidence. | E4, E5, E6 |
| D5 | **Circuit units are slices of write-projection inputs** (attention heads, MLP neurons), found on one half of the facts and validated on the other. MLP circuits should be built at single-neuron granularity: groups of 64 hid the domain specificity that E14 finds. Attention heads stay head-level and behave as shared infrastructure. | E9, E10, E14 |
| D6 | **Evaluate allocation at equal actual bytes with the real formats** (nf4 baseline, budgets as extra bytes). Simulated per-row int4/3/2 whole-model baselines are too destructive to be informative. | E12, E13 |
| D7 | **Report intervals or random-draw spreads, and mark metric/data limits.** Several results below lack confidence intervals; they are flagged. | all |
| D8 | **Judge protection on group-wise quantization at a bit width where uniform quantization loses accuracy without collapsing, and measure damage by accuracy, not by the NLL of the correct letter** (that NLL fell under quantization, from 5.07 to 3.43, so it is not a damage measure). | E18 |
| D9 | **Letter accuracy of one quantized model is not a reliable damage measure.** Equivalent quantizers differ by up to ±6 OAB points (E22), so every accuracy comparison from a single rounding realization, including E18 to E20, is weaker than its question-level interval suggests. Use continuous metrics (NLL on neutral text, log-probability of held-out facts, and planned KL to the fp16 model and option-permuted letter probabilities) and several realizations (calibration draws) per condition. | E21, E22 |
| D10 | **Improve the quantizer before the protection set.** Clipping alone moves GPTQ int3 from a NLL gap of 0.18 to 0.07 over fp16 (E23), more than any circuit effect measured. | E22, E23 |

## 3. Experiment log

"Verdict" is what the evidence supports for H and for the project, not whether the experiment ran.

### E1 — Block-level domain signal (SmolLM-135M)
- **Setup:** 210 Linear blocks; math (92) and law (194) calibration samples from MMLU *val*; diagonal `D` per format.
- **Result:** rank correlation of math vs law error 0.995 (int8, int4) and 0.996 (nf4); importance 0.996 (per weight 0.994). Per-block `D_math/D_law` (nf4): 0.49–2.8, median 1.02.
- **Verdict:** raw block sums are dominated by block size; the domains look almost identical. This motivated TARC (E2).
- **Artifacts:** `scripts/make_mmlu_calib.py`, `outputs/errors_math_law.json` (not tracked).

### E2 — Target-anchored contrast and row-group units (SmolLM-135M)
- **Setup:** five domains (math, law, bio, cs, hum) each split in two halves, ≈40 lines, max length 128. TARC `S_T = R_T − median_{d≠T} R_d` over rank-normalized importance. Units: whole layers (210) and 64-row groups (2,430). Size budget 40% of the way from all-int4 to fp16 (≈117 of 212 MB).
- **Result:** `S_T` std 0.023 (range −0.14…0.13, split-half 0.91) per block and 0.030 (−0.25…0.19, split-half 0.93) per row group. Units changed vs the error-only plan, λ = 1/10/100/1000: 1/1/7/39 of 210 and 0/13/33/277 of 2,430, with estimated target error −1.84/−1.84/+10.60/+87.37% and +0.00/+0.78/+2.06/+60.03%. Math-target vs law-target error-only plans differ in 2/210 and 22/2,430 units.
- **Verdict:** reproducible but small; plans barely depend on the target. Supports D1.
- **Artifacts:** `compress/allocate.py` (`target_anchored_contrast`), `compress/quant.py` (row groups), `compress/serialize.py`, `tests/test_allocate.py`.

### E3 — Per-input-channel signal (SmolLM-135M)
- **Result:** per-channel `S_T` std 0.198, 20.5% of channels with |S_T| > 0.25, split-half 0.84; `V_math / median V_others` percentiles 0.58/0.88/1.04/1.25/2.17; 13.1% of channels exceed 1.5 and carry 16.3% of math's second-moment mass.
- **Verdict:** the signal is wider and reproducible at channel level, which motivated causal tests (E4–E7).

### E4 — Mean ablation with a dose control (SmolLM-135M)
- **Setup:** replace the selected input channels with the other domains' mean; 30 held-out MMLU *test* lines per domain; 5 draws per random control; k = 1%, 2%.
- **Result (selectivity, nats):**

  | anchor | k | tarc | V-matched random | energy-matched random |
  |---|---|---|---|---|
  | math | 1% | +0.100 | +0.007 | +0.091 |
  | math | 2% | +0.156 | +0.022 | +0.144 |
  | law | 1% | +0.087 | +0.005 | +0.079 |
  | law | 2% | +0.135 | −0.061 | +0.141 |

  The gap tarc − energy-random is +0.009, +0.012, +0.008, −0.006, inside the control spread.
- **Verdict:** the selectivity is explained by perturbation dose. The top-`A_T` ("importance") set is catastrophic (~6 nats) and not selective. Established D4.
- **Artifacts:** `compress/patching.py`, `tests/test_patching.py`, `outputs/probe2_{math,law}.json`.

### E5 — Does a persona prompt move the answer margin? (wealth persona, 200 questions)
- **Metric:** `margin = logp(matching) − logp(other)`, paired seek − reject; original Persona prompts.
- **Result:** Qwen3.5-0.8B +2.82 [2.31, 3.34], shuffled-word prompts +1.95 [1.53, 2.36], unrelated prompts +0.91 [0.51, 1.31]. SmolLM-135M: real +0.83, shuffled +0.97, unrelated +0.23. Choices stay near chance (0.52–0.55) under every prompt.
- **Verdict:** about two thirds of Qwen's shift is lexical; SmolLM responds only to word overlap and is excluded from persona work. The persona-specific part is +0.85 [0.45, 1.28] nats.
- **Artifacts:** `scripts/persona_gate.py`, `scripts/make_persona_calib.py`.
- **Note:** `wealth_test_infer.csv` has 200 questions (211 physical lines because some fields span lines).

### E6 — Minimal-pair patching of the persona margin (Qwen3.5-0.8B)
- **Setup:** clean (seek) vs corrupt (reject) prompts, Linear inputs patched over the aligned suffix; k = 5% of channels (13,782); `zdiff` (standardized mean difference), `tarc`, `importance`, 5 V-matched random draws; three prompt pairs on the same questions.
- **Result:** sufficiency (real/shuffled/unrelated): zdiff 0.84/0.63/0.23, tarc 0.64/0.43/0.75, importance 0.15/0.16/0.51, random 0.18/0.18/0.17. Necessity: zdiff 0.80 [0.72, 0.89]/0.67/0.28, random 0.15. Paired real − shuffled in nats restored: zdiff +1.11 [0.85, 1.37], tarc +0.94, importance +0.09, random +0.15. Single layers 11–19 restore 13–43% (layer 12: 0.43); layer 23 is negative. Patching all Linear inputs reproduces the clean run exactly (1.000).
- **Verdict:** `zdiff` selects channels that are sufficient and necessary for the prompt-induced shift, specific to the persona content, partly beyond word overlap. `tarc` is not specific (it restores the unrelated pair). One trait, one model, one seed.
- **Artifacts:** `compress/pair_patching.py`, `scripts/persona_patch.py`, `tests/test_pair_patching.py`, `outputs/pp_{fwd,rev}.json`.

### E7 — Quantizing the persona channels (Qwen3.5-0.8B)
- **Setup:** fake-quantize only the selected weight columns (per-row scales over the selected columns), int4/3/2 (below 4 = intervention strength only), k = 5%, 20%; retention of the 2.82-nat shift; diagonal dose reported.
- **Result:** retention ≈ 1.0 for zdiff and random at int4 and int3 (doses within ~10%); int2: 0.73 vs 0.89 (5%), 0.21 vs 0.41 (20%) with wide random spreads. All-columns int4: 0.82 [0.65, 1.05].
- **Verdict:** no quantization signal at 3–4 bits. Informs D2.
- **Artifacts:** `compress/quant_probe.py`, `scripts/persona_quant.py`.

### E8 — Measured per-layer sensitivity vs the diagonal `D` (Qwen3.5-0.8B, 186 layers)
- **Setup:** quantize one layer at a time (int8, int4, nf4), ΔNLL on 64 held-out lines for each of math/law/bio/hum, paired; then 48 multi-layer plans (4, 12, 40 layers; uniform int4 and mixed) on the same and on a disjoint split.
- **Result:** Spearman(`D`, ΔL) between −0.07 and +0.05 per domain and format (−0.28…+0.13 above the noise floor); top-20 overlap 3/20 (chance ≈2). Damage sits in residual-writing projections that `D` ranks low (`o_proj` 0.23, `down_proj` 0.12, `out_proj` 0.13 percentile) while `in_proj_qkv` (0.95) and `q_proj` (0.83) hurt least. The 10 worst layers carry 52% of the positive int4 damage; only 16/44/30% of cells (int8/int4/nf4) exceed 2 standard errors. math–law ΔL correlation across layers 0.58 (int4). Additivity: Spearman 0.99 (same text) and 0.85–0.97 (held-out); median measured/predicted 1.02, 1.05, 1.10 for 4, 12, 40 layers.
- **Verdict:** `D` is unreliable across layers and inverted by layer type; the measured table is cheap, roughly additive, and mostly domain-shared. Established D3.
- **Artifacts:** `compress/sensitivity.py`, `scripts/sensitivity.py`, `outputs/sensitivity.json`, `outputs/errors_dom.json`.

### E9 — Circuit discovery on factual recall (Qwen3.5-0.8B)
- **Setup:** templated domains (capitals, element symbols, English→French) as entity-swap minimal pairs of equal token length; metric `LD = logit[a_c] − logit[a_k]` at the last position; only facts the model answers correctly (54 of 88, 27 of 44, 52 of 54), split in two halves. 1,680 units: 288 linear-attention heads (128 ch), 48 full-attention heads (256 ch), 1,344 MLP groups (64 neurons). Single-unit scan on half A, circuits validated on half B against class-matched random sets (10 draws).
- **Result:** top-16 units (≈1%) restore 0.98/1.01/0.98 and destroy the same fraction of held-out `LD` (random −0.03/+0.04/+0.02); best single unit 0.62/0.34/0.11, with 33/55/44 units above 0.01. Circuits sit in layer 0 (3 units), layers 14–21 and a few at 22–23. All units give exactly 1.00 (the read position's residual is the embedding plus write-projection outputs).
- **Verdict:** factual recall is carried by sparse, joint, cross-layer circuits that generalize to held-out facts.
- **Artifacts:** `compress/facts.py`, `compress/circuit.py`, `scripts/circuit_probe.py`, `tests/test_circuit.py`, `outputs/circuit_probe.json`.

### E10 — Entanglement: shared core and periphery
- **Result:** top-64 circuits give necessity 0.99–1.00 on every domain (French→capitals 0.88; random −0.13…+0.16); Jaccard 0.36 (capitals–elements), 0.20 (capitals–French), 0.26 (elements–French). The shared core (units in the top 32 of ≥2 domains) has 18 units (9 in all three) and alone gives sufficiency/necessity 0.72/0.75, 1.03/1.03, 0.93/0.91. Own/other-domain necessity: full circuits at k=4 0.93/0.20 (capitals), 0.81/0.41 (elements), 0.54/0.34 (French), converging to ≈1/≈1 by k=64; with the core removed at k=16 0.91/0.11, 0.33/0.09, 0.48/0.07; specific-only units at k=64 0.55/0.13, 0.46/0.09, 0.43/0.07.
- **Verdict:** a small shared core plus a domain-specific periphery that is clean for capitals, moderate for French and thin for elements. No confidence intervals; elements has 14 validation pairs.
- **Artifacts:** `scripts/entanglement.py`, `outputs/entanglement.json`.

### E11 — Quantizing circuits at equal dose
- **Result (loss of mean answer log-prob, circuit vs dose-matched random):** capitals int4 k=16 +0.29 (acc −0.30) vs +0.00 (doses 0.52/0.46); capitals int3 k=16 +0.11 vs +0.06; elements int2 k=16 +0.87 vs +0.05 and k=32 +9.68 (acc −1.00) vs +0.07 (doses 8.06/8.02); French int2 k=32 +5.02 vs +0.85. Elements and French are almost unaffected at int3/int4. Dose cannot be matched for k ≥ 64. Circuits also raise neutral-text NLL far more than equal-dose sets (elements int2 k=32: +4.58 vs +0.20). No diagonal dominance across domains. Joint/sum of single-unit damage at int2, k=16: 2.54/2.45, 0.87/0.14, 1.43/0.91.
- **Verdict:** circuits are specifically fragile at 2 bits and are generally critical, not recall-specific; joint effects are not predictable from singles at 2 bits.

### E12 — Protection inside a whole-model quantization (per-row int, Qwen3.5-0.8B)
- **Result:** uniform per-row int4 already collapses capitals (acc 1.00→0.00) and French (−0.77), NLL +0.69; int3 +7.33, int2 +12.49. Share of NLL damage recovered by protecting the same number of unit columns at fp16: core (0.63% of weights) 21/26/2% (int4/3/2) vs random 3/3/0% vs dose-top 24/16/−1%; union of circuits (1.90%) 32/33/−2% vs random 10/10/0% vs dose-top 47/34/1%. Capitals accuracy is zero in every condition at int4.
- **Verdict:** circuit protection beats random but not a dose-based choice. The crude quantizer makes this baseline uninformative for accuracy (D6).
- **Artifacts:** `compress/protect.py`, `scripts/protect.py`, `tests/test_protect.py`, `outputs/protect.json`.

### E13 — End-to-end allocation at equal bytes (nf4 baseline, Qwen3.5-0.8B)
- **Setup:** uniform nf4 over 186 layers (264.4 MB; NLL cost +0.098, fact log-prob cost +0.127; uniform int8 at +89% bytes ≈ 0). Extra-byte budgets +1/2/5/10%. Methods: layers by `D`, layers by the measured table, fp16-protected units by circuit sufficiency, by nf4 dose per byte, and at random (5 draws). Exact bytes from `plan_nbytes`. Scored on held-out fact prompts and 32 held-out lines per text domain, disjoint from the table's lines.
- **Result (share of nf4's NLL damage recovered, +1/2/5/10%):** layers by `D` −1/+1/+3/+5%; layers measured 24/32/46/61%; units circuit 11/19/22/24%; units dose 16/22/33/36%; random 0/+1/6/8%. Fact log-prob loss at +5%: D 0.083, measured 0.149, circuit 0.114, dose 0.200, random 0.120 (range 0.061–0.261) vs 0.127 for nf4; most differences are within the random spread (27, 14, 26 prompts per domain).
- **Verdict:** the measured table is the best signal for the metric it was measured on; `D` at layer level is no better than random units; circuit knowledge adds nothing over domain-agnostic signals. The measured allocation optimizes text NLL and does not improve factual recall at ≤ +5%. Its NLL advantage is partly in-distribution.
- **Artifacts:** `scripts/alloc_e2e.py`, `outputs/alloc_e2e.json`.

### E14 — Neuron-level re-test of the entanglement (Qwen3.5-0.8B)
- **Setup:** every MLP neuron (input channel of `down_proj`) of the 19 layers holding the group-level circuits (0–6, 10, 13–23) was scored by batched single-neuron patching (68,096 neurons per domain, 53 minutes on the 3090; the batched scan is checked against the one-unit-at-a-time scan in tests). Heads kept the head-level scores of E9. Entanglement test as in E10 on two families: *mixed* (heads + neurons) and *mlp_only*, with core removal and strictly-specific variants, class-matched random controls (5 draws), held-out pairs (27/14/26).
- **Single neurons are weak but nearly disjoint across domains:** best neuron 0.057 / 0.032 / 0.036 (capitals / elements / French); only 103 / 124 / 137 neurons exceed 0.002. The overlap of those sets is 6 (capitals–elements), 0 (capitals–French) and 1 (elements–French) neurons, against 0.2 expected by chance; top-128 overlap 9 / 1 / 1 (chance 0.24); top-512 overlap 77 / 27 / 26 (chance 3.9). Top neurons sit in layers 2–6 and 14–23 (layer 14 holds 29 of capitals' top 128; French concentrates in layers 18–23).
- **MLP-only circuits are domain-leaning, with clean controls.** Necessity on the own domain / mean of the others, k neurons = k channels: k=512 0.55/0.15 (capitals), 0.58/0.11 (elements), 0.29/0.06 (French); k=2048 0.69/0.27, 0.68/0.19, 0.49/0.18; class-matched random 0.00–0.03. They reach only about half to two thirds of the effect.
- **Mixed circuits stay entangled because heads dominate.** At k=32, 27 of 32 units are heads and necessity is ≈1.0 on every domain; by k=2048 they hold 130–173 heads. With the shared core removed the own/off gap is clear at small k (k=128: 0.54/0.08, 0.40/0.04, 0.29/0.02) but the class-matched random baseline for heads climbs to 0.3–0.7 at k ≥ 512, so those rows are confounded.
- **Verdict:** the group-level entanglement was partly a granularity artifact for MLP content: neuron-level MLP circuits are domain-specific (own ≫ off ≫ random) while attention heads form the shared part. The remaining 30–50% of the effect rides on shared heads. The off-domain effect of MLP circuits (0.06–0.27) is not zero. The "Spearman among active neurons" printed by the script (−0.70…−0.76) is an artifact of conditioning on the union of two nearly disjoint sets and should not be quoted; use the overlap counts.
- **Limits:** no confidence intervals, tiny validation sets (14 elements pairs), greedy ranking ignores interactions, patching rather than quantization, scan restricted to 19 layers.
- **Artifacts:** `compress/circuit.py` (`scan_neurons`), `scripts/neuron_scan.py`, `tests/test_circuit.py`, `outputs/neuron_scan.json`.

### E15 — How much Brazilian law do the models know? (Qwen3.5-0.8B and 4B)
- **Data:** `eduagarcia/oab_exams` (2,210 four-option OAB questions, 2010 onward; no license in its card, check before redistributing), converted to JSONL by `scripts/oab_prepare.py`; a 60-fact draft sheet of cloze facts (49 sigla expansions, 5 LIMPE principles, 6 TRF seats) in `docs/revisao_juridica/`, not yet reviewed by a jurist.
- **Result:** OAB accuracy on 800 sampled questions 36.9% (0.8B) and 42.4% (4B), chance 25%. Cloze facts known with no examples in the prompt: 1 of 60 (0.8B) and 9 of 60 (4B). With a two-example prefix (examples not in the sheet): siglas 15 (0.8B) and 26 (4B) of the 31 that survive the tokenization and equal-length filter; LIMPE 0 of 5 and TRF seats 1 of 6 for both.
- **Verdict:** the 4B is the smallest model with a usable law domain, with 26 siglas (13 pairs per half). The sheet needs a jurist's review and more relations (instructions in `docs/revisao_juridica/LEIAME.md`). Siglas measure memorized official names, not legal reasoning.
- **Artifacts:** `compress/oab.py`, `scripts/br_law_knowledge.py`, `tests/test_oab.py`, `outputs/br_law_*.json`.

### E16 — Circuits on Qwen3.5-4B, including Brazilian-law siglas
- **Setup:** as E9 with 5,504 units (768 linear-attention heads of 128 channels, 128 full-attention heads of 256, 4,608 MLP groups of 64) and four domains: capitals (76 known facts, 38/38 pairs), elements (27, 13/14), French (54, 27/27) and `direito_siglas` (26, 13/13).
- **Result (held-out sufficiency, random sets about 0):** top-8 units 0.91 / 0.72 / 0.55 / 0.74 (capitals / elements / French / law), top-128 0.98 / 0.99 / 0.99 / 0.98. Best single unit 0.40 / 0.34 / 0.19 / 0.30. The top units are mostly late-layer attention heads (layers 23 to 31) plus early linear-attention heads; for law, MLP groups enter from k=16 (3 at 16, 12 at 32, 29 at 64, 67 at 128).
- **Verdict:** sparse, joint, held-out-validated circuits exist for the Brazilian-law siglas, but they are less compact than in the 0.8B (k=16 gives 0.72 to 0.94 against 0.98 to 1.01). Thin evidence for law: 13 validation pairs and siglas only.
- **Note:** the first 4B run ran out of GPU memory at the fourth domain (another job shares the GPU). Logits are now computed for the last position only and cached activations are offloaded to the CPU; the three finished domains were kept, the law domain was rerun alone and merged.
- **Artifacts:** `scripts/circuit_probe.py` (`--facts-csv`, `--csv-domain`, `--no-quant`), `compress/circuit.py` (`Scanner.offload`), `outputs/circuit_probe_4b_all.json`.

### E17 — Entanglement on the 4B (four domains)
- **Result:** shared core of 15 units (top 32 in at least two domains), only 1 in all four; alone it gives necessity 0.75 / 0.68 / 0.66 / 0.78. Own-domain / mean of the others: full circuits at k=32 0.96/0.23, 0.87/0.63, 0.87/0.56, 0.87/0.37; with the core removed at k=64 0.76/0.11, 0.72/0.21, 0.30/0.12, 0.62/0.14; strictly specific units at k=64 0.74/0.08, 0.56/0.11, 0.23/0.08, 0.51/0.06 (random 0.00 to 0.08).
- **Verdict:** less entangled than the 0.8B at the same k (there off-domain was about 0.9 at k=32), and law has its own periphery. No confidence intervals; the core threshold is my choice.
- **Artifacts:** `scripts/entanglement.py`, `outputs/entanglement_4b.json`.

### E18 — Protecting the law circuit under group-wise quantization (Qwen3.5-4B)
- **Setup:** every Linear of the text tower fake-quantized with one scale per 64 columns (`fake_quant_group`), unit columns protected at fp16, 600 OAB questions identical across conditions (per-question results stored for paired comparison), 3 random draws, protected sets of 32 and 128 units (0.30% and 1.08% of the weights).
- **Result:** fp16 43.8%; uniform int4 35.0%, int3 25.2%, int2 24.0% (chance), so only int4 is informative. At int4, gain in accuracy points over uniform (95% interval over questions): circuit +3.2 [+1.5, +4.8] at 32 units and +2.3 [+0.8, +4.0] at 128; highest-dose units +0.2 [−0.8, +1.3] and +1.7 [+0.5, +3.0]; highest-activation columns (AWQ-like) +2.7 [0.0, +5.2] and −0.8 [−3.0, +1.2]. Against the random sets (35.0 to 36.7%) the circuit gains +2.6 [+1.3, +4.1] and +1.7 [+0.3, +3.1]. At int3 and int2 no method beats random; the sigla log-probability is unaffected at int4.
- **Verdict:** a small but consistent compression-side signal: protecting 0.3% of the weights recovers about a third of the OAB loss, and the circuit is the only method with a significant gain at both sizes. It is not shown to be law-specific (the circuit was found on siglas, the benchmark is OAB, and circuits were generally critical in E11). Intervals cover questions only, not the choice of units; one seed.
- **Artifacts:** `scripts/protect_law.py`, `compress/protect.py` (`group`), `compress/quant_probe.py` (`fake_quant_group`), `outputs/protect_law_4b.json`.

### E19 — Control: protect the circuits of other domains instead (Qwen3.5-4B, int4, group 64)
- **Setup:** as E18 (same 600 OAB questions, verified identical), protecting the capitals, elements and French circuits at the same sizes (32 and 128 units, 0.28% to 1.00% of the weights).
- **Result (OAB accuracy; uniform int4 35.0%, law circuit 38.2% and 37.3%):** capitals 34.8% / 35.8%, elements 36.7% / 32.8%, French 34.0% / 34.2% (k=32 / k=128). Against uniform: -0.2, +1.7 [+0.7, +2.8], -1.0 at k=32 and +0.8, -2.2 [-3.8, -0.5], -0.8 at k=128. The law circuit beats each control by 1.5 to 4.5 points (the interval excludes zero in 5 of 6 comparisons; capitals at k=128 is -1.5 [-3.2, +0.2] against it) and the mean of the three by +3.0 [+1.6, +4.6] (k=32) and +3.1 [+1.4, +4.8] (k=128).
- **Verdict:** protecting just any circuit does not help; the gain of E18 belongs to the law circuit. **Confound:** the law prompts are in Portuguese and the OAB is in Portuguese, while all three controls use English prompts, so the gain may reflect Portuguese processing rather than law. The deciding control is a circuit for non-legal facts with Portuguese prompts. Intervals cover questions only, one seed.
- **Artifacts:** `scripts/protect_law.py --controls`, `outputs/protect_law_4b_controls.json`.

### E20 — Portuguese non-legal control: general siglas (Qwen3.5-4B, int4, group 64)
- **Setup:** 54 non-legal Brazilian and international sigla expansions (universities, research bodies, companies, UN agencies) in `docs/controles/fatos_siglas_gerais_ptbr.csv`, same Portuguese template and few-shot format as the legal siglas (prefix examples IFPE and FINEP, not in the list). The 4B knows 29 of them (14/15 pairs); its circuit has held-out sufficiency 0.70 (k=8), 0.83 (k=32), 0.96 (k=128) against about 0 for random sets, with 19 heads and 13 MLP groups at k=32. Its units were then protected under the same int4 setup and 600 OAB questions as E18.
- **Result:** OAB accuracy 35.2% (k=32) and 32.3% (k=128), against 35.0% for uniform int4. Gain over uniform +0.2 [-1.2, +1.7] and -2.7 [-4.3, -1.0]; over the random sets -0.4 [-1.8, +0.9] and -3.3 [-4.9, -1.8]. The law circuit beats it by +3.0 [+1.3, +4.7] and +5.0 [+3.0, +7.0] points.
- **Verdict:** same language and task format but non-legal content does not give the gain, so the E18 gain is not explained by Portuguese or by the sigla format. Among the eight protected sets of each size (law, four control circuits, three random draws) the law circuit is the best at both sizes (+3.2 and +2.3 points over uniform; the best of the others are +1.7 and +1.7), and some circuits hurt (down to -2.7), so which units are protected matters. With eight sets the chance of the law circuit being the best by luck would be 1 in 8 per size, so this is supportive, not conclusive. Intervals cover questions only; one seed.
- **Still open:** a non-law Portuguese benchmark (does the law circuit help only law?), another legal task (the reviewed sheet), more random draws and seeds, and a larger model.
- **Artifacts:** `docs/controles/fatos_siglas_gerais_ptbr.csv`, `compress/oab.py` (`FEW_SHOT["siglas_gerais"]`), `outputs/circuit_probe_4b_gerais.json`, `outputs/protect_law_4b_gerais.json`.

### E21 — Does the law circuit help only law? Portuguese MMMLU (Qwen3.5-4B, int4, group 64, RTN)
- **Setup:** 500 law questions (professional law, jurisprudence, international law; mostly US law translated) and 500 non-law questions from MMMLU PT-BR, one seeded sample for every condition; protected sets of 32 and 128 units (0.30% and 1.08%), 6 random draws per size, the general-siglas circuit as control; paired bootstrap over questions.
- **Result:** fp16 41.0% law / 54.2% other; uniform int4 39.0% / 53.6% (only 2.0 and 0.6 points of headroom). Gain over uniform in points (law / other): circuit k=128 +1.8 [0.0, +3.6] / +1.2 [−0.2, +2.6]; circuit k=32 +0.8 / +0.8; random k=128 (mean of 6) +0.7 / +0.9; random k=32 0.0 / −0.4; dose k=128 +0.6 / +0.8; general-siglas circuit k=128 −1.8 / −2.6; AWQ-like k=128 −3.2 / −3.6.
- **Verdict:** no law specificity: the circuit helps law and non-law questions alike, and random sets give the same gain at k=128. Together with E22 this weakens E18 to E20; the headroom is too small for any set to show a clear effect.
- **Artifacts:** `compress/oab.py` (`load_mmmlu`), `scripts/protect_law.py --benchmark mmmlu`, `outputs/protect_mmmlu_4b.json`.

### E22 — Real baselines: GPTQ, the authors' GPTQ and rounding noise (Qwen3.5-4B, int4, group 64, 600 OAB questions)
- **Setup:** `compress/gptq.py` (sequential GPTQ on the same symmetric group grid, calibration of 160 lines from five domains), compared with RTN and with the authors' code (IST-DASLab/gptq, not vendored) run on the same grid and Hessians; protection sets with equal weight budgets under GPTQ, plus a global outlier baseline (`outlier_all`: columns of any layer by dose per weight).
- **Validation:** on test matrices the output error of our GPTQ matches the reference (16.88 vs 16.85; 236.6 vs 235.8; RTN 32.8 and 493). End to end: NLL on 48 neutral lines fp16 2.331, RTN 2.394, our GPTQ 2.362, reference 2.352.
- **Noise:** OAB accuracy of fp16 43.8%; RTN 35.0%; reference GPTQ 34.7%; our GPTQ 49.5% on the 3080 Ti and 40.7% on the 3090 with identical code and calibration. Quantizers of the same quality differ by about ±6 points, a spread our question-level intervals never covered.
- **Protection under GPTQ (one realization, 3080 Ti; k=32 / k=128):** uniform 49.5; circuit 47.8 / 46.7; dose 49.2 / 53.2; AWQ-like 48.5 / 38.7; outlier_all 40.5 / 42.7; random 46.8 to 49.8 / 36.0 to 40.8. Circuit columns hardly overlap the outlier columns (1 of 4,224 and 19 of 15,040) or the AWQ columns (83 and 772).
- **Verdict:** under GPTQ protection gives no demonstrated gain: the circuit is indistinguishable from uniform, and the larger differences (outlier_all −7 to −9, random −9 to −13 at k=128, dose +3.7) sit inside the realization noise. A synthetic test (massive-activation channels) shows protecting high-dose columns lowers layer error 20x and that neither damping nor error-sink handling (`sink=True`) explains a loss, so the outlier_all drop is not mechanistically explained. Inconclusive; replicates with NLL pending (suite `protect`).
- **Artifacts:** `compress/gptq.py`, `scripts/baselines.py`, `tests/test_gptq.py`, `outputs/baselines_4b.json`, `outputs/gptq_ref_4b.json`.

### E23 — The quantizer grid at int3 (Qwen3.5-4B, group 64; partial, one realization)
- **Setup:** `scripts/quant_suite.py --suites grid`, GPTQ with clipping (per group and row, the range shrink with least rounding error), asymmetric grid and act-order; metrics NLL on neutral text, log-probability of held-out sigla facts and OAB accuracy.
- **Result (neutral NLL / law log-probability / OAB; fp16 2.331 / about −0.17 / 43.8%):** RTN 2.883 / −1.66 / 25.2%; GPTQ 2.513 / −1.39 / 36.3%; GPTQ+clip 2.397 / −0.63 / 43.3%; +asym 2.432 / −0.60 / 33.8%; clip+asym 2.393 / −0.31 / 34.8%; +act-order 2.491 / −1.34 / 34.0%; clip+asym+act-order 2.387 / −0.50 / 27.2%.
- **Verdict:** clipping is the main lever: it closes about 64% of the NLL gap GPTQ leaves to fp16 (0.182 to 0.066); asymmetry and act-order add little in NLL (asymmetry helps the sigla log-probability). OAB accuracy does not follow NLL (27.2% for the best NLL) and is dominated by noise (D9). Remaining suites (int4, group 32, domain calibration, protection with replicates, soft weights, mixed bits, low rank) are running or queued.
- **Artifacts:** `scripts/quant_suite.py`, `outputs/suiteA.json`.

## 4. Consequences

**What the project can now claim (measured, with the stated caveats):**
- Causal channel sets for a behaviour can be found by minimal-pair patching and validated against dose-matched and lexical controls (E6).
- Factual recall sits in sparse cross-layer circuits (~1% of units). At neuron level the MLP part is domain-specific and nearly disjoint across domains, while attention heads form a shared part that carries the rest (E9, E10, E14).
- Measured per-layer sensitivity is roughly additive and a much better allocation signal than the diagonal `D` (E8, E13).
- On Qwen3.5-4B, a Brazilian-law (siglas) circuit exists and protecting 0.3% of the weights chosen by it recovers about a third of the OAB accuracy lost to int4 group-wise quantization (E16, E18); circuits of other domains and a non-legal Portuguese sigla circuit give no comparable gain (E19, E20), so under RTN the gain is not explained by language or by the sigla format (one seed, supportive evidence). **E21 and E22 weaken this**: on Portuguese MMMLU the circuit helps law and non-law alike and random sets match it, and under GPTQ protection is indistinguishable from uniform; rounding noise of about ±6 points was never sampled (D9).
- Clipping in the group-wise grid is the largest improvement measured so far on the quantizer side (GPTQ int3 NLL gap to fp16 0.18 to 0.07, E23).

**What it cannot claim:**
- That a domain-specialization score improves bit allocation; every test of that came back null or dose-explained (E2, E4, E7, E11–E13).
- That persona or knowledge sub-networks are separable by unit: the circuits are strongly shared at the granularity tested.
- Any inference speedup: all quantization here is simulated on dequantized weights.
- That the law circuit is law-specific (E21) or that circuit protection adds anything on top of GPTQ (E22); the protection effect of E18 to E20 may be rounding noise.

**Costs:** several experiments need a GPU machine (`maidai`); `outputs/` is gitignored, so results live only in local and remote `outputs/` directories and in `docs/method.pdf`.

## 5. Threats to validity (apply to the whole record)

- **Rounding noise:** letter accuracy of equivalent quantized models varies by about ±6 points (E22); intervals over questions do not cover it, so E18 to E20 are single-realization results.
- One model per experiment, one persona trait, one seed; small fact sets (27, 14, 26 held-out prompts per domain) and no confidence intervals in E10–E13.
- Metric mismatch: the measured table and E13's NLL score optimize text NLL, while the goal is a target-domain behaviour; the table's text matches the evaluation text distribution (different lines).
- Random controls match unit size class and (where stated) dose, but not depth or position.
- Calibration is small MMLU validation text; the knowledge domains are templated prompts, so results may reflect the templates.
- Bit widths below 4 are intervention strengths, not storage formats.

## 6. Corrections to earlier statements

- The persona test set has **200** questions, not 210 (the 211 was the CSV's physical line count).
- Patching *all* write-projection units **does** reproduce the clean run exactly at the read position (an earlier statement said it would not).
- The first mean-ablation read ("tarc is causally selective") was wrong: the energy-matched control reproduced it (E4).
- `S_T`-style rank contrasts were first read as persona detectors; E6 shows `tarc` also restores the unrelated pair.
- Persona calibration text must be read with `csv`; reading the CSVs line by line puts headers and split rows into the calibration text.

## 7. Open questions and next steps (Proposed)

0. **Next controls for E18 to E20:** a non-law Portuguese benchmark (does the law circuit help only law?), a second legal task from the reviewed fact sheet, more random draws and seeds, and a larger model. The Portuguese language control itself is done (E20).
1. **Neuron-level protection at equal bytes** (follow-up to E14, which answered the granularity question): protect the neuron-level MLP circuit (about 2,000 neurons, roughly 1% extra bytes at fp16) plus the shared heads in a whole-model nf4 plan and compare with dose-based and measured-table protection as in E13. The earlier suggestion was motivated by Goodfire's "Neurons as an Interpretable Basis" (https://www.goodfire.com/research/neurons-as-an-interpretable-basis, Qwen3.6-35B-A3B, one model), read through a summarizing tool without its appendix, so its numbers are unverified; it has no quantization or weight-pruning results.
2. **A damage metric with less noise.** Build the measured table with KL divergence to the base model's output distribution, on target-domain facts and on secondary text, and allocate under two constraints (`ΔL_T ≤ ε_T`, `ΔL_S ≤ ε_S`) as proposed in the project notes.
3. **Tail-conditioned statistics.** Test whether a `V` weighted toward high-activation tokens makes `D` rank layers better (Spearman against the measured table).
4. **More statistical power:** larger fact sets, paired intervals for E10–E13.
5. **Replication:** a second trait (honesty needs prompts written first; the `matching` label of `h_test_infer.csv` appears to mark the dishonest answer) and a second model.

8. **New evaluation protocol (D9):** score each question under the four cyclic permutations of the options, use the probability of the correct letter normalized over A to D, KL and flip rate against the fp16 model, perplexity on held-out Portuguese legal text, several calibration draws per condition and a bootstrap over questions and realizations; make Hessian accumulation deterministic (fp64, no TF32).
9. **Pending runs:** the rest of `scripts/quant_suite.py` (cal, protect with replicates, soft, mixed, lowrank) and `scripts/baselines.py` (RTN protection); vendor or pin the reference GPTQ code (`third_party/gptq_ref`, only on `maidai`); export to a real packed format for speed claims.

## 8. Reproduction pointers

- Seeds: 42; calibration from MMLU `val`, evaluation from `test`; Persona calibration via `scripts/make_persona_calib.py`.
- Commands per experiment are listed in `README.md` ("Causal probes") and `docs/method.pdf` (Section 8, Appendix B).
- Scripts for E8–E13 take `--model-path Qwen/Qwen3.5-0.8B` and need `PYTHONPATH=.`; each reads the previous stage's JSON (`stats_dom.pt` → `errors_dom.json` → `sensitivity.json` → `circuit_probe.json` → `entanglement.json` / `protect.json` / `alloc_e2e.json`).
- Tests: `python -m pytest tests/ -q` (offline, CPU).
