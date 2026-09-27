# Implementation plan: Business Entity Resolution cascade (ML Challenge 2026)

## Context

We've settled on this architecture: **Token/Q-gram blocking + Cardinality Node Pruning (CNP)** as the core, with a **GPU cascade** on top. The cascade has an optional dense kNN channel, a GBDT scorer, a multilingual cross-encoder for pairs the GBDT is unsure about, and an LLM judge for the hardest pairs, plus LLM-built normalization dictionaries.

The task: for each Source 1 (S1) entity, find its matching Source 2/Source 3 (S2/S3) records. The test set has 1.73M S1 entities and pools of 4.9M (S2) and 5.1M (S3) records. France is 15% of test S1 and has **no training labels**. Scoring is macro F0.5 per S1 entity; singletons count, and any false match on one scores 0.

Constraints fixed with the user:
- **The deadline is tonight. Don't cut the architecture;** instead, write a valid submission after every tier.
- **Your friend runs everything on the remote box** (RTX 5070 with 12 GB, 32 GB RAM, full internet) from **one end-to-end command**. I write the code and smoke-test it on the laptop (RTX 4060, local `.venv` with torch).
- **Test data use, per README:** learn and tune **only on training data** (a held-out validation split). Test data is inference input only. Per-pool IDF is computed as part of blocking. The only test-derived output is a diagnostic log (e.g. France's predicted match rate); nothing is tuned on it.
- **Models:** MIT/Apache licenses, with a **combined** size under 8B: e5-small 0.12B + xlm-roberta-base 0.28B + Qwen3-4B 4.0B ≈ 4.4B. Self-hosted only; no API calls.
- **Measured data facts the design relies on:**
  - Each S2/S3 record matches **at most one** S1.
  - Country agrees in 100% of true pairs.
  - Every true pair shares at least one token.
  - About 7% of India pairs have the S2/S3 name in an Indic script.
  - Dropping tokens with document frequency of at least 20k keeps 99.5% of pairs.
  - The average match count is 3.46 and the maximum is 11.

## Architecture (stages, in the order run_all.sh runs them)

```text
00 check env ─ 01 load ─ 02 dictionaries+normalize ─┬─ [GPU] 03 dense channel ──────┐
                                                    └─ [CPU] 04a lexical channels ───┴─ 04b RRF+CNP ─► candidate_pairs.tsv
04b ─┬─ [CPU] 05 features ─ 06 GBDT+decide ─► submission v1 ─┐
     └─ [GPU] 07a cross-encoder fine-tune ───────────────────┴─ 07b score uncertain pairs ─► v2 ─ 08 LLM judge ─► v3 ─ 09 finalize
```

The two lanes run in parallel wherever the diagram forks. Every stage writes to `work/<stage>/` with a `_DONE` marker and a hash of its config, so re-running skips finished work and resumes after a crash.

## Code layout (built directly in the submission-zip structure)

```text
student_resource/code/business_entity_resolution/
  README.md  requirements.txt (pinned from the tested local venv)  run_all.sh
  src/er/  config.py io.py splits.py metrics.py normalize.py dictionaries.py blocking.py
           dense.py features.py gbdt.py cross_encoder.py llm.py decide.py submit.py
  scripts/ 00_check_env.py 01_load.py 02_normalize.py 03_dense.py 04_blocking.py 05_features.py
           06_gbdt.py 07_cross_encoder.py 08_llm_judge.py 09_finalize.py download_models.py
  tools/   make_sample.py  package_submission.sh
  tests/   test_metrics.py test_normalize.py test_blocking.py
```

- **Default paths** match the zip: `--data-dir ../../dataset`, `--out-dir ../../output`, `--work-dir ../../work`.
- **Validator reuse:** `submit.py` reuses `validate()` from `utils/validate_submission.py` when it's present, imported via its path. The zip doesn't ship `utils/`, so it falls back to a built-in format check.

## Stage specs

**00 check env.** Checks the Python, torch and CUDA versions, and runs a small bf16 matmul to confirm the 5070 (sm_120) has working kernels. Also checks free RAM and disk, CPU core count, and Hugging Face reachability. `download_models.py` pre-fetches e5-small, xlm-roberta-base and Qwen3-4B-Instruct-2507, falling back to Qwen/Qwen3-4B.

**01 load.** Reads files with pandas `sep="\t", dtype=str, keep_default_na=False`, writes parquet, and encodes IDs as int64. Splits train S1 entities by entity, stratified by country, with seed 2026: **T = 300k** (for training), **Vcal = 50k** (for calibration, gating and threshold tuning) and **Vtest = 50k** (reported score only).

**02 dictionaries + normalize** (`dictionaries.py`, `normalize.py`, `llm.py`):
- **Order matters:** transliterate non-Latin scripts *before* stripping accent marks, because Devanagari vowel signs are combining characters. Then NFKC, casefold, and strip accents.
- **Clean-up:** remove noise prefixes (`--`, `***`, `>>`, `[..]`, `#`, `@`) and domain endings (`.com/.in/.fr`).
- **Separate fields:** legal form (a seed list including LLC/Pvt/Ltd/SARL/SAS/SCI, plus the most frequent name-final tokens in each country's pool, so any country is covered), address number tokens, and 5–6-digit postcodes.
- **Indic→Latin dictionary**, built in this order:
  1. Positional token alignment on training India pairs (support of at least 2).
  2. For tokens still missing, the **LLM (batched, greedy decoding)** transliterates them.
  3. `anyascii` as the last fallback.
- **Abbreviation map:** frequent short tokens per country are sent to the LLM with the country name in the prompt (e.g. `r`→rue, `av`→avenue, `pvt`→private). An expansion is accepted **only if the expanded word already exists in that country's vocabulary**, which guards against hallucination.
- **Also produced:** a consonant skeleton and a space-stripped version of each name, used for q-grams.

**03 dense (GPU, optional channel).**
- Fine-tune multilingual-e5-small on (S1, match) pairs from T for 1 epoch, using in-batch negatives.
- Embed all records in fp16.
- Run an **exact** top-60 search per country with chunked torch matmul. No approximate index is needed, because the largest pool is about 3.6 GB of VRAM.

**04 blocking** (`blocking.py`, numba):
- **Index:** per country, compute IDF over the S2∪S3 pool and build CSR posting lists.
- **Word channel:** name and address tokens, plus combined address-number+street keys. Drop keys whose document frequency is above 0.5% of the pool.
- **Name 3-gram channel:** drop keys above 2% of the pool.
- **Scoring:** IDF-weighted cosine. A parallel accumulator keeps a fixed-size heap per S1 entity and returns the top 60 per channel.
- **RRF fusion:** Σ 1/(60+rank) across channels, then **CNP keeps the top k per S1**.
  - k is picked automatically on Vcal as the smallest value in {10,15,20,30,40,50} whose recall is within 0.001 of recall at 50.
  - The dense channel is included only if it improves Vcal recall at k by at least 0.002.
- **Competition statistics:** block **all 2.2M train S1** entities (and all test S1), keeping only per-record aggregates: the best and second-best S1 score pointing at each S2/S3 record, and how many S1 entities point at it. This way the "competing S1" features look the same in training as at test time.
- **Report:** recall at each k, reduction ratio and pair quality (the doc's PC/RR/PQ), overall and per country.
- **Output:** `candidate_pairs.tsv` = the fused CNP set, which is exactly what the scorers run on.

**05 features** (`features.py`, numba + rapidfuzz `cpdist`, processed per country and per chunk). No country one-hot encoding.
- **Name:**
  - Token Jaccard and IDF-weighted cosine; 3-gram cosine.
  - ratio, token_sort, token_set and Jaro-Winkler on the name with the legal form removed.
  - Abbreviation-subsequence matches and an acronym match.
  - Legal form: equal / different / missing.
  - Whether the name was transliterated.
  - How common the name is: IDF sum, and how many S1 records share the same core name.
- **Address:**
  - Jaccard, IDF-weighted cosine, token_set.
  - Count of shared numbers, and a **number-conflict flag** (both have numbers, none shared).
  - Postcode equal.
  - Missing flag.
- **Graph:**
  - Per-channel scores and ranks, the fused RRF score and rank, and the gap to this S1's best candidate.
  - The **reverse rank** (this S1's position among the S1 entities pointing at the record) and the gap to that record's best score, both from the stage-04 aggregates.
  - An S3 flag.

**06 GBDT + decide → v1** (`gbdt.py`, `decide.py`):
- **Model:** XGBoost `hist` on `cuda`, depth 8, eta 0.05, early stopping on Vcal. Also a leave-one-country-out diagnostic: train on US and test on India, then the reverse.
- **Decision** (the same function is used for every tier):
  1. A per-tier logistic-regression stacker calibrates the scores on Vcal.
  2. **Each S2/S3 record keeps only its highest-probability S1.**
  3. **Per-entity expected-F0.5 decoding:** sort candidates by probability, pick the prefix size m (m = 0 means "no match") that maximizes expected F0.5, and apply a minimum-probability floor τ tuned on Vcal.
- Writes `output/v1/`, validates it, and records Vtest F0.5 per country.

**07 cross-encoder → v2** (`cross_encoder.py`):
- **Model:** xlm-roberta-base on the pair text "name | address ‖ name | address", max_len 128, bf16, lr 2e-5.
- **Training:** 1 epoch on about 1.5M pairs from T: the uncertain pairs plus a sample of confident ones. Fine-tuning starts in the GPU lane as soon as 04b finishes.
- **Which pairs it scores:** GBDT p between 0.03 and 0.97 within each S1's top 10, plus each entity's top-1 candidate when its p is between 0.1 and 0.9. Capped by `--ce-max-pairs` (default 6M; the pairs closest to 0.5 go first).

**08 LLM judge → v3** (`llm.py`):
- **Model:** Qwen3-4B-Instruct-2507 in bf16, prompted with 4 fixed examples taken from T. The score is logit(Yes) − logit(No) from a single forward pass, so no generation is needed.
- **Which pairs it scores:** those where the tier-1 combined probability is between 0.15 and 0.85, plus marginal "match or no match" top-1 decisions. Capped by `--llm-max-pairs` (default 200k test pairs).
- `--llm-finetune lora` exists but is off by default, because bitsandbytes/peft support on Blackwell is a risk.

**Gating** (stage 09): a tier is used only if Vcal macro F0.5 improves by at least 0.001 **and** no country drops by more than 0.002. `09_finalize` copies the chosen tier to `output/`, runs the validator with `--check-ids`, and writes `work/reports/summary.json`. That file contains recall and reduction ratio, F0.5 per tier and country, the leave-one-country-out results, France's predicted-match rate (diagnostic only), runtimes, and every automatically chosen parameter.

## Delivery sequence tonight
1. **Hand-off 1 (early):** `requirements.txt`, `00_check_env.py`, `download_models.py` and setup steps. Your friend creates the environment, downloads the models and copies `dataset/` while I build the rest.
2. Build stages 01–06 with unit tests, and smoke-test them locally on `tools/make_sample.py` data. That sample has about 3k train S1 with their matches plus distractors, and about 2k test S1 across all 3 countries.
3. Build stages 03, 07, 08 and 09, and smoke-test them on the laptop's RTX 4060 with a tiny LLM (`--llm-model Qwen/Qwen3-0.6B`) and small budgets. Also test resuming by killing a run and restarting it.
4. **Hand-off 2:** the full code. Your friend runs `bash run_all.sh` inside tmux. Estimated remote wall time (my estimates): **v1 about 3–3.5 h after start, v2 about 4.5 h, v3 about 5.5–6 h.** Upload whichever is newest when the deadline arrives; the budget flags shorten runtime without changing the architecture.
5. Your friend sends back `work/reports/` and `logs/`. I then fill in `Documentation_template.md` with real numbers, and `tools/package_submission.sh <team>` builds `<team>_submission.zip`.

## Verification
- **Unit tests** (`pytest tests/`):
  - The README example gives F0.5 = 0.714, and the singleton rules score 1/0 as specified.
  - Normalization is checked on real rows: Devanagari names, noise prefixes, `capitalholding.com`, and French accents and abbreviations.
  - The numba CNP top-k matches a brute-force computation on random small data.
- **Local smoke run:** the full `run_all.sh` on the sample data finishes every stage including the GPU ones. v1, v2 and v3 all get `PASS` from `python3 utils/validate_submission.py --matching … --candidate … --test-dir <sample test>`. A killed-and-restarted run resumes without redoing finished stages.
- **Remote:** `00_check_env` must pass before anything else runs. The final `output/` must pass the validator with `--check-ids`. In `summary.json`, recall at k should be at least 0.99, Vtest F0.5 should be reported per country, and the gating decisions should be logged.
