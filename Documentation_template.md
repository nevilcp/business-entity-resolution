# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TODO: team name]  
**Team Members:** [TODO: list all team members]  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We solve the task as a cascade: **blocking, then scoring, then decoding**. Blocking uses token and character 3-gram matching with IDF-weighted cosine scores. It fuses two lexical channels and an optional fine-tuned multilingual-e5 dense channel with Reciprocal Rank Fusion (RRF), then prunes to the top k candidates per Source 1 entity (Cardinality Node Pruning, CNP). An XGBoost scorer rates every candidate pair using name, address and "graph" features. Two optional tiers then re-score only the pairs the GBDT is unsure about: a fine-tuned xlm-roberta-base cross-encoder, then a Qwen3-4B LLM judge. Stage 09 keeps each tier only if it improves validation F0.5. The main idea is the decision step: each S2/S3 record goes to at most one S1, and each S1 gets the prefix of its ranked candidates that maximizes expected F0.5, with a minimum-probability floor tuned on validation data. This decision step optimizes the macro F0.5 metric directly and handles singletons naturally.

---

## 2. Methodology

### 2.1 Problem Analysis

EDA on the training files (`docs/Dataset_Analysis_Report.md`) and on the full ground truth found:

| Finding | Consequence for the design |
|---|---|
| Train: 2.21M S1, 5.03M S2 and 5.29M S3 records. Test: 1.73M S1 (US 663k, India 810k, France 259k) against about 10M S2+S3 | All-pairs comparison is impossible, so blocking has to be sparse, chunked and streaming (the target machine has 16 GB RAM) |
| Country agrees in **100%** of true pairs | Everything runs per country and country is a hard block. Country is treated as an open set of labels, so France needs no special code |
| Each S2/S3 record matches **at most one** S1 | "Record uniqueness" constraint in the decision step |
| Matches per S1: mean 3.46, max 11. **5.5% of S1s are singletons** | The decoder must be able to predict "no match" (m = 0) |
| Every true pair shares at least one token. Dropping tokens with DF of at least 20k keeps 99.5% of pairs | Token-based blocking with DF caps is safe |
| Source 1 names are Latin-only. About 12–14% of S2/S3 names are non-ASCII, mostly **Devanagari** for India (about 7% of India true pairs have an Indic-script S2/S3 name) | Transliteration dictionary built from training alignments, then LLM, then `anyascii` |
| Injected prefix noise in S2/S3 names (`--`, `***`, `>>`, `[..]`, `#`, `@`) in about 2.3% of rows (0.1% in S1). Domain-style names (`capitalholding.com`) | Noise-stripping regexes in normalization |
| Legal suffixes (LLC/Inc/Ltd/Pvt/…) in 48–62% of names, inconsistently present across sources | Legal form is split off into its own field and compared as a feature |
| Addresses are single free-text strings with no fixed component order (US "street, city, state" vs "state, city, street"; India has up to 12 comma-separated parts). About 3.5% of S2/S3 addresses are empty. ZIP/PIN codes appear in only about 6–7% of rows | Addresses are treated as bags of tokens. Number and postcode features are extracted separately. A "missing address" state is its own feature |
| Generic names repeat across businesses ("Physical Therapy", "Internal Medicine") | Name frequency (IDF sum, how many S1s share the core name) and address evidence are features |
| France appears only in test and has no labels | No country one-hot. Dictionaries and IDF are rebuilt on each pool (unsupervised). Leave-one-country-out check that features transfer |

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Blocking (lexical + dense, RRF + CNP) → GBDT classifier → cross-encoder and LLM re-scoring of uncertain pairs, each tier gated → expected-F0.5 decoding.  
**Core Innovation:** A decision layer built for the metric. Calibrated pair probabilities are first reduced so that each S2/S3 record keeps only its best S1. Each S1 then gets the candidate-prefix size that **maximizes expected per-entity F0.5** (m = 0 scores P(no true match) = ∏(1−pᵢ)), with a probability floor τ tuned on validation data. A second contribution is "competing-S1" graph features. Every train and test S1 is blocked so that, for each pool record, we know how strongly other S1s claim it. This gives the scorer the same information at train and test time.

Pipeline (`run_all.sh`; each stage resumes from a `_DONE` marker keyed on a hash of its config):

```text
00 check env ─ 01 load/split ─ 02 dictionaries + normalize ─ 03 dense kNN (GPU, optional)
  ─ 04 blocking: word + 3-gram (+ dense) → RRF → CNP top-k  ──► candidate_pairs.tsv
  ─ 05 features ─ 06 XGBoost + decide ──► v1
  ─ 07 cross-encoder fit / score uncertain pairs ──► v2
  ─ 08 LLM judge on the hardest pairs ──► v3
  ─ 09 finalize: gate v2/v3 against the previous tier, validate, copy best to output/
```

**Train/validation protocol.** Train S1 entities are split by entity, stratified by country, with seed 2026. **T = 300k** is used for fitting, **Vcal = 50k** for calibration, τ, CNP-k and gating, and **Vtest = 50k** only for reporting scores. The rest of train S1 is still blocked so the competing-S1 statistics match test conditions, but it is never used for fitting. Test data is inference input only. The only statistics computed on test data are unsupervised per-pool ones (IDF, legal-form and abbreviation vocabularies), exactly as for train.

---

## 3. Candidate Generation (Blocking)

Blocking is done per country. The S2 ∪ S3 records of one country form the "pool", and each S1 is a query against it (`src/er/blocking.py`, `scripts/04_blocking.py`).

- **Blocking keys used:**
  1. **Word channel:** normalized name tokens, address tokens, and combined *address-number + next street token* keys (e.g. `221_main`). Pool IDF is computed per country, vectors are L2-normalized, and scores are **IDF-weighted cosine** via a sparse S1 × pool matmul, keeping the top 60 per S1.
  2. **Name 3-gram channel:** character 3-grams of the space-stripped normalized name (robust to typos, word splits and transpositions). IDF-weighted cosine, top 60.
  3. **Dense channel (optional, GPU):** `intfloat/multilingual-e5-small` fine-tuned for one epoch on 200k (S1, match) pairs from T with in-batch negatives. Records are embedded in fp16, then an **exact** chunked top-60 inner-product search runs per country (no ANN index needed). It is folded in only if it raises Vcal recall at the chosen k by at least 0.002.
  4. **Fusion and pruning:** Reciprocal Rank Fusion Σ 1/(60 + rank) over channels, then **Cardinality Node Pruning** keeps the top k per S1. k is chosen automatically on Vcal as the smallest value in {10, 15, 20, 30, 40, 50} whose pairs-completeness is within 0.001 of the value at k = 50.
  5. **Frequent-key caps:** word keys with more than 1,000 pool postings and 3-gram keys with more than 2,000 are dropped. These are *absolute* caps, not a fraction of the pool. A fractional cap let keys with tens of thousands of postings through on the real 6M-record pool and blew the matmul up to 10¹¹ non-zeros. S1 queries are scored in chunks of 50k to bound memory.
  6. **Competing-S1 aggregates:** while blocking *all* S1s (train and test), we keep, for each pool record, the best and second-best fused S1 score pointing at it and how many S1s point at it. These feed the graph features in §4.
- **Candidate pairs generated:** at CNP k = 50 on the full data, **18.7M train pairs** (2.2M S1 × 10.3M pool) and **82.8M test pairs** (1.73M S1 × 10.0M pool). This is a reduction ratio above 0.9999. `output/candidate_pairs.tsv` is exactly this fused CNP set, which is what every scorer runs inference on.
- **How we ensured true matches were not lost:**
  - Country is the only hard filter, and it holds for 100% of true pairs.
  - Two lexical channels with complementary failure modes (whole-word vs sub-word), plus an optional semantic dense channel, merged with RRF so one weak channel doesn't hide a candidate another channel found.
  - Aggressive normalization *before* key extraction (transliteration, noise stripping, abbreviation expansion, legal-form split), so surface noise doesn't change the keys.
  - k is chosen on Vcal pairs-completeness rather than fixed. The stage logs pairs-completeness (PC), reduction ratio (RR) and pair quality (PQ) per country for every k.
  - **Measured recall:** on the development sample, PC = 0.997 (India) and 0.998 (US) at k = 50 with RR ≈ 0.996. On the full data, the measured PC was **only 0.856 (India) and 0.804 (US)**. At full scale the absolute DF caps remove most keys (the 3-gram channel keeps only about 7% of postings), and this is the pipeline's current recall ceiling (see §5 and §6). Blocking now runs in 11 min at 6.2 GB peak RAM, so there is room to loosen the caps.

---

## 4. Matching Model

**Normalization (stage 02, applied before any feature):** Indic → Latin transliteration comes first. It uses a dictionary built by positional token alignment on training India pairs (support ≥ 2, 1,518 entries on the full data), with tokens still missing transliterated by the LLM (batched, greedy decoding), then `anyascii` as the last resort. Transliteration has to happen before accent stripping, because Devanagari vowel signs are combining marks. Then come NFKC, casefold and accent stripping, removal of noise prefixes and domain suffixes, and abbreviation expansion. Expansion candidates are each country's frequent short tokens (e.g. `pvt` → private, `r` → rue), expanded by the LLM, and an expansion is accepted **only if the expanded word already exists in that country's pool vocabulary** (a guard against hallucination; 40 train / 56 test expansions). Finally the legal form is split off (seed list plus each pool's most frequent name-final tokens, so a new country is covered automatically). The output fields are the clean name, legal form, a transliteration flag, a space-stripped name, a consonant skeleton, the clean address, postcode (5–6 digits) and address numbers.

**Features used** (26 per pair, `src/er/features.py`, `scripts/05_features.py`; no country one-hot):
- **Name features:** token Jaccard; IDF-weighted token cosine; character 3-gram cosine; rapidfuzz `ratio`, `token_sort_ratio`, `token_set_ratio` and Jaro-Winkler on the name with the legal form removed; abbreviation-subsequence match (fraction of the shorter name's tokens that are prefixes of the longer name's tokens, e.g. `intl` / `international`); acronym match; legal form equal / different / missing; transliterated flag; S1 name IDF sum; number of S1s sharing the same core name (catches generic names).
- **Address features:** token Jaccard; IDF-weighted cosine; `token_set_ratio`; number of shared address numbers; **number conflict** (both sides have numbers and none are shared, a strong negative signal); postcode equal; missing-address flag.
- **Other (graph / blocking context):** fused RRF score and rank within the S1's candidate list; gap to the S1's best candidate; **reverse rank** (where this S1 ranks among all S1s claiming the record: 1st, 2nd or lower) and gap to the record's best claimant, both from the stage-04 aggregates; S3 flag.

**Model type:** three tiers, each followed by the same decision function:
1. **v1: XGBoost** (`hist`, CUDA when available; depth 8, η 0.05, AUC-PR objective, up to 2,000 rounds with early stopping on Vcal; `QuantileDMatrix` to fit about 15M training rows into 16 GB RAM). A leave-one-country-out diagnostic (train on US → test on India, and the reverse) checks that features transfer to an unseen country, as a proxy for France.
2. **v2: cross-encoder.** `xlm-roberta-base` fine-tuned for 1 epoch on up to 1.5M T pairs (positives plus negatives retrieved by blocking), with input text `name | address` for each side, max_len 128, bf16, lr 2e-5. The 250k-row embedding matrix is frozen so training fits in 8 GB of VRAM. It scores only uncertain pairs: GBDT p ∈ [0.03, 0.97] within an S1's top 10, or a top-1 candidate with p ∈ [0.1, 0.9]. At most 6M pairs are scored, closest to 0.5 first.
3. **v3: LLM judge.** `Qwen3-4B-Instruct-2507` (Apache-2.0; falls back to `Qwen/Qwen3-4B`), prompted with 4 fixed examples taken from T. The score is logit("Yes") − logit("No") from one forward pass, with no generation. It is loaded in bf16 on GPUs with ≥ 14 GB and NF4 on smaller ones. It scores pairs with tier-2 p ∈ [0.15, 0.85] and marginal top-1 decisions, capped by `--llm-max-pairs` (default 50k, closest to 0.5 first).

Each tier combines scores with a per-tier **logistic-regression stacker** fit on Vcal, whose inputs are the previous tier's probability and the new raw score. For v1 this is just Platt scaling of the GBDT score. All models are MIT or Apache-2.0 licensed and self-hosted, with a combined size of about 4.4B parameters (e5-small 0.12B + xlm-roberta-base 0.28B + Qwen3-4B 4.0B), under the 8B limit. No external lookups or APIs are used at any stage.

**Threshold selection method:** F0.5 optimization on Vcal, in three steps (`src/er/decide.py`):
1. **Record uniqueness:** each S2/S3 record keeps only its highest-probability S1.
2. **Per-entity expected-F0.5 prefix:** sort the S1's candidates by probability and choose m ∈ {0, …, n} to maximize expected F0.5, assuming independence. Expected TP is Σ pᵢ over the prefix, and the expected number of true matches is Σ pᵢ over all candidates. The m = 0 option scores ∏(1 − pᵢ). This makes singletons a first-class decision.
3. **Probability floor τ:** trim the prefix until its last member has p ≥ τ. τ is grid-searched over {0.05, 0.10, …, 0.95} for the best Vcal macro F0.5.

**Tier gating (stage 09):** v2 or v3 replaces the previous tier only if Vcal macro F0.5 improves by ≥ 0.001 **and** no country drops by more than 0.002. The chosen tier is validated with `utils/validate_submission.py --check-ids` and copied to `output/`.

---

## 5. Results & Error Analysis

> **Status:** the full-data run has completed stages 01–03 (normalization and the dense channel). Blocking, scoring and decoding at full scale are still to be run. Every measured number below comes from the **development sample** (`tools/make_sample.py`: 3,000 train S1 with their true matches plus distractors, US + India; about 2,000 test S1 across US, India and France). The whole pipeline, GPU tiers included, ran end to end on it. The sample used `Qwen/Qwen3-0.6B` as a stand-in LLM. Sample scores are optimistic, because the sample's pools are about 1,000 times smaller than the real ones. [TODO: replace with full-data `work/reports/summary.json` numbers once the run completes.]

- **F0.5 Score (macro):** **0.9908 on Vtest** (India 0.9921, US 0.9899) with the chosen tier v1 (GBDT); Vcal 0.9858 at τ = 0.80. Full-data Vcal/Vtest: [TODO: pending full run]. Earlier full-data work suggests the ceiling there is around 0.87 because blocking recall is 80–86%.

  | Tier | Vcal F0.5 | Vcal India / US | Vtest F0.5 | Gate decision |
  |---|---|---|---|---|
  | v1 GBDT | 0.9858 | 0.9824 / 0.9881 | **0.9908** | baseline, **chosen** |
  | v2 + cross-encoder | 0.9875 | 0.9802 / 0.9924 | 0.9897 | rejected: India dropped by 0.0022 (> 0.002) |
  | v3 + LLM judge | 0.9885 | 0.9802 / 0.9941 | 0.9890 | rejected: India dropped by 0.0022 |

  The gate worked as intended. v2 and v3 raised overall Vcal but hurt India, and their Vtest scores were in fact lower than v1's.

- **Common false positives (wrong merges):**
  - **Generic or templated names** ("Physical Therapy", "Life Projects Private Limited", "Eye Group") where different businesses share a name. When addresses are missing or only city-level, name similarity alone gives a high score. Mitigations: the S1 name-sharing count, name IDF sum, address-number conflict and the reverse-rank features.
  - **Branches or chains at nearby addresses:** same name, same street or locality, different number. The number-conflict flag catches the cases where both sides have a number; landmark-style India addresses ("Near SBI ATM") often have none.
  - **Records claimed by several S1s:** before record uniqueness, a pool record can be predicted for two S1s. The uniqueness step keeps only the stronger claim, which removes this entire class of error.
- **Common false negatives (missed matches):**
  - **Lost at blocking (the dominant cause at full scale):** on the full pools the absolute DF caps drop most keys, so pairs whose only shared tokens are frequent words or common 3-grams never become candidates. Full-data blocking recall was 80.4% for US and 85.6% for India, compared with 99.7–99.8% on the sample.
  - **Heavy transliteration drift:** Devanagari names whose tokens aren't in the aligned dictionary and whose `anyascii` rendering differs a lot from S1's romanization (vowel length, schwa deletion).
  - **DBA or trade names versus legal names** with little token overlap, together with a missing S2/S3 address (about 3.5% of rows). These pairs sit in the uncertain band and are exactly what the cross-encoder and LLM tiers target.
  - **Precision bias:** because F0.5 favours precision, τ = 0.80 deliberately drops low-confidence true matches in S1s that have many candidates.

---

## 6. Conclusion

We built a resumable, memory-bounded ER cascade that runs within 16 GB RAM and 8 GB VRAM: token/3-gram/dense blocking with RRF and CNP; a GBDT with name, address and competing-S1 graph features; gated cross-encoder and LLM re-scoring; and a decoder that maximizes expected per-entity F0.5 under a one-S1-per-record constraint. On the development sample it reaches 0.991 Vtest macro F0.5 with the GBDT tier alone. The main lesson is that **at full scale, blocking recall, not the matcher, limits the score**. Caps that are harmless on a small sample removed 15–20% of true pairs on the real pools, so the next step is to loosen the DF caps (and/or rely more on the dense channel), which the 11-minute, 6.2 GB blocking stage can afford. A second lesson: re-scoring tiers must be gated per country. The LLM and cross-encoder improved the overall validation score while quietly hurting India.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` (all source under `src/er/`, stage scripts under `scripts/`):

```text
README.md  requirements.txt (pinned)  run_all.sh        # one command, end to end
src/er/   config.py  io.py  splits.py  metrics.py       # paths, parquet I/O + _DONE markers, T/Vcal/Vtest split, macro F0.5
          normalize.py  dictionaries.py  llm.py          # normalization, translit/abbrev/legal-form dictionaries, Qwen helper
          dense.py  blocking.py  pairs.py                # e5 channel, IDF channels + RRF + CNP (numba), pair utilities
          features.py  gbdt.py  cross_encoder.py         # pair features, XGBoost, xlm-roberta cross-encoder
          decide.py  submit.py                           # calibration, uniqueness, expected-F0.5 decoding, TSV writer + validation
scripts/  00_check_env.py  01_load.py  02_normalize.py  03_dense.py  04_blocking.py
          05_features.py  06_gbdt.py  07_cross_encoder.py  08_llm_judge.py  09_finalize.py  download_models.py
tools/    make_sample.py  package_submission.sh
tests/    metrics, normalization, blocking (CNP top-k vs brute force, chunking), decide, LLM precision choice, submission format
```

Reproduce both output files:

```bash
cd code/business_entity_resolution
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python3 scripts/download_models.py     # e5-small, xlm-roberta-base, Qwen3-4B (one-time model download only)
bash run_all.sh                        # reads ../../dataset, writes ../../output/{matching_results,candidate_pairs}.tsv
```

A valid submission is written after every tier (`work/06_gbdt`, `07_cross_encoder`, `08_llm_judge`), so the run is useful even if cut short. `09_finalize` writes the gated best tier to `output/` and a full report to `work/reports/summary.json`. Tests: `PYTHONPATH=src pytest tests/`.

### B. Additional Results

**Blocking on the development sample (CNP k = 50 chosen, dense channel not included):**

| Country | S1 | Pool | Pairs completeness | Reduction ratio | Pair quality |
|---|---|---|---|---|---|
| India | 1,201 | 11,436 | 0.9969 | 0.9956 | 0.070 |
| US | 1,799 | 16,920 | 0.9984 | 0.9970 | 0.069 |

Vcal pairs-completeness by k: 10 → 0.964, 15 → 0.979, 20 → 0.984, 30 → 0.994, 40 → 0.994, 50 → 0.998.

**Leave-one-country-out GBDT (average precision, sample):** train US → test India 0.9952; train India → test US 0.9975. Features transfer across countries without country-specific terms, which supports applying the model to France.

**Full-data resource profile** (16 GB RAM / RTX 4060 8 GB laptop; peak RAM of the whole stage process tree):

| Stage | Peak RAM | Peak VRAM | Wall time |
|---|---|---|---|
| 03 dense (fine-tune + embed + exact kNN, 5 country/scope blocks) | — | — | ≈ 2 h |
| 04 blocking | 6.2 GB | – | 11 min |
| 05 features (4 workers) | 6.3 GB | – | 16 min |
| 06 GBDT | 6.0 GB | ≈ 2.5 GB | 7 min |
| 07 cross-encoder fit / score | 3.8 / 5.9 GB | 4.0 GB | ≈ 100 min / ≈ 25 min |
| 08 LLM judge (50k pairs, NF4) | 5.2 GB | 4.0 GB | ≈ 3.2 h |

[TODO: add the full-data per-tier/per-country F0.5 table, the chosen CNP k and blocking PC/RR per country, and France's predicted match rate (diagnostic only) from `work/reports/summary.json` after the full run.]

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
