# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TODO: team name]  
**Team Members:** [TODO: list all team members]  
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

Our solution works in three steps. First, a fast search step (blocking) narrows each Source 1 business down to at most 50 likely candidates from Source 2 and Source 3. It uses word overlap, character-level overlap and an optional neural text embedding. Second, an XGBoost model scores every candidate pair, and two optional models re-check only the pairs it is unsure about: a fine-tuned cross-encoder and a small LLM. Each of these is kept only if it improves the validation score. Third, a decision step picks the final matches for each business so that the expected F0.5 score is as high as possible. It also enforces the rule that a Source 2/3 record can belong to only one Source 1 business, and it can confidently answer "no match".

*Some technical terms and parameters are used throughout. Each is explained in plain words in the legend in Appendix C.*

---

## 2. Methodology

### 2.1 Problem Analysis

Before designing anything, we studied the training files and the full ground truth. The main findings, and what each one meant for the design:

| What we found in the data | What we did about it |
|---|---|
| The data is large. Training has 2.21M Source 1 (S1) records, 5.03M Source 2 (S2) and 5.29M Source 3 (S3). The test set has 1.73M S1 records (US 663k, India 810k, France 259k) to match against about 10M S2 and S3 records. | Comparing every pair is impossible (around 10¹³ comparisons), so we need a blocking step. It must also read the data in chunks so memory use stays bounded. |
| In **100%** of true pairs, both records have the same country. | Everything runs one country at a time. Country is the only hard filter. It is treated as an open list of labels, so France needs no special code. |
| Each S2/S3 record matches **at most one** S1 record. | The decision step enforces this rule ("record uniqueness"). |
| An S1 record has 3.46 matches on average and at most 11. **5.5% of S1 records have no match at all** (singletons). | The model must be able to answer "no match". Under the scoring rule, a correct empty answer is worth a full 1.0. |
| Every true pair shares at least one word. Ignoring words that appear in 20,000 or more records still keeps 99.5% of true pairs. | Word-based blocking is safe, and very common words can be dropped to save time. |
| S1 names are always in Latin script. About 12–14% of S2/S3 names are not plain ASCII, mostly Hindi (Devanagari script) names for Indian businesses. About 7% of Indian true pairs have a Devanagari name on the S2/S3 side. | We convert Indian scripts to Latin letters (transliteration) before any comparison. See §4. |
| About 2.3% of S2/S3 names start with junk characters such as `--`, `***`, `>>`, `[..]`, `#` or `@` (0.1% in S1). Some names are written like web domains, for example `capitalholding.com`. | These patterns are removed during cleaning. |
| Legal endings (LLC, Inc, Ltd, Pvt, and so on) appear in 48–62% of names, and one source often includes them while another leaves them out. | The legal ending is split into its own field and compared separately, so it does not distort name similarity. |
| An address is one free-text string with no fixed order. US addresses appear as both "street, city, state" and "state, city, street", and Indian addresses can have up to 12 comma-separated parts. About 3.5% of S2/S3 addresses are empty. ZIP/PIN codes appear in only 6–7% of rows. | Addresses are compared as bags of words. House numbers and postcodes are pulled out as separate signals, and an empty address is marked as "missing" instead of being treated as a mismatch. |
| Generic names repeat across different businesses ("Physical Therapy", "Internal Medicine"). | The model is told how common a name is, and it leans on the address when the name is generic. |
| France appears only in the test set and has no labels. | We use no country-specific features. Word statistics and dictionaries are rebuilt for each country's data without labels, and a leave-one-country-out test checks that the model works on a country it has never seen. |

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Blocking (word + character + optional neural channels, merged with RRF and pruned with CNP), then a gradient-boosted tree classifier, then optional cross-encoder and LLM re-scoring of uncertain pairs (each kept only if it helps), then a decision step that maximizes expected F0.5.

**Core Innovation:** The main contribution is a decision step built around the scoring rule. We first turn model scores into calibrated probabilities. Each S2/S3 record then keeps only its single most likely S1. Finally, for each S1 we sort its candidates by probability and keep however many of the top ones give the highest *expected* F0.5 for that business, which can be zero. On top of that, a minimum probability τ is tuned on validation data. Singletons are therefore handled by the same calculation instead of a separate rule. A second contribution is the "competing S1" features. We run blocking for every train and test S1, so for any S2/S3 record we know how strongly other S1 businesses also claim it. The model sees the same kind of information in training and at test time.

The pipeline runs from one script, `run_all.sh`. Each stage saves a completion marker linked to its settings, so an interrupted run picks up where it stopped:

```text
00 check environment ─ 01 load and split ─ 02 build dictionaries + clean text ─ 03 neural search (GPU, optional)
  ─ 04 blocking: word + 3-gram (+ neural) → RRF → CNP top-k  ──► candidate_pairs.tsv
  ─ 05 pair features ─ 06 XGBoost + decision ──► submission v1
  ─ 07 cross-encoder: train, then re-score uncertain pairs ──► submission v2
  ─ 08 LLM judge on the hardest pairs ──► submission v3
  ─ 09 finalize: compare v2/v3 with the tier before, validate, copy the best to output/
```

**Training and validation.** Training S1 businesses are split by business, with equal country proportions in each part (random seed 2026):

- **T** (300,000 S1): used to train the models.
- **Vcal** (50,000 S1): used to calibrate probabilities and to choose τ, the candidate count k, and which tiers to keep.
- **Vtest** (50,000 S1): used only to report the final score. Nothing is tuned on it.

The remaining training S1 records are still passed through blocking so that the "competing S1" statistics look the same as on the test set, but they are never used to train. Test data is used only as input for predictions. The only things computed from test data are unlabelled statistics of each country's pool (word rarity, common legal endings, common abbreviations), which are computed the same way for training data.

### 2.3 Constraints

**Competition rules we designed around:**

1. **Output format.** `matching_results.tsv` must have exactly one row for every test S1 business, list only S2/S3 IDs that exist in the test set, and contain no duplicate IDs. `candidate_pairs.tsv` must be the exact candidate set the model scored, and every final match must appear in it. Both files are checked by the official validator (`utils/validate_submission.py --check-ids`) after every tier.
2. **Model licences and size.** Every model must be MIT or Apache-2.0 licensed with at most 8 billion parameters. Our three models together have about **4.4B** parameters (details in §4).
3. **No external data.** No external databases, APIs, geocoding or web lookups are used anywhere. All models are downloaded once and run locally. Every dictionary is built only from the provided training data.
4. **Unseen country.** Country must be treated as an open set of labels (no hard-coding of US/India, no one-hot country features), and France, which has no training labels, must still be predicted.
5. **No tuning on test data.** All learning and tuning use training data only (the Vcal split). The test set is used only for inference.

**Compute.** The pipeline was built to be memory-bounded rather than to depend on a large server. It loads one country and one data split at a time and processes pairs in chunks. Measured on the full dataset, no stage needed more than about 6.3 GB of RAM (Appendix B), so it runs comfortably on the machine below. The final run uses:

| Component | Specification |
|---|---|
| Operating system | Ubuntu 24.04.2 LTS (x86-64) |
| CPU | Intel Core i9-14900KS, 24 cores / 32 threads, up to 5.9 GHz |
| System memory | 32 GB |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB |
| Software | Python 3.12, PyTorch 2.14 (CUDA 13), XGBoost 3.4, Transformers 5.17. All versions are pinned in `requirements.txt` |

**Time.** A full run takes several hours, so the pipeline writes a valid submission after every tier (v1, v2, v3). If the run is cut short, the best finished tier can still be submitted. Two settings, `--llm-max-pairs` and `--ce-max-pairs`, limit how many pairs the slow tiers score without changing the design.

---

## 3. Candidate Generation (Blocking)

Blocking is done separately for each country. All S2 and S3 records of one country form a "pool", and each S1 record is a query against that pool (`src/er/blocking.py`, `scripts/04_blocking.py`). Each search method below is called a *channel*.

- **Blocking keys used:**
  1. **Word channel.** The keys are the cleaned name words, the address words, and combined "house number + next street word" keys (for example `221_main`). Each key is weighted by how rare it is in that country's pool (IDF), so a shared rare word counts for much more than a shared common one. Every S1 is compared with the pool through a sparse matrix product using IDF-weighted cosine similarity, and the best 60 pool records per S1 are kept.
  2. **Name 3-gram channel.** The keys are every run of 3 consecutive characters in the name with spaces removed. `sunrise` gives `sun, unr, nri, ris, ise`. This still works when words are misspelled, split, joined or reordered. It is also scored with IDF-weighted cosine similarity, and the top 60 are kept.
  3. **Neural channel (optional, uses the GPU).** The small multilingual text model `intfloat/multilingual-e5-small` is fine-tuned for one epoch on 200,000 known matching pairs from T. It turns each record into a vector of numbers (an embedding), and records with similar meaning get similar vectors. For each S1 we find the 60 closest pool records by exact search, which is fast enough on the GPU without an approximate index. This channel is used only if it raises Vcal recall by at least 0.002.
  4. **Merging and pruning.** The ranked lists from the channels are merged with **Reciprocal Rank Fusion (RRF)**. A candidate's merged score is the sum of 1/(60 + its rank) over the channels that found it, so a record ranked highly by any one channel stays near the top. Then **Cardinality Node Pruning (CNP)** keeps only the top *k* merged candidates for each S1. *k* is chosen automatically on Vcal as the smallest value in {10, 15, 20, 30, 40, 50} whose recall is within 0.001 of the recall at k = 50.
  5. **Limits on very common keys.** A word key found in more than 1,000 pool records, or a 3-gram key found in more than 2,000, is ignored. These keys match too many records to be useful and would make the matrix product huge. The limits are fixed counts, not a percentage of the pool: an earlier percentage-based limit let keys with tens of thousands of matches through on the full 6M-record pool, and memory use exploded. S1 queries are also processed in batches of 50,000.
  6. **"Competing S1" statistics.** While blocking every S1 (train and test), we record three things for each pool record: the best and second-best merged score from any S1 that picked it, and how many S1 records picked it. These feed the "Other" features in §4.
- **Candidate pairs generated:** at k = 50 on the full data, **18.7M training pairs** (2.2M S1 against a 10.3M pool) and **82.8M test pairs** (1.73M S1 against a 10.0M pool). Only about 1 in 10,000 of all possible pairs is kept (reduction ratio above 0.9999). `output/candidate_pairs.tsv` is exactly this merged and pruned set, and it is exactly what every model scores.
- **How we ensured true matches were not lost:**
  - Country is the only hard filter, and it holds for 100% of true pairs.
  - The two word-based channels fail in different ways (whole words vs pieces of words), and the neural channel catches meaning-level similarity. RRF merges them, so a weak score in one channel cannot hide a candidate that another channel found.
  - All text is cleaned *before* keys are built (transliteration, junk removal, abbreviation expansion, legal ending split), so surface noise does not change the keys.
  - *k* is chosen from measured recall on Vcal instead of being fixed by hand. For every *k* the stage reports recall (pair completeness), reduction ratio and pair quality for each country.
  - **Measured recall.** On the development sample, recall at k = 50 was 0.997 (India) and 0.998 (US), with a reduction ratio of about 0.996. On the full data, recall fell to **0.856 (India) and 0.804 (US)**. At full scale, the fixed limits on common keys remove far more keys (the 3-gram channel keeps only about 7% of its entries), and this is the current ceiling on our score (see §5 and §6). Blocking peaks at about 6 GB of RAM, so there is room to loosen these limits.

---

## 4. Matching Model

**Text cleaning (stage 02, done before any feature is computed).**

1. **Transliteration.** Indian-script text is converted to Latin letters first, using three sources in order:
   - a dictionary learned from training India pairs by lining up the words of matched names by position (a word pair must be seen at least twice; 1,518 entries on the full data);
   - the LLM, for words the dictionary does not cover;
   - the `anyascii` library as a last resort.

   This has to come before accent stripping, because Devanagari vowel signs are stored as accent-like combining marks and would otherwise be deleted.
2. **Standard cleaning.** Unicode normalization (NFKC), lower-casing, accent removal, removal of junk prefixes and domain endings (`.com`, `.in`, `.fr`).
3. **Abbreviation expansion.** Frequent short words in each country (for example `pvt` → private, `r` → rue) are expanded by the LLM. An expansion is accepted **only if the full word already appears in that country's data**, which prevents made-up expansions (40 accepted for train, 56 for test).
4. **Legal ending split.** Endings such as LLC, Pvt Ltd or SARL are moved into a separate field. The list starts from a fixed seed list and adds each pool's most frequent final words, so a new country is covered automatically.

The cleaned output for each record: clean name, legal ending, a transliteration flag, name without spaces, consonant skeleton of the name, clean address, postcode (5–6 digits) and address numbers.

**Features used** (26 per candidate pair, `src/er/features.py`; no country feature):
- **Name features:** word overlap (Jaccard); IDF-weighted word cosine; character 3-gram cosine; four fuzzy string scores from the `rapidfuzz` library (`ratio`, `token_sort_ratio`, `token_set_ratio` and Jaro-Winkler), computed on the name without its legal ending; abbreviation match (share of the shorter name's words that are prefixes of the longer name's words, e.g. `intl` / `international`); acronym match; legal ending equal / different / missing; transliterated flag; how rare the S1 name is (sum of IDF weights); how many S1 records share the same core name (catches generic names).
- **Address features:** word overlap (Jaccard); IDF-weighted cosine; `token_set_ratio`; number of shared house/street numbers; **number conflict** (both addresses have numbers but none match, a strong sign of different businesses); postcode equal; missing-address flag.
- **Other (context from blocking):** merged RRF score and rank in the S1's candidate list; gap to the S1's best candidate; **reverse rank**, meaning where this S1 ranks among all S1 records that picked the same pool record (1st, 2nd or lower), and the gap to that record's strongest claimant; whether the record comes from Source 3.

**Model type:** three tiers. Each ends with the same decision step described below.

1. **v1: XGBoost** (gradient-boosted decision trees). Tree depth 8, learning rate 0.05, up to 2,000 trees with early stopping on Vcal, optimizing the area under the precision-recall curve, trained on the GPU. We also run a leave-one-country-out check (train on US and test on India, then the reverse) to confirm the features transfer to a country the model has not seen, as a stand-in for France.
2. **v2: cross-encoder.** `xlm-roberta-base`, a multilingual transformer, reads both records together as text (`name | address` for each side, at most 128 tokens) and outputs a match score. It is fine-tuned for one epoch on up to 1.5M pairs from T (true matches plus look-alike non-matches found by blocking), with learning rate 2e-5 in bf16. The word-embedding table is frozen to save GPU memory. It re-scores only uncertain pairs: XGBoost probability p between 0.03 and 0.97 within an S1's top 10, or a top-ranked candidate with p between 0.1 and 0.9. At most 6M pairs are scored, the most uncertain (p closest to 0.5) first.
3. **v3: LLM judge.** `Qwen3-4B-Instruct-2507` (Apache-2.0) is shown 4 fixed example pairs from T and then the pair to judge. We run a single forward pass and use the difference between its scores for the answer "Yes" and the answer "No", without generating any text. The model is loaded in bf16 on GPUs with at least 14 GB of memory, which includes the 16 GB card used for the final run, and in 4-bit NF4 on smaller GPUs. It judges pairs whose tier-2 probability is between 0.15 and 0.85, plus borderline top-ranked decisions, up to `--llm-max-pairs` (50,000 by default), the most uncertain first.

After each tier, a small logistic regression fitted on Vcal combines the previous tier's probability with the new model's raw score into a calibrated probability. For v1 this is just Platt scaling of the XGBoost score. All models are MIT or Apache-2.0 licensed and run locally. Their combined size is about 4.4B parameters (e5-small 0.12B + xlm-roberta-base 0.28B + Qwen3-4B 4.0B), well under the 8B limit.

**Threshold selection method:** direct F0.5 optimization on Vcal, in three steps (`src/er/decide.py`):

1. **Record uniqueness.** Each S2/S3 record keeps only the S1 it most likely belongs to.
2. **Best number of matches per S1.** Sort the S1's candidates by probability and try keeping the top m, for every m from 0 up to the number of candidates. For each m we compute the expected F0.5, treating the candidates' probabilities as independent. The expected number of correct picks is the sum of the kept probabilities, and the expected number of true matches is the sum over all candidates. For m = 0, the expected score is the probability that none of the candidates is a true match, ∏(1 − pᵢ). We keep the m with the highest expected score. This is how singletons are handled.
3. **Probability floor τ.** Drop kept candidates from the bottom until the last one has p ≥ τ. τ is chosen from {0.05, 0.10, …, 0.95} as the value that gives the best macro F0.5 on Vcal.

**Tier gating (stage 09).** v2 or v3 replaces the tier before it only if Vcal macro F0.5 improves by at least 0.001 **and** no single country gets worse by more than 0.002. The chosen tier is checked with the official validator (`--check-ids`) and copied to `output/`.

---

## 5. Results & Error Analysis

> **Status:** the full-data run has finished text cleaning and the neural channel (stages 01–03). Blocking, scoring and the final decision on the full data are still running. The measured scores below therefore come from the **development sample** (built by `tools/make_sample.py`: 3,000 training S1 with their true matches plus distractors from US and India, and about 2,000 test S1 across US, India and France). The whole pipeline, including the GPU tiers, ran end to end on this sample, with the smaller `Qwen/Qwen3-0.6B` standing in for the LLM. These scores are optimistic, because the sample's pools are about 1,000 times smaller than the real ones and blocking is much easier. [TODO: replace with the full-data numbers from `work/reports/summary.json` once the run completes.]

- **F0.5 Score (macro):** **0.9908 on Vtest** (India 0.9921, US 0.9899), using the XGBoost tier (v1), which the gate selected. Vcal was 0.9858 at τ = 0.80. Full-data Vcal/Vtest: [TODO: pending full run]. Earlier partial runs on the full data point to a score of around 0.87, limited by blocking recall of 80–86%.

  | Tier | Vcal F0.5 | Vcal India / US | Vtest F0.5 | Gate decision |
  |---|---|---|---|---|
  | v1 XGBoost | 0.9858 | 0.9824 / 0.9881 | **0.9908** | baseline, **chosen** |
  | v2 + cross-encoder | 0.9875 | 0.9802 / 0.9924 | 0.9897 | rejected: India fell by 0.0022 (limit 0.002) |
  | v3 + LLM judge | 0.9885 | 0.9802 / 0.9941 | 0.9890 | rejected: India fell by 0.0022 |

  The gate did its job. The cross-encoder and LLM tiers raised the overall Vcal score but made India worse, and both scored lower than v1 on Vtest, the held-out set that played no part in the choice.

- **Common false positives (wrong merges):**
  - **Generic or template-like names** ("Physical Therapy", "Life Projects Private Limited", "Eye Group"), where different businesses share a name. When the address is missing or only gives a city, name similarity alone produces a high score. The name-sharing count, name rarity, number conflict and reverse-rank features are there to counter this.
  - **Branches or chains at nearby addresses:** same name, same street or area, different number. The number-conflict feature catches this when both sides have a number, but Indian landmark-style addresses ("Near SBI ATM") often have none.
  - **One record claimed by several S1s:** before the uniqueness step, a pool record can be predicted for two different S1 businesses. The uniqueness step keeps only the stronger claim, which removes this type of error completely.
- **Common false negatives (missed matches):**
  - **Lost during blocking (the biggest cause at full scale):** on the full pools, the limits on common keys drop most keys, so pairs whose only shared words or 3-grams are common never become candidates. Full-data blocking recall was 80.4% (US) and 85.6% (India), against 99.7–99.8% on the sample.
  - **Transliteration differences:** Devanagari names whose words are not in our learned dictionary, where the `anyascii` spelling differs a lot from S1's romanized spelling (for example in vowel length or dropped vowels).
  - **Trade names vs legal names** ("doing business as") with few shared words, especially when the S2/S3 address is missing (about 3.5% of rows). These pairs land in the uncertain range, which is exactly what the cross-encoder and LLM tiers are for.
  - **Deliberate precision bias:** F0.5 rewards precision more than recall, so τ = 0.80 intentionally drops low-confidence true matches for S1 businesses that have many candidates.

---

## 6. Conclusion

We built a memory-bounded entity resolution pipeline that can resume after interruption. It combines word, character and neural blocking (merged with RRF and pruned with CNP), an XGBoost matcher with name, address and "competing S1" features, optional cross-encoder and LLM re-scoring that is kept only when it helps every country, and a decision step that directly maximizes expected F0.5 while allowing each S2/S3 record to join only one business. On the development sample it reaches 0.991 macro F0.5 on held-out data with XGBoost alone. Two lessons stand out. First, at full scale blocking recall limits the score more than the matcher does: the limits on common keys, harmless on a small sample, removed 15–20% of true pairs on the real data, and loosening them (or relying more on the neural channel) is the clear next step. Second, extra re-scoring models must be checked per country, because the LLM and cross-encoder raised the overall validation score while quietly hurting India.

---

## Appendix

### A. Code Artefacts

The code is in `code/business_entity_resolution/`. Library code is under `src/er/` and one script per stage is under `scripts/`:

```text
README.md  requirements.txt (pinned)  run_all.sh        # one command runs everything
src/er/   config.py  io.py  splits.py  metrics.py       # paths, file I/O + resume markers, T/Vcal/Vtest split, macro F0.5
          normalize.py  dictionaries.py  llm.py          # text cleaning, transliteration/abbreviation/legal-ending dictionaries, LLM helper
          dense.py  blocking.py  pairs.py                # neural channel; word/3-gram channels, RRF, CNP; pair utilities
          features.py  gbdt.py  cross_encoder.py         # pair features, XGBoost, cross-encoder
          decide.py  submit.py                           # calibration, uniqueness, expected-F0.5 decision, output writer + validation
scripts/  00_check_env.py  01_load.py  02_normalize.py  03_dense.py  04_blocking.py
          05_features.py  06_gbdt.py  07_cross_encoder.py  08_llm_judge.py  09_finalize.py  download_models.py
tools/    make_sample.py  package_submission.sh
tests/    metrics, text cleaning, blocking (CNP vs brute force, batching), decision step, LLM precision choice, output format
```

To reproduce both output files:

```bash
cd code/business_entity_resolution
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python3 scripts/download_models.py     # one-time download of e5-small, xlm-roberta-base, Qwen3-4B
python3 scripts/00_check_env.py        # must print PASS
bash run_all.sh                        # reads ../../dataset, writes ../../output/{matching_results,candidate_pairs}.tsv
```

A valid submission is written after every tier (`work/06_gbdt`, `work/07_cross_encoder`, `work/08_llm_judge`), so the run is useful even if it is stopped early. `09_finalize` writes the best tier that passed the gate to `output/`, and a full report to `work/reports/summary.json`. Tests: `PYTHONPATH=src pytest tests/`.

### B. Additional Results

**Blocking on the development sample** (k = 50 chosen, neural channel not used):

| Country | S1 records | Pool records | Recall (PC) | Reduction ratio (RR) | Pair quality (PQ) |
|---|---|---|---|---|---|
| India | 1,201 | 11,436 | 0.9969 | 0.9956 | 0.070 |
| US | 1,799 | 16,920 | 0.9984 | 0.9970 | 0.069 |

Vcal recall by number of candidates kept (k): 10 → 0.964, 15 → 0.979, 20 → 0.984, 30 → 0.994, 40 → 0.994, 50 → 0.998.

**Leave-one-country-out check for XGBoost** (average precision, sample): trained on US and tested on India, 0.9952; trained on India and tested on US, 0.9975. The features work across countries without any country-specific terms, which supports using the model on France.

**Memory use per stage on the full dataset** (peak RAM of the whole stage, including worker processes):

| Stage | Peak RAM | Peak GPU memory |
|---|---|---|
| 04 blocking | 6.2 GB | – |
| 05 features (4 workers) | 6.3 GB | – |
| 06 XGBoost | 6.0 GB | about 2.5 GB |
| 07 cross-encoder (train / score) | 3.8 / 5.9 GB | 4.0 GB |
| 08 LLM judge | 5.2 GB | 4.0 GB when loaded in 4-bit; about twice that in bf16 |

[TODO: add the full-data F0.5 table per tier and per country, the chosen k with recall and reduction ratio per country, and France's predicted match rate (for information only) from `work/reports/summary.json` after the full run.]

### C. Legend

| Term | Meaning |
|---|---|
| S1, S2, S3 | Source 1, 2 and 3. S1 is the clean reference list; for each S1 business we look for its records in S2 and S3. |
| Pool | All S2 and S3 records of one country, which the S1 records of that country are searched against. |
| Singleton | An S1 business with no matching record. The correct answer for it is an empty list. |
| Precision / Recall | Precision: the share of predicted matches that are correct. Recall: the share of true matches that were found. |
| F0.5 (macro) | The competition score. It combines precision and recall but counts precision twice as much. "Macro" means it is computed for each S1 business separately and then averaged. |
| Blocking | A cheap first pass that picks a short list of likely candidates, so the expensive models don't have to compare every possible pair. |
| Candidate pair | An (S1, S2/S3) pair that survived blocking and is scored by the model. |
| PC (pair completeness) | Blocking recall: the share of true pairs that made it into the candidate list. |
| RR (reduction ratio) | The share of all possible pairs that blocking removed. Closer to 1 means fewer pairs left to score. |
| PQ (pair quality) | The share of candidate pairs that are true matches. |
| Token | A single word after cleaning. |
| 3-gram | A run of 3 consecutive characters, for example `sun` in `sunrise`. Useful when words are misspelled. |
| DF (document frequency) | How many records in the pool contain a given key. Keys with a very high DF are ignored. |
| IDF (inverse document frequency) | A weight that is high for rare words and low for common ones, so that sharing a rare word counts for more. |
| Cosine similarity | A 0-to-1 measure of how similar two weighted word lists (or vectors) are. |
| Jaccard | Shared words divided by all distinct words across both texts. |
| Jaro-Winkler, `ratio`, `token_sort_ratio`, `token_set_ratio` | Fuzzy string-similarity scores from the `rapidfuzz` library. They tolerate typos, reordered words and extra words, respectively. |
| Transliteration | Writing a word from one script in another, for example Devanagari राम as `ram`. |
| NFKC | A standard Unicode clean-up that makes look-alike characters identical. |
| Embedding / dense channel | A neural model turns each record into a list of numbers. Records with similar meaning get similar lists, which lets us find matches that share few exact words. |
| multilingual-e5-small | A small (0.12B parameter) open multilingual model that produces these embeddings. |
| kNN (k-nearest neighbours) | Finding the k closest records to a query. |
| RRF (Reciprocal Rank Fusion) | A way to merge several ranked lists: each list adds 1/(60 + rank) to an item's score. The 60 is a standard damping constant. |
| CNP (Cardinality Node Pruning) | Keeping only the top *k* candidates for each S1, which puts a fixed upper limit on how many pairs the model has to score. |
| k | The number of candidates kept per S1 after CNP (chosen from 10 to 50). |
| T / Vcal / Vtest | Our split of the training S1 records: T trains the models, Vcal is used for tuning, and Vtest is kept aside only to report the final score. |
| XGBoost / GBDT | Gradient-boosted decision trees: many small decision trees, each correcting the errors of the ones before. Fast and strong on tabular features. |
| Depth, learning rate (η), early stopping | XGBoost settings: how many questions deep each tree can go, how large a step each new tree takes, and stopping training once the validation score stops improving. |
| AUC-PR / average precision | A single number (0 to 1) summarizing how well a model ranks true matches above non-matches across all thresholds. |
| Cross-encoder | A transformer model that reads both records together and outputs one match score. More accurate than comparing separate embeddings, but slower. |
| xlm-roberta-base | A 0.28B parameter multilingual transformer, used here as the cross-encoder. |
| LLM judge | A large language model (here Qwen3-4B) asked whether two records are the same business. Its confidence in "Yes" over "No" is used as a score. |
| Few-shot | Showing the model a few solved examples in the prompt before the real question. |
| bf16, fp16 | 16-bit number formats that halve GPU memory use compared with standard 32-bit numbers, with little loss in accuracy. |
| NF4 | A 4-bit compressed way of loading a model's weights, used when the GPU has too little memory for bf16. |
| Logit | The raw score a model gives an answer before it is turned into a probability. |
| Epoch | One full pass over the training data. |
| Calibration / Platt scaling | Turning raw model scores into trustworthy probabilities (a score of 0.8 should be right about 80% of the time). Platt scaling does this with a simple logistic regression. |
| Stacker | A small logistic regression that combines the previous tier's probability with a new model's score. |
| τ (tau) | The minimum probability a candidate needs to be kept as a match. |
| p | The model's probability that a candidate pair is a true match. |
| m | How many of an S1's top candidates are kept as matches (0 means "no match"). |
| Expected F0.5 | The average F0.5 we would get if each candidate's probability were exactly right. We choose m to make it as high as possible. |
| Record uniqueness | Our rule that each S2/S3 record is assigned to at most one S1. |
| Reverse rank | For a pool record, where the current S1 ranks among all S1 records that picked that record as a candidate. |
| Tier / gating | Tier: one level of the model cascade (v1, v2, v3). Gating: keeping a higher tier only if it measurably improves validation F0.5 without hurting any country. |
| Leave-one-country-out | Training on one country and testing on another, to check that the model works on a country it has never seen. |

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.
