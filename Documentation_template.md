# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [TODO: team name]  
**Team Members:** [TODO: list all team members]  
**Submission Date:** 2026-09-29

---

## 1. Executive Summary

Our solution works in three steps. First, a fast search step (blocking) narrows each Source 1 business down to at most 50 likely candidates from Source 2 and Source 3. It uses word overlap, character-level overlap and an optional neural text embedding. Second, an XGBoost model scores every candidate pair. A fine-tuned cross-encoder then re-reads each business's ten best candidates, and a small second-stage model combines both scores with how each candidate compares to the other candidates of the same business. A small LLM judge is available as one more optional step, kept only if it improves the validation score; it was not used in the final run. Third, a decision step picks the final matches for each business so that the expected F0.5 score is as high as possible. It also enforces the rule that a Source 2/3 record can belong to only one Source 1 business, and it can confidently answer "no match".

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

**Approach Type:** Hybrid. Blocking (word + character + optional neural channels, merged with RRF and pruned with CNP), then a gradient-boosted tree classifier, then a cross-encoder re-score of each business's ten best candidates, combined by a small stacking model (an optional LLM judge is kept only if it helps), then a decision step that maximizes expected F0.5.

**Core Innovation:** The main contribution is a decision step built around the scoring rule. We first turn model scores into calibrated probabilities. Each S2/S3 record then keeps only its single most likely S1. Finally, for each S1 we sort its candidates by probability and keep however many of the top ones give the highest *expected* F0.5 for that business, which can be zero. On top of that, a minimum probability τ is tuned on validation data. Singletons are therefore handled by the same calculation instead of a separate rule. A second contribution is the "competing S1" features. We run blocking for every train and test S1, so for any S2/S3 record we know how strongly other S1 businesses also claim it. The model sees the same kind of information in training and at test time.

The pipeline runs from one script, `run_all.sh`. Each stage saves a completion marker linked to its settings, so an interrupted run picks up where it stopped:

```text
00 check environment ─ 01 load and split ─ 02 build dictionaries + clean text ─ 03 neural search (GPU, optional)
  ─ 04 blocking: word + 3-gram (+ neural) → RRF → CNP top-k  ──► candidate_pairs.tsv
  ─ 05 pair features ─ 06 XGBoost + decision ──► submission v1
  ─ 07 cross-encoder: train, re-score top-10 candidates, stack ──► submission v2
  ─ 08 LLM judge on the hardest pairs (optional, skipped in the final run) ──► submission v3
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

**Compute.** The pipeline was built to be memory-bounded rather than to depend on a large server. It loads one country and one data split at a time and processes pairs in chunks. Measured on the full dataset, no stage needed more than about 6.3 GB of RAM (Appendix B), so the whole pipeline runs on the laptop below, which is what the final run used:

| Component | Specification |
|---|---|
| Operating system | Ubuntu 24.04.5 LTS (x86-64) |
| CPU | Intel Core i5-13500HX, 20 threads |
| System memory | 15 GB |
| GPU | NVIDIA GeForce RTX 4060 Laptop, 8 GB |
| Software | Python 3.12.3, PyTorch 2.14.0 (CUDA 13.0), XGBoost 3.4.1, Transformers 5.17.0, scikit-learn 1.9.1. All versions are pinned in `requirements.txt` |

**Time.** The final run took about 6.5 hours of compute on that laptop: text cleaning 12 min, neural embedding and search 1 h 45 min, blocking 12 min, pair features 22 min, XGBoost 8 min, cross-encoder training about 90 min and cross-encoder scoring 2 h 17 min (the optional LLM judge was skipped). The pipeline writes a valid submission after every tier (v1, v2, v3). If the run is cut short, the best finished tier can still be submitted. Two settings, `--llm-max-pairs` and `--ce-max-pairs`, limit how many pairs the slow tiers score without changing the design.

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
- **Candidate pairs generated:** blocking runs for every train and test S1 (2.21M and 1.73M businesses), because the "competing S1" statistics need all of them. At k = 50 it keeps exactly 50 candidates per business: **86.6M test pairs** (1.73M S1 against a pool of about 10.0M records) and **20.0M training pairs** for the 400,000 T, Vcal and Vtest businesses. Fewer than 1 in 80,000 of all possible pairs is kept (reduction ratio above 0.99998). `output/candidate_pairs.tsv` is exactly this merged and pruned set, and it is exactly what every model scores.
- **How we ensured true matches were not lost:**
  - Country is the only hard filter, and it holds for 100% of true pairs.
  - The two word-based channels fail in different ways (whole words vs pieces of words), and the neural channel catches meaning-level similarity. RRF merges them, so a weak score in one channel cannot hide a candidate that another channel found.
  - All text is cleaned *before* keys are built (transliteration, junk removal, abbreviation expansion, legal ending split), so surface noise does not change the keys.
  - *k* is chosen from measured recall on Vcal instead of being fixed by hand. For every *k* the stage reports recall (pair completeness), reduction ratio and pair quality for each country.
  - **Measured recall.** On the training businesses, the pair completeness (share of true pairs kept) at k = 50 is **0.9948 for India and 0.9964 for US**, with a reduction ratio above 0.99998. The neural channel is what makes this work: with only the word and 3-gram channels, recall at k = 50 was 0.822 on Vcal, because at full scale the fixed limits on common keys remove most 3-gram keys (the 3-gram channel keeps only about 7% of its entries). Adding the neural channel lifted it to 0.996 (Appendix B has the full table). On Vtest, only 709 of 173,072 true pairs (0.41%) are missing from the candidate set, so blocking is no longer what limits the score; the matcher is.
---

## 4. Matching Model

**Text cleaning (stage 02, done before any feature is computed).**

1. **Transliteration.** Indian-script text is converted to Latin letters first, using three sources in order:
   - a dictionary learned from training India pairs by lining up the words of matched names by position (a word pair must be seen at least twice; 1,518 entries on the full data);
   - the LLM, for words the dictionary does not cover;
   - the `anyascii` library as a last resort.

   This has to come before accent stripping, because Devanagari vowel signs are stored as accent-like combining marks and would otherwise be deleted.
2. **Standard cleaning.** Unicode normalization (NFKC), lower-casing, accent removal, removal of junk prefixes and domain endings (`.com`, `.in`, `.fr`). Commas and semicolons then become spaces and periods, brackets and slashes are dropped, so `road,` matches `road` and `inc.`, `pvt.` and `l.l.c.` are recognised as legal endings.
3. **Abbreviation expansion.** Frequent short words in each country (for example `pvt` → private, `r` → rue) are expanded by the LLM. An expansion is accepted **only if the full word already appears in that country's data**, which prevents made-up expansions (40 accepted for train, 56 for test).
4. **Legal ending split.** Endings such as LLC, Pvt Ltd or SARL are moved into a separate field. The list starts from a fixed seed list and adds each pool's most frequent final words, so a new country is covered automatically.

The cleaned output for each record: clean name, legal ending, a transliteration flag, name without spaces, consonant skeleton of the name, clean address, postcode (5–6 digits) and address numbers. A 5–6 digit number at the very start of an address is treated as a house number, not a postcode (US house numbers are often five digits).

**Features used** (26 per candidate pair, `src/er/features.py`; no country feature):
- **Name features:** word overlap (Jaccard); IDF-weighted word cosine; character 3-gram cosine; four fuzzy string scores from the `rapidfuzz` library (`ratio`, `token_sort_ratio`, `token_set_ratio` and Jaro-Winkler), computed on the name without its legal ending; abbreviation match (share of the shorter name's words that are prefixes of the longer name's words, e.g. `intl` / `international`); acronym match; legal ending equal / different / missing; transliterated flag; how rare the S1 name is (sum of IDF weights); how many S1 records share the same core name (catches generic names; counted over all S1 records of the country, in training and test alike).
- **Address features:** word overlap (Jaccard); IDF-weighted cosine; `token_set_ratio`; number of shared house/street numbers; **number conflict** (both addresses have numbers but none match, a strong sign of different businesses); postcode equal; missing-address flag.
- **Other (context from blocking):** merged RRF score and rank in the S1's candidate list; gap to the S1's best candidate; **reverse rank**, meaning where this S1 ranks among all S1 records that picked the same pool record (1st, 2nd or lower), and the gap to that record's strongest claimant; whether the record comes from Source 3.

**Model type:** three tiers. Each ends with the same decision step described below.

1. **v1: XGBoost** (gradient-boosted decision trees). Tree depth 8, learning rate 0.05, up to 2,000 trees with early stopping on Vcal, optimizing the area under the precision-recall curve, trained on the GPU. We also run a leave-one-country-out check (train on US and test on India, then the reverse) to confirm the features transfer to a country the model has not seen, as a stand-in for France.
2. **v2: cross-encoder and stacker.** `xlm-roberta-base`, a multilingual transformer, reads both records together as text (`name legal-ending | address` for each side, at most 128 tokens) and outputs a match score. It is fine-tuned for one epoch on up to 1.5M pairs from T: every true match plus look-alike non-matches taken from each business's ten best blocking candidates, which are the pairs it will actually be asked to judge. Learning rate is 2e-5 in bf16 with a short linear warm-up (6% of the steps) and linear decay; the word-embedding table is frozen to save GPU memory. At prediction time it re-scores every candidate ranked in a business's top 10 by XGBoost whose XGBoost probability is at most 0.97, about 7 pairs per business (12.0M of the 86.6M test pairs). The stacking model described below then combines the two scores.
3. **v3: LLM judge (optional, not run in the final submission).** `Qwen3-4B-Instruct-2507` (Apache-2.0) is shown 4 fixed example pairs from T and then the pair to judge. We run a single forward pass and use the difference between its scores for the answer "Yes" and the answer "No", without generating any text. The model is loaded in bf16 on GPUs with at least 14 GB of memory, which includes the 16 GB card used for the final run, and in 4-bit NF4 on smaller GPUs. It judges pairs whose tier-2 probability is between 0.15 and 0.85, plus borderline top-ranked decisions, up to `--llm-max-pairs` (50,000 by default), the most uncertain first.

Each tier's probabilities are calibrated on Vcal. For v1 this is Platt scaling (a one-input logistic regression) of the XGBoost score. For v2 and v3 a small gradient-boosted stacker (scikit-learn `HistGradientBoostingClassifier`, 300 trees, learning rate 0.05, 31 leaves) is fitted on Vcal. Its inputs are the previous tier's probability, whether the pair was re-scored, the new model's score, and how the pair compares with the other candidates of the same business: that business's best and summed score, the gap to its best, how many of its candidates score above 0.5, and the pair's rank. It replaced a two-input logistic regression that only saw the re-scored pairs, and it is applied to every candidate, in chunks so memory stays bounded. All models are MIT or Apache-2.0 licensed and run locally. Their combined size is about 4.4B parameters (e5-small 0.12B + xlm-roberta-base 0.28B + Qwen3-4B 4.0B), well under the 8B limit.

**Threshold selection method:** direct F0.5 optimization on Vcal, in three steps (`src/er/decide.py`):

1. **Record uniqueness.** Each S2/S3 record keeps only the S1 it most likely belongs to.
2. **Best number of matches per S1.** Sort the S1's candidates by probability and try keeping the top m, for every m from 0 up to the number of candidates. For each m we compute the expected F0.5, treating the candidates' probabilities as independent. The expected number of correct picks is the sum of the kept probabilities, and the expected number of true matches is the sum over all candidates. For m = 0, the expected score is the probability that none of the candidates is a true match, ∏(1 − pᵢ). We keep the m with the highest expected score. This is how singletons are handled.
3. **Probability floor τ.** Drop kept candidates from the bottom until the last one has p ≥ τ. τ is chosen from {0.05, 0.10, …, 0.95} as the value that gives the best macro F0.5 on Vcal.

**Tier gating (stage 09).** v2 or v3 replaces the tier before it only if Vcal macro F0.5 improves by at least 0.001 **and** no single country gets worse by more than 0.002. The chosen tier is checked with the official validator (`--check-ids`) and copied to `output/`. In the final run v2 passed (Vcal 0.9443 → 0.9773, no country worse) and v3 was not run.

---

## 5. Results & Error Analysis

> **Status:** final full-data run. Every number below was measured on the real training data (the Vcal and Vtest splits) with the pipeline in this submission; the raw figures are in `work/reports/summary.json`.

- **F0.5 Score (macro):** **0.9767 on Vtest** (India 0.9791, US 0.9751), using the cross-encoder tier (v2), which the gate selected. Vcal was 0.9773 (India 0.9795, US 0.9759) at τ = 0.50. France has no labels and is not scored, but its output looks like the labelled countries: 3.49 matches per business and 4.4% empty answers, against 3.39 and 5.4% for India and 3.42 and 5.2% for US (training truth: 3.46 and 5.6%).

  | Tier | Vcal F0.5 | Vcal India / US | Vtest F0.5 | Vtest India / US | Gate decision |
  |---|---|---|---|---|---|
  | v1 XGBoost | 0.9443 | 0.9555 / 0.9368 | 0.9447 | 0.9554 / 0.9376 | baseline |
  | v2 + cross-encoder and stacker | 0.9773 | 0.9795 / 0.9759 | **0.9767** | 0.9791 / 0.9751 | accepted, **chosen** |
  | v3 + LLM judge | – | – | – | – | not run |

  The LLM judge was not run for the final submission. In an earlier full run, with an older and narrower cross-encoder step, it lowered Vtest slightly (0.9562 → 0.9556) while taking about 3.9 hours on this laptop, so we left it out.

- **Where the remaining error is (Vtest, 50,000 businesses, v2):**

  | What happened to the business | Share | F0.5 lost |
  |---|---|---|
  | Answer exactly right | 79.6% | 0 |
  | Correctly answered "no match" | 5.0% | 0 |
  | Only correct matches given, but some true ones missed | 11.0% | 0.0083 |
  | At least one wrong match in the answer | 3.8% | 0.0091 |
  | No true match exists but a match was given | 0.4% | 0.0038 |
  | True matches exist but nothing was given | 0.2% | 0.0021 |

  At the level of single pairs, precision is 0.987 and recall 0.963. Only 709 of the 173,072 true pairs (0.41%) never reach the matcher because blocking dropped them.
- **Common false positives (wrong merges):**
  - **Generic or template-like names** ("Physical Therapy", "Life Projects Private Limited", "Eye Group"), where different businesses share a name. When the address is missing or only gives a city, name similarity alone produces a high score. The name-sharing count, name rarity, number conflict and competing-business features are there to counter this.
  - **Branches or chains at nearby addresses:** same name, same street or area, different number. The number-conflict feature catches this when both sides have a number, but Indian landmark-style addresses ("Near SBI ATM") often have none.
  - **One record claimed by several S1s:** before the uniqueness step, a pool record can be predicted for two different S1 businesses. The uniqueness step keeps only the stronger claim, which removes this type of error completely.
- **Common false negatives (missed matches):**
  - **Answers that are correct but incomplete:** 94% of the missed pairs (5,952 of 6,321 on Vtest) belong to businesses whose answer was otherwise correct (11.0% of businesses). F0.5 rewards precision more than recall, so the decision step deliberately leaves out low-confidence candidates. On Vtest the score is flat for τ between 0.3 and 0.6 (0.9766 to 0.9768) and only falls above 0.7 (0.9762 at 0.7, 0.9750 at 0.8).
  - **Likely causes we did not measure separately:** Devanagari names whose words are not in our learned transliteration dictionary, and trade names versus legal names with few shared words, especially when the S2/S3 address is missing (about 3.5% of rows).
  - **Lost during blocking:** only 0.41% of true pairs, so this is no longer a main cause.
- **A limit of our validation:** only the 400,000 training businesses in T, Vcal and Vtest have their candidates scored, so a Source 2/3 record has far fewer rival businesses on Vtest than on the test set. In an earlier full run the one-business-per-record rule removed 42% of the candidate rows on Vtest but 89% on the test set, and 3.3% of test businesses lost a confident candidate to a rival (0.05% on Vtest). The number of businesses taking part did not change since, so the real test score may be somewhat lower than 0.9767. Scoring all 2.2M training businesses would close this gap at a large compute cost; we did not do it.

---

## 6. Conclusion

We built a memory-bounded entity resolution pipeline that can resume after interruption. It combines word, character and neural blocking (merged with RRF and pruned with CNP), an XGBoost matcher with name, address and "competing S1" features, a cross-encoder re-score of each business's ten best candidates combined by a small stacking model, and a decision step that directly maximizes expected F0.5 while allowing each S2/S3 record to join only one business. On the held-out Vtest split it reaches **0.9767** macro F0.5 (India 0.9791, US 0.9751).

Three things mattered most. First, the neural channel: word and character blocking alone lost about 18% of true pairs at full scale (recall 0.822), and adding it brought recall to 0.996, so blocking now misses only 0.4% of true pairs. Second, correcting the inputs: putting the competing-S1 features on a consistent scale, counting name frequency the same way in training and test, and cleaning punctuation and postcodes together raised the XGBoost tier from 0.9278 to 0.9447. Third, the second stage: letting the cross-encoder see every top-10 candidate and combining its score with how each pair ranks inside its business raised the score from 0.9447 to 0.9767 (the earlier, narrower version of this step reached 0.9562 on the earlier features). What is left is mostly matcher recall on businesses with many true matches. One caveat applies to all the numbers: Vtest has fewer rival businesses per record than the test set does (§5), so the test score may come out somewhat lower.

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

**Blocking on the full training data** (k = 50 chosen, neural channel used):

| Country | S1 records | Pool records | Recall (PC) | Reduction ratio (RR) | Pair quality (PQ) |
|---|---|---|---|---|---|
| India | 883,188 | 4,133,346 | 0.9948 | 0.99999 | 0.069 |
| US | 1,323,633 | 6,186,873 | 0.9964 | 0.99999 | 0.069 |

Vcal recall by number of candidates kept (k):

| k | 10 | 15 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|
| Word + 3-gram channels only | 0.716 | 0.753 | 0.773 | 0.797 | 0.812 | 0.822 |
| With the neural channel | 0.957 | 0.980 | 0.989 | 0.994 | 0.995 | 0.996 |

**Leave-one-country-out check for XGBoost** (average precision, full data): trained on US and tested on India, 0.9567; trained on India and tested on US, 0.9653. The features work across countries without any country-specific terms, which supports using the model on France.

**Memory use per stage on the full dataset** (peak RAM of the whole stage, including worker processes):

| Stage | Peak RAM | Peak GPU memory |
|---|---|---|
| 04 blocking | 6.2 GB | – |
| 05 features (4 workers) | 6.3 GB | – |
| 06 XGBoost | 6.0 GB | about 2.5 GB |
| 07 cross-encoder (train / score) | 3.8 / 5.9 GB | 4.0 GB |
| 08 LLM judge | 5.2 GB | 4.0 GB when loaded in 4-bit; about twice that in bf16 |

These figures were measured on an earlier run of the same stages. Stage 07's scoring step has since changed: the stacking model builds its features in chunks of 5M rows, and a synthetic test on a frame of the full test size (86.6M rows) peaked at 2.3 GB for that step. The final run stayed within an 8.3 GB per-stage limit.

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
