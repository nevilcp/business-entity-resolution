# Dataset Analysis Report — Business Entity Matching Task

**Scope of analysis:** first 5,000 rows of each file (out of much larger full files — see "Full-file sizes" below). Files are TSV, UTF-8 encoded.

## 1. Task shape (inferred)

This is an **entity resolution / record linkage** dataset. Three sources (`source1`, `source2`, `source3`) each list business records (name, address, country). `train_ground_truth.tsv` maps each `source1` entity to the set of `source2`/`source3` entity IDs that refer to the **same real-world business**. The modeling task is: given a source1 record, find its matching records in source2 and source3 (a many-to-many, cross-source deduplication / matching problem), likely evaluated as a retrieval or linking task.

## 2. Full-file sizes (row counts, via `wc -l`, includes header)

| File | Rows |
|---|---|
| `train_source1.tsv` | 2,206,822 |
| `train_source2.tsv` | 5,034,617 |
| `train_source3.tsv` | 5,285,604 |
| `train_ground_truth.tsv` | ~2,206,822 (1 row per source1 entity, not directly counted but implied) |

Source1 is ~2.2M records; source2 and source3 are each ~5M — source1 is the smaller "query" side, source2/3 are the larger "candidate pools" to search/match against.

## 3. Schema (identical across source1/2/3)

| Column | Type | Notes |
|---|---|---|
| `entity_id` | string | Format `S{n}-{digits}`, e.g. `S1-925783039`. Prefix always matches the source file (100% in sample). Fully unique per file (no duplicate IDs in samples). Numeric suffix is **not zero-padded** (length varies 5–9 digits) and spans nearly the full range up to ~999,999,999 even within the first 5,000 rows → **rows are randomly shuffled, not sorted by ID**. Important: this means a plain head/prefix sample of any single file is representative of the whole file, but matching entities are *not* co-located across files (see §7). |
| `business_name` | string | Free text, avg length ~24–25 chars, min 2, max ~70–79. |
| `business_address` | string | Free text, **not decomposed** into street/city/state/zip fields — a single unstructured string. Missing in ~3.4–3.5% of source2/source3 rows; **never missing in source1** (0/5000). |
| `country` | categorical | Only two values observed in all three sources: `US` and `India` (~60%/40% split, consistent across sources: source1 60.5%/39.5%, source2 60.7%/39.3%, source3 59.5%/40.5%). |

`train_ground_truth.tsv` schema:

| Column | Notes |
|---|---|
| `source1_entity_id` | One row per source1 entity (no duplicates in sample). |
| `matched_entity_ids` | Comma-separated list mixing `S2-*` and `S3-*` IDs. **Can be empty** (no match found in either source) — 5.46% of sampled rows. Distribution of match-set size: 0 matches 5.5%, 1: 5.1%, 2: 17.7%, 3: 24.8%, 4: 21.3%, 5: 14.1%, 6: 7.4%, 7: 3.0%, 8: 0.7%, 9: 0.2%, 10: 0.02%. Mean ≈ 3.45 matches/entity, max observed 10. Roughly 62% of matched IDs are `S3-*` vs 48% `S2-*` in the exploded list (both sources contribute, S3 slightly more in this sample). |

## 4. Address structure differs sharply by country — no fixed schema

`business_address` is a single free-text field whose internal structure (comma count, ordering of components) differs by country and is inconsistent even within a country:

- **US addresses**: typically low comma count (2 commas dominant: "street, city, state"), but ordering is **not fixed** — e.g. `"1795 Westchester Drive, High Point, NC"` (street-first) vs `"OH, Columbus, 5559 Orville Avenue"` (state-first) both appear in source1. Some include unit/apartment info (`"Unit APARTMENT G"`) or `PO Box` (present in ~0.8–1% of source2/source3 US-ish rows, 0 observed in source1 sample).
- **India addresses**: much higher comma counts (long tail up to 12 commas), reflecting India's deeper administrative hierarchy (building/floor → locality → city → district → state). Highly verbose and inconsistently ordered, e.g. `"H.No.16-11-23/37/A, 2Nd Floor, Flat No.207, Sagar Hotel Building, Opp.Rta Office, Mo, Osarambagh, Hyderabad, Telangana"`. Some India addresses in source3 contain **regional-script text** (e.g. Kannada `ಕರ್ನಾಟಕ`) mixed into otherwise Latin-script strings.
- Zip/postal codes appear in a minority of rows only (~6–7% match a 5-digit pattern) — cannot be relied on as a join key.
- **~3.4–3.5% of addresses are entirely missing** in source2 and source3 (empty string), but never in source1.

**Implication:** address parsing/normalization (tokenizing into components, fuzzy string matching) will be necessary; there's no structured street/city/state/zip split to exploit directly, and matching logic must be robust to missing addresses.

## 5. Business names: multilingual, noisy, and normalization-heavy

- **Non-ASCII / non-English names are common in source2/3 but absent in source1**: 720/5000 (14.4%) in source2 and 591/5000 (11.8%) in source3 contain non-ASCII characters (0/5000 in source1). These are predominantly **Devanagari (Hindi) script** business names for India-based entities, e.g. `राम मार्केटिंग प्राइवेट लिमिटेड` (source2), `रियल इन्वेस्टमेंट प्राइवेट लिमिटेड` (source3). Source1's India-country rows apparently always use Latin-script/transliterated/English names — **this is a real cross-source asymmetry**: a matching pipeline must handle transliteration or script-normalization between source1 (Latin only) and source2/3 (mixed Latin/Devanagari) for the same India entities.
- **Noisy leading punctuation/prefixes** are common in source2/source3 names (not seen in source1 samples): `"-- Holloway Peak Inc Seafood"`, `"[INCORPORATED] PEAK TRADIN6 NETWORKS SOUTHSIDE"`, `"*** Sai Tech Private Limited"`, `"@DENTCOFFEE"`, `"... White Safe Futurecorp LP"`, `"#centraleducation"`, `">> First  Consultants Private [Limited]"`, `"(Ltd) Producer Aim Solutions"`. About 2.2–2.4% of names in source2/source3 start with a dash/punctuation/bracket vs 0.12% in source1. This looks like injected noise (possibly deliberately, to simulate OCR/scrape artifacts) and should be stripped during normalization.
- **Legal-entity suffixes** (LLC, Inc, Ltd, Pvt, LLP, Corp, etc.) appear in ~50–62% of names across sources (source1: 62%, source2: 48%, source3: 49%) — useful signal but also a common source of false mismatches if not normalized away before comparing core business name.
- **Near-duplicate names within a single source** exist (e.g., "Life Projects Private Limited" appears 3× in source1's 5000-row sample; generic names like "Physical Therapy", "Internal Medicine", "Eye Group", "Wildlife Center" repeat 2×) — these are almost certainly *different* real businesses with generic/templated names, a hard case for name-only matching (address becomes the disambiguating signal).
- A handful of single/double-character or symbol-only names exist in source2 (1) and source3 (9), e.g. effectively degenerate names — edge cases to handle defensively (avoid crashing on very short strings).
- No exact-duplicate `(business_name, business_address)` pairs were found within any single source's sample (0/5000 in each) — no obvious intra-source exact-duplicate entities in this slice.

## 6. Data quality issues summary

| Issue | Where | Severity |
|---|---|---|
| Missing `business_address` (~3.4–3.5%) | source2, source3 only | Medium — must handle null/empty address gracefully in matching features |
| No missing values in `entity_id`/`business_name`/`country` | all sources | — (clean) |
| Free-text noise (stray leading symbols: `--`, `...`, `***`, `#`, `@`, `[]`, `()`, `>>`) | source2 (~2.4%), source3 (~2.2%), source1 (~0.1%) | Medium — needs stripping before name comparison |
| Mixed-script business names (Devanagari + Latin) | source2 (~14%), source3 (~12%), absent in source1 | High — cross-lingual matching problem for India entities |
| Unstructured, inconsistent address format (order varies, variable granularity by country) | all sources | High — no reliable field-level join keys |
| Ground truth can be empty (no match) | ~5.5% of source1 entities | Medium — model/pipeline must support "no match" as a valid prediction |
| Generic/templated business names causing intra-source collisions | all sources | Medium — name alone is insufficient, address must disambiguate |
| `entity_id` numeric suffix not zero-padded, spans near-full range even in head sample | all sources | Informational — confirms rows are pre-shuffled randomly (good: sampling is unbiased; bad: can't assume any locality/ordering) |

## 7. Critical structural finding: files are shuffled independently — matched entities are *not* co-located

Cross-checking the first 5,000 `source1_entity_id`s from ground truth against the first-5,000-row sample of `train_source1.tsv` found only **19 overlapping IDs** (0.4%) — confirming source1 itself is randomly shuffled relative to ground truth order. Checking those 19 rows' matched `S2-*`/`S3-*` IDs against the first-5,000-row samples of source2/source3 found **zero matches present** in either sample.

**Why this matters:** source2 and source3 are ~5M rows each; a 5,000-row head sample covers only ~0.1% of each file. Since files are independently shuffled, the probability that a specific entity's ~3.45 average matches land in the first 5,000 rows of a 5M-row file is negligible. **Any analysis or prototyping that needs actual matched triples (source1 row + its true source2/source3 matches) must either sample by ID (look up specific entity_ids across full files, e.g. via `grep`/indexed lookup) rather than by row position, or load full files** — a naive `head -n 5000` join across the three source files will almost never find aligned matches.

## 8. Recommendations for downstream modeling

1. **Normalize business names** before comparison: strip leading noise characters/brackets, lowercase, strip legal-entity suffixes into a separate feature, and consider transliteration/romanization for Devanagari-script India names to align with source1's Latin-only names.
2. **Treat address as unstructured text** for fuzzy/token-based similarity (e.g., token-set overlap, edit distance) rather than assuming parseable street/city/state/zip fields; handle missing address as a distinct "unknown" state rather than empty string.
3. **Use `country` as a hard pre-filter** (blocking key) — it's clean, categorical, and consistent across all three sources, making it a cheap first-pass filter before expensive name/address similarity scoring.
4. **Support "no match" predictions** — ~5.5% of entities in ground truth have no match at all.
5. **Do not rely on row order/position across files** for joining or sampling — always join on `entity_id`; sample by ID lookup (not row slicing) when full ground-truth verification is needed.
6. Because source2/source3 are ~2.3–2.4x larger than source1, expect the matching/retrieval step to be a search-over-large-candidate-pool problem (blocking + candidate generation before fine-grained scoring will likely be necessary at full scale, given 5M-row candidate pools).
