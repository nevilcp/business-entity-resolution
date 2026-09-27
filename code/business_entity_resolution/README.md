# Business Entity Resolution pipeline

Token/Q-gram blocking + Cardinality Node Pruning, with a GPU cascade
(dense e5 channel, GBDT, cross-encoder, LLM judge) on top. See
`../../docs/Implementation_Plan.md` for the full design.

## Setup

```bash
cd code/business_entity_resolution
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/download_models.py       # pre-fetches e5-small, xlm-roberta-base, Qwen3-4B
python3 scripts/00_check_env.py          # must PASS before anything else
```

## Run everything

```bash
bash run_all.sh               # or: bash run_all.sh --skip-dense
```

Reads from `../../dataset` and writes `../../output/matching_results.tsv` +
`../../output/candidate_pairs.tsv`, using `../../work/` as scratch space.
Every stage writes a `_DONE` marker keyed by a hash of its config under
`work/<stage>/`, so killing and restarting `run_all.sh` resumes instead of
redoing finished work. Pass `--data-dir` / `--out-dir` / `--work-dir` to
point at different locations; any extra flag is forwarded to every stage
script (each stage ignores flags it doesn't declare).

A submission is written after every tier finishes (`work/06_gbdt/`,
`work/07_cross_encoder/`, `work/08_llm_judge/`), so there is always a valid
`output/` even if the run is cut short before the LLM-judge tier. The last
stage, `09_finalize`, gates each tier against the previous one (Vcal macro
F0.5 must improve by >= 0.001 with no country dropping by more than 0.002)
and copies the best one to `output/`.

## Resource use on the 16GB-RAM / RTX 4060 (8GB) laptop

Measured on the full dataset (2.2M train S1 / 10.3M train pool, 1.7M test S1
/ 10M test pool; CNP k=50 -> 18.7M train and 82.8M test candidate pairs).
RAM is proportional anonymous memory of the stage's whole process tree.

| stage | peak RAM | peak VRAM | wall time |
|---|---|---|---|
| 03 dense | see below | see below | see below |
| 04 blocking | 6.2 GB | - | 11 min |
| 05 features (4 workers) | 6.3 GB | - | 16 min |
| 06 GBDT | 6.0 GB | ~2.5 GB | 7 min |
| 07 cross-encoder fit / score | 3.8 / 5.9 GB | 4.0 GB | ~100 min fit (1.5M pairs), ~25 min score |
| 08 LLM judge | 5.2 GB | 4.0 GB | ~3.2 h at the default `--llm-max-pairs 50000` |

What keeps it inside 16GB (see each script's docstring for details): one
scope and one country loaded at a time, column-pruned parquet reads, no
Python dict/list over every record, per-pair work streamed in chunks and
written straight to parquet, numpy/numba decide instead of pandas groupby,
the memory-hungry reference validator replaced for `candidate_pairs.tsv` by
a streaming check with the same rules (`er/submit.py`), and allocator
settings in `run_all.sh` (`MALLOC_MMAP_THRESHOLD_`, system Arrow pool) that
return freed memory to the OS. `run_all.sh` also caps each stage's RAM via
`systemd-run` (`ER_MEM_MAX`), so an overrun kills just that stage.

## Run stages individually

```bash
python3 scripts/01_load.py
python3 scripts/02_normalize.py
python3 scripts/03_dense.py       # GPU; skips itself if no CUDA GPU is visible
python3 scripts/04_blocking.py
python3 scripts/05_features.py
python3 scripts/06_gbdt.py        # -> v1
python3 scripts/07_cross_encoder.py --mode fit    # needs only 04; run_all.sh runs it after 06 to keep RAM/VRAM free
python3 scripts/07_cross_encoder.py --mode score  # -> v2, needs 06 done first
python3 scripts/08_llm_judge.py   # -> v3
python3 scripts/09_finalize.py    # gates tiers, writes output/ + work/reports/summary.json
```

## Validate before submitting

```bash
cd ../..   # student_resource/
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids
```

`src/er/submit.py` already calls this validator automatically after every
tier and after `09_finalize`; it falls back to a built-in format check if
`utils/validate_submission.py` isn't present (e.g. inside the packaged
submission zip, which doesn't ship `utils/`).

## Smoke-testing on a small local sample

`tools/make_sample.py` builds a small dataset (about 3k train S1 with their
matches plus distractors, about 2k test S1 across all 3 countries) so the
full pipeline, GPU stages included, can be exercised on a laptop:

```bash
python3 tools/make_sample.py --out-dir ../../dataset_sample
bash run_all.sh --data-dir ../../dataset_sample --work-dir ../../work_sample \
    --min-ram-gb 2 --min-disk-gb 5 \
    --n-train 2000 --n-vcal 300 --n-vtest 300 \
    --max-pairs 2000 --max-train-pairs 5000 --ce-max-pairs 2000 \
    --llm-model Qwen/Qwen3-0.6B --llm-fallback-model Qwen/Qwen3-0.6B --llm-max-pairs 500
```

`--n-train`/`--n-vcal`/`--n-vtest` shrink stage 01's T/Vcal/Vtest split to
match the sample's size (the full-scale defaults are 300k/50k/50k and will
raise on a 3k-entity sample). `--llm-model` swaps in a tiny model so the
LLM-judge tier finishes in reasonable time on a single consumer GPU.

## Tests

```bash
pip install pytest
PYTHONPATH=src pytest tests/
```

## Package the submission zip

```bash
tools/package_submission.sh <team_name>
```

Writes `<team_name>_submission.zip` at the `student_resource/` root from
`output/`, this `code/business_entity_resolution/` checkout, and
`Documentation_template.md`.
