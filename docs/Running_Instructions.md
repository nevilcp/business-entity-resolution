# Business Entity Resolution Pipeline — Setup & Execution Guide

## 1. Directory Structure

The default paths baked into `run_all.sh` (`--data-dir ../../dataset`,
`--out-dir ../../output`, `--work-dir ../../work`) assume
`code/business_entity_resolution/` sits inside a parent directory laid out
like this (the parent is referred to below as `student_resource/`, but it
can be named anything — only the *relative* layout matters):

```text
student_resource/
├── dataset/                          <- copy the dataset here
├── output/                           <- created by the pipeline
├── work/                             <- created by the pipeline (scratch space)
└── code/business_entity_resolution/  <- this codebase
```

> **Different layout?** Pass `--data-dir /path/to/dataset`,
> `--out-dir /path/to/output`, and `--work-dir /path/to/work` explicitly to
> every command below instead of relying on the defaults.

## 2. Environment Setup (Hand-off 1)

From the project root, create a virtual environment, install dependencies,
pre-fetch the required models, and verify the environment:

```bash
cd code/business_entity_resolution
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/download_models.py   # pre-fetches intfloat/multilingual-e5-small,
                                      # xlm-roberta-base, Qwen/Qwen3-4B-Instruct-2507
python3 scripts/00_check_env.py
```

Confirm `00_check_env.py` prints `00_check_env: PASS` before continuing.

## 3. Pipeline Execution (Hand-off 2)

Run the full pipeline inside a `tmux` session so it survives a dropped SSH
connection during long-running stages:

```bash
tmux new -s er
source .venv/bin/activate   # if not already active in the new tmux pane
bash run_all.sh
```

With no extra flags, `run_all.sh` runs the full pipeline against the full
dataset using production budgets and models (`multilingual-e5-small` for
dense retrieval, `xlm-roberta-base` for the cross-encoder, and
`Qwen3-4B-Instruct-2507` for the LLM judge).

### Running on the 16GB-RAM / RTX 4060 (8GB) laptop

The pipeline is sized to run here, not only on the 32GB / 12GB remote box:
every stage loads one scope (train or test) and one country at a time, and
the large per-pair stages (05 features, 06 prediction, 07/08 rescoring)
stream in chunks. Measured peaks on the full dataset are listed in
`code/business_entity_resolution/README.md`.

- `run_all.sh` runs the stages one after another and, when `systemd-run` is
  available, caps each stage's RAM (`ER_MEM_MAX`, default: RAM available at
  launch minus 1.5GB). A stage that hits the cap is killed on its own
  (exit 137) instead of the kernel OOM-killing the desktop. Close the
  browser/IDE before a full run to give it more headroom, then just rerun to
  resume.
- `bash run_all.sh --skip-dense` skips the optional dense channel (stage 03,
  ~1h of GPU time).
- Keep at least ~20GB of disk free: `work/` grows to ~15GB on the full data
  (stage 05's test features alone are several GB).
- If a crash ever leaves the GPU in a bad state (`nvidia-smi` shows `ERR!`,
  kernel log shows `Xid ... GPU Reset Required`), reboot before rerunning;
  `00_check_env` fails fast in that state instead of letting the GPU stages
  fall back to CPU.

## 4. Session Management

- **Detach from tmux:** `Ctrl+b`, then `d`
- **Reattach to check progress:**

  ```bash
  tmux attach -t er
  ```

## 5. Process Recovery & Resume

If the run crashes or is stopped, just rerun the same command — every stage
checks a `_DONE` marker under `work/<stage>/` and skips work it already
finished:

```bash
bash run_all.sh
```

## 6. Output Verification

Once the run finishes (or at any point to inspect progress so far):

- **View the metrics report:**

  ```bash
  cat work/reports/summary.json
  ```

  Contains blocking Recall/Reduction-Ratio, F0.5 per tier and per country,
  the gating log (which tier was accepted/rejected and why), and the LOCO
  (leave-one-country-out) diagnostic.

- **Inspect the output directory:**

  ```bash
  ls output/
  ```

  Contains `matching_results.tsv` and `candidate_pairs.tsv` for the
  highest tier that passed gating.

> **Tip:** `output/` is (re)written automatically with the best passing
> tier every time stage `09_finalize` completes. Whatever is in `output/`
> is safe to submit at any point.

## 7. Validate & Package the Submission

`09_finalize` already validates `output/` automatically, but you can rerun
the check manually at any time:

```bash
cd student_resource/
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test --check-ids
```

Once `output/` is valid and `Documentation_template.md` (at the
`student_resource/` root) is filled in, build the submission zip:

```bash
cd student_resource/code/business_entity_resolution
tools/package_submission.sh <team_name>
```

This writes `<team_name>_submission.zip` at the `student_resource/` root,
bundling `output/`, this codebase, and `Documentation_template.md`.
