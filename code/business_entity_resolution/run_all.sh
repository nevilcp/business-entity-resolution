#!/usr/bin/env bash
# End-to-end pipeline: 00 -> 01 -> 02 -> 03 (GPU, optional) -> 04 -> ... -> 09.
#
# Stages run strictly one after another. (The original plan ran 05+06 in
# parallel with 07's fine-tuning; on a 16GB-RAM / 8GB-GPU laptop two stages
# at once can exhaust both RAM and VRAM, so that lane is serialized.)
#
# Every stage resumes from its work/<stage>/_DONE marker (see
# er/io.py's stage_is_done/mark_stage_done), so killing and restarting this
# script picks up where it left off instead of redoing finished work.
#
# Memory guard: when systemd-run is available, each stage runs in its own
# transient scope capped at ER_MEM_MAX (default: RAM available at launch
# minus 1.5GB, at least 6GB -- close the browser/IDE first for more). If a
# stage ever exceeds it, only that stage is killed -- instead of the kernel's
# global OOM killer picking off the desktop, the browser or the GPU driver's
# clients -- and a rerun resumes from the last finished stage. Set
# ER_MEM_MAX=off to disable, or e.g. ER_MEM_MAX=10G to override.
#
# Usage: bash run_all.sh [extra flags passed through to every stage]
#   Full run:   bash run_all.sh
#   Skip the optional dense channel (saves ~1h of GPU time):
#     bash run_all.sh --skip-dense
#   Laptop smoke test on tools/make_sample.py's small dataset:
#     bash run_all.sh --data-dir ../../dataset_sample \
#       --n-train 2000 --n-vcal 300 --n-vtest 300 \
#       --max-pairs 2000 --max-train-pairs 5000 --ce-max-pairs 2000 \
#       --llm-model Qwen/Qwen3-0.6B --llm-max-pairs 500
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PYTHON:-python3}"
ARGS=("$@")

# Fewer glibc malloc arenas: many-threaded numpy/numba/arrow code otherwise
# fragments the heap and holds on to freed memory.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
# A fixed mmap threshold: blocks >= 16MB (parquet batches, feature chunks)
# are always mmap'd and returned to the OS on free. glibc's default *raises*
# the threshold after the first large free, after which such blocks land on
# the heap and fragment (measured ~1GB of RSS ratcheting up in stage 06).
export MALLOC_MMAP_THRESHOLD_="${MALLOC_MMAP_THRESHOLD_:-16777216}"
# Arrow's default mimalloc pool holds on to freed parquet buffers (GBs);
# er/io.py switches it too, this also covers anything read before that import.
export ARROW_DEFAULT_MEMORY_POOL="${ARROW_DEFAULT_MEMORY_POOL:-system}"
# Lets PyTorch return fragmented-but-unused VRAM blocks instead of OOMing on
# an 8GB card.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MEM_WRAP=()
if [[ "${ER_MEM_MAX:-}" != "off" ]] && command -v systemd-run >/dev/null 2>&1 \
    && systemd-run --user --scope -q true >/dev/null 2>&1; then
  if [[ -z "${ER_MEM_MAX:-}" ]]; then
    avail_mb=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 ))
    cap_mb=$(( avail_mb - 1536 ))
    (( cap_mb < 6144 )) && cap_mb=6144
    ER_MEM_MAX="${cap_mb}M"
  fi
  MEM_WRAP=(systemd-run --user --scope -q -p "MemoryMax=${ER_MEM_MAX}" -p MemorySwapMax=0)
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run() {
  local name="$1"
  shift
  log "=== $name ==="
  local rc=0
  "${MEM_WRAP[@]}" "$PY" "$@" "${ARGS[@]}" || rc=$?
  if (( rc != 0 )); then
    log "!!! $name failed (exit $rc). Exit 137 = killed at the memory cap" \
        "(ER_MEM_MAX=${ER_MEM_MAX:-off}); free up RAM or raise the cap, then rerun to resume."
    exit "$rc"
  fi
}

if [[ ${#MEM_WRAP[@]} -gt 0 ]]; then
  log "per-stage memory cap: ${ER_MEM_MAX}"
fi

run "00_check_env" scripts/00_check_env.py
run "01_load" scripts/01_load.py
run "02_normalize" scripts/02_normalize.py
run "03_dense" scripts/03_dense.py
run "04_blocking" scripts/04_blocking.py
run "05_features" scripts/05_features.py
run "06_gbdt" scripts/06_gbdt.py
run "07_cross_encoder (fit)" scripts/07_cross_encoder.py --mode fit
run "07_cross_encoder (score)" scripts/07_cross_encoder.py --mode score
run "08_llm_judge" scripts/08_llm_judge.py
run "09_finalize" scripts/09_finalize.py

log "=== done: see output/ and work/reports/summary.json ==="
