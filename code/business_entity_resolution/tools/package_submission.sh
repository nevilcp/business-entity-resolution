#!/usr/bin/env bash
# Build <team_name>_submission.zip at the student_resource/ root from
# output/, this code/business_entity_resolution/ checkout, and
# Documentation_template.md -- the layout the README's "Final Submission
# Package" section describes.
#
# Usage: tools/package_submission.sh <team_name>
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <team_name>" >&2
  exit 1
fi
TEAM="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"        # code/business_entity_resolution
REPO_ROOT="$(cd "$CODE_DIR/../.." && pwd)"      # student_resource

OUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/output}"
DOC="${DOC_PATH:-$REPO_ROOT/Documentation_template.md}"
ZIP_PATH="$REPO_ROOT/${TEAM}_submission.zip"

for f in "$OUT_DIR/matching_results.tsv" "$OUT_DIR/candidate_pairs.tsv" "$DOC"; do
  if [ ! -f "$f" ]; then
    echo "missing required file: $f" >&2
    exit 1
  fi
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/output" "$STAGE/code/business_entity_resolution"
cp "$OUT_DIR/matching_results.tsv" "$OUT_DIR/candidate_pairs.tsv" "$STAGE/output/"
cp "$DOC" "$STAGE/Documentation_template.md"

cp -r "$CODE_DIR/src" "$CODE_DIR/scripts" "$CODE_DIR/tools" "$CODE_DIR/tests" "$STAGE/code/business_entity_resolution/"
cp "$CODE_DIR/README.md" "$CODE_DIR/requirements.txt" "$CODE_DIR/run_all.sh" "$STAGE/code/business_entity_resolution/"
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$STAGE" -name "*.pyc" -delete

rm -f "$ZIP_PATH"
(cd "$STAGE" && zip -qr "$ZIP_PATH" .)
echo "wrote $ZIP_PATH"
