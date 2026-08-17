#!/usr/bin/env bash
# Poll a submitted review batch until it completes, then download and collect
# it automatically — chains the 3 manual steps (poll, download, review-collect)
# into one background-able run instead of re-checking and re-running each by hand.
#
# Usage:
#   scripts/wait_and_collect_review.sh <job_id> <lang> <role> [provider]
#
# <role> must match whatever --role was passed to review-submit for this job —
# review-collect uses it to rebuild the same Finding schema the batch was built
# with (see domain/review_collect.py), and different roles can have different
# category sets (e.g. native_reader_batch_pt has no "awkward_phrasing"), so the
# wrong role here silently mis-parses results rather than erroring clearly.
#
# Example:
#   scripts/wait_and_collect_review.sh native-reader-pt-arc-2025-review pt native_reader_batch_pt

set -euo pipefail

JOB_ID="$1"
LANG_CODE="$2"
ROLE="$3"
PROVIDER="${4:-fireworks_batch_devotional_gen}"
DEST="data/batch_output/${JOB_ID}"

cd "$(dirname "$0")/.."

echo "[wait_and_collect] polling ${JOB_ID}..."
.venv/bin/content-batch poll --batch-id "${JOB_ID}" --provider "${PROVIDER}"

echo "[wait_and_collect] downloading to ${DEST}..."
.venv/bin/content-batch download \
  --output-file-id "${JOB_ID}-out" \
  --dest "${DEST}" \
  --provider "${PROVIDER}"

echo "[wait_and_collect] collecting..."
.venv/bin/content-batch review-collect \
  --results "${DEST}/BIJOutputSet.jsonl" \
  --lang "${LANG_CODE}" \
  --role "${ROLE}"

echo "[wait_and_collect] done."
