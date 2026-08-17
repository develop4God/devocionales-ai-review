#!/usr/bin/env bash
# Repeatedly resumes run_review_batch_pipeline.py against the PT ARC batch,
# since Groq's free-tier gpt-oss-20b TPM cap (8000/min) causes the driver to
# stop-on-error every few items. The driver itself is resume-safe (ledger is
# the source of truth for what's done); this loop just re-invokes it after a
# short pause until every item is processed.
set -euo pipefail
cd "$(dirname "$0")/.."

TOTAL=264
ALREADY_DONE=0
STALL_COUNT=0
MAX_STALLS=5

while true; do
  DONE=$(wc -l < data/checkpoints/pt_arc_2025_review_ledger.jsonl 2>/dev/null || echo 0)
  echo "=== ledger has $DONE/$TOTAL rows ==="
  if [ "$DONE" -ge "$TOTAL" ]; then
    echo "all done."
    break
  fi
  if [ "$DONE" -eq "$ALREADY_DONE" ]; then
    STALL_COUNT=$((STALL_COUNT + 1))
    echo "no progress this iteration (still $DONE), stall $STALL_COUNT/$MAX_STALLS"
    if [ "$STALL_COUNT" -ge "$MAX_STALLS" ]; then
      echo "stalled $MAX_STALLS times in a row -- stopping to avoid a silent infinite loop."
      exit 1
    fi
  else
    STALL_COUNT=0
  fi
  ALREADY_DONE=$DONE

  uv run python scripts/run_review_batch_pipeline.py \
    --review-json data/reviews/review_pt_20260817_010829.json \
    --corpus-file /home/develop4god/Projects/devocionales-json/Devocional_year_2025_pt_ARC.json \
    --language "Brazilian Portuguese" \
    --language-key pt \
    --checkpoint data/checkpoints/pt_arc_2025_review.sqlite \
    --ledger data/checkpoints/pt_arc_2025_review_ledger.jsonl \
    || true

  sleep 15
done
