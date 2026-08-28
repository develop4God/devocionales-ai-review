#!/usr/bin/env bash
# Dynamic multi-provider dispatcher for run_live_validation.py.
#
# Problem this replaces: run_live_validation.py's own --shard i/N partitions the
# full corpus item list into fixed slices at launch. If a provider dies mid-run
# (daily quota, as Groq's TPD cap does most days on this project), its shard's
# remaining items just sit there -- someone has to notice, compute which shard
# number is stalled, and manually launch a replacement worker on a fresh
# provider config pointed at that same shard number. That manual reassignment,
# not the ledger's already-correct skip-done logic, is what this script
# automates.
#
# How: no shard flag is ever passed to run_live_validation.py. Every worker in
# the pool is invoked with the SAME full pending list (no --shard), reading the
# SAME shared --ledger. run_live_validation.py's own pending computation
# (main(), scripts/run_live_validation.py) already reads the ledger fresh at
# each of ITS invocations -- so this loop's job is just: keep re-launching one
# worker per still-healthy provider config, in a round-robin, each pass, until
# the ledger reports every item done. A provider that dies on daily quota
# (run_live_validation.py returns exit code 1 and logs "STOPPED on daily
# quota") is dropped from the pool for the rest of this script's run, since a
# daily cap won't clear before this loop would give up anyway; a stall-safe
# limit still applies for any other reason progress stops.
#
# This is intentionally a serial-per-config loop, not a background-process
# supervisor: run_live_validation.py only writes to the ledger from inside a
# single try/finally-flushed block per item, and two OS processes racing to
# read the *same, still-shared* full pending list at truly the same instant
# (both computing "not yet in ledger" before either has written its first row)
# could pick the same first item and both spend a real call on it -- wasteful
# but not corrupting, since the ledger just gets one duplicate row for that
# item, not a broken file. Running configs one pass at a time avoids paying
# that waste at all, at the cost of not literally overlapping API calls across
# providers within the same instant. If true concurrent overlap is wanted
# later, that's a different, larger design (a real work queue) -- flag it
# separately rather than building it speculatively here.
#
# Usage: run_live_validation_multi_provider.sh <total_items> <corpus_file> \
#          <language> <language_key> <ledger_path> <checkpoint_prefix> \
#          <config1> [config2] [config3] ...
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -lt 7 ]; then
  echo "usage: $0 <total_items> <corpus_file> <language> <language_key> <ledger_path> <checkpoint_prefix> <config1> [config2 ...]" >&2
  exit 2
fi

TOTAL="$1"; CORPUS="$2"; LANGUAGE="$3"; LANG_KEY="$4"; LEDGER="$5"; CKPT_PREFIX="$6"
shift 6
CONFIGS=("$@")

MAX_STALLS=5
STALL_COUNT=0
ALREADY_DONE=0

ledger_done_count() {
  python3 -c "
import json
seen=set()
try:
    with open('$LEDGER') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            d=json.loads(line)
            seen.add((d['entry_id'], d['field']))
except FileNotFoundError:
    pass
print(len(seen))
"
}

while true; do
  DONE=$(ledger_done_count)
  echo "=== ledger has $DONE/$TOTAL (entry_id, field) rows done ==="
  if [ "$DONE" -ge "$TOTAL" ]; then
    echo "all done."
    exit 0
  fi
  if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "no healthy provider configs left in the pool -- stopping." >&2
    exit 1
  fi
  if [ "$DONE" -eq "$ALREADY_DONE" ]; then
    STALL_COUNT=$((STALL_COUNT + 1))
    echo "no progress this pass (still $DONE), stall $STALL_COUNT/$MAX_STALLS"
    if [ "$STALL_COUNT" -ge "$MAX_STALLS" ]; then
      echo "stalled $MAX_STALLS passes in a row -- stopping to avoid a silent infinite loop." >&2
      exit 1
    fi
  else
    STALL_COUNT=0
  fi
  ALREADY_DONE="$DONE"

  REMAINING_CONFIGS=()
  for CFG in "${CONFIGS[@]}"; do
    DONE=$(ledger_done_count)
    if [ "$DONE" -ge "$TOTAL" ]; then
      break
    fi
    NAME=$(basename "$CFG" .yml)
    echo "--- dispatching to $NAME ---"
    if .venv/bin/python scripts/run_live_validation.py \
      --corpus-file "$CORPUS" --language "$LANGUAGE" --language-key "$LANG_KEY" \
      --checkpoint "${CKPT_PREFIX}_${NAME}.sqlite" \
      --ledger "$LEDGER" --config "$CFG"; then
      REMAINING_CONFIGS+=("$CFG")
    else
      echo "  $NAME exited non-zero (daily quota or error) -- dropping from pool for this run." >&2
    fi
  done
  CONFIGS=("${REMAINING_CONFIGS[@]}")

  sleep 5
done
