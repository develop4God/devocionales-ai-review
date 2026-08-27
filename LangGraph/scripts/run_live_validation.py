"""
Drive the real graph live, over every (entry_id, field) in a corpus file, with no
pre-collected findings -- flag_pass calls a real model per item (unlike
run_review_batch_pipeline.py, which replays Fireworks-batch-collected findings via
a monkeypatched run_flag_pass). Built for validating a freshly generated file (e.g.
a new devotional year) against a live sync provider (Groq) before any batch step
exists for it.

Same ledger/checkpoint/thread-per-item pattern as run_review_batch_pipeline.py: one
thread_id per item so a crash never touches earlier items' checkpointed state, a
JSONL ledger flushed after every item as the real "what's done" record, and a
resumable re-run driven by load_ledger_done_keys.

human_confirm is not bypassed -- it still runs as a real interrupt(); this driver
auto-resolves it the same way run_review_batch_pipeline.py's --interactive=False
path does: apply every critic_findings index the critic itself marked is_valid.

Stops cleanly (not a crash) on a 429 rate-limit/quota error from the provider,
since the whole point of this driver is running until a real quota wall, keeping
whatever's already in the ledger for a later resume once the quota resets.

Usage:
    uv run python scripts/run_live_validation.py \
        --corpus-file /path/to/Devocional_year_2027_es.json \
        --language Spanish \
        --language-key es \
        --fields reflexion,oracion \
        --checkpoint data/checkpoints/2027_es_live_validation.sqlite \
        --ledger data/checkpoints/2027_es_live_validation_ledger.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import openai
from langgraph.types import Command

from content_batch_graph.graph import compile_graph


def iter_entry_fields(document: dict, language_key: str, fields: list[str]):
    """Yields (entry_id, field, field_path, text) for every entry/field where the
    field is present and non-empty, in corpus (date, index) order."""
    lang_data = document.get("data", {}).get(language_key, {})
    for date in sorted(lang_data):
        for i, entry in enumerate(lang_data[date]):
            entry_id = entry.get("id")
            if not entry_id:
                continue
            for field in fields:
                text = entry.get(field, "")
                if not text:
                    continue
                field_path = f"data.{language_key}.{date}.{i}.{field}"
                yield entry_id, field, field_path, text


def load_ledger_done_keys(ledger_path: str) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    p = Path(ledger_path)
    if not p.exists():
        return done
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((row["entry_id"], row["field"]))
    return done


def auto_decide(critic_findings: list[dict]) -> list[int]:
    return [i for i, cf in enumerate(critic_findings) if cf["is_valid"]]


def run_one(
    graph,
    entry_id: str,
    field: str,
    file_path: str,
    file_text: str,
    field_path: str,
    language: str,
) -> dict:
    thread_id = f"{entry_id}:{field}"
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(
        {
            "file_path": file_path,
            "file_text": file_text,
            "field_path": field_path,
            "language": language,
            "entry_id": entry_id,
        },
        config=config,
    )

    if "__interrupt__" not in result:
        raise RuntimeError(
            f"{thread_id}: graph did not pause at human_confirm as expected"
        )

    critic_findings = result["__interrupt__"][0].value["critic_findings"]
    apply_indices = auto_decide(critic_findings)

    final = graph.invoke(Command(resume={"apply": apply_indices}), config=config)

    status = "validated_not_applied"
    if "__interrupt__" in final:
        final = graph.invoke(Command(resume={"apply": []}), config=config)
        status = "drift_needs_review"

    return {
        "entry_id": entry_id,
        "field": field,
        "thread_id": thread_id,
        "raw_findings_count": len(result.get("raw_findings", [])),
        "verified_count": len(final.get("verified_findings", [])),
        "discarded_count": len(final.get("discarded_findings", [])),
        "critic_findings": [
            {
                "quoted_text": cf["quoted_text"],
                "category": cf["category"],
                "is_valid": cf["is_valid"],
                "replacement_text": cf.get("replacement_text"),
                "critic_reasoning": cf["critic_reasoning"],
            }
            for cf in critic_findings
        ],
        "applied_indices": apply_indices,
        "fix_summary": final.get("fix_summary"),
        "drift_detected": final.get("drift_detected"),
        "drift_notes": final.get("drift_notes"),
        "validation_passed": final.get("validation_passed"),
        "validation_error": final.get("validation_error"),
        "status": status,
    }


def is_rate_limit_error(e: Exception) -> bool:
    if isinstance(e, openai.RateLimitError):
        return True
    if isinstance(e, openai.BadRequestError):
        code = e.body.get("code") if isinstance(e.body, dict) else None
        return code in {"rate_limit_exceeded", "quota_exceeded"}
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-file", required=True)
    parser.add_argument("--language", required=True, help='e.g. "Spanish"')
    parser.add_argument("--language-key", default="es")
    parser.add_argument("--fields", default="reflexion,oracion")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]

    with open(args.corpus_file, encoding="utf-8") as f:
        document = json.load(f)

    all_items = list(iter_entry_fields(document, args.language_key, fields))
    done = load_ledger_done_keys(args.ledger)
    pending = [item for item in all_items if (item[0], item[1]) not in done]

    print(f"total (entry_id, field) items in corpus: {len(all_items)}")
    print(f"already in ledger: {len(done)}")
    print(f"pending this run: {len(pending)}")

    if not pending:
        print("nothing to do.")
        return 0

    graph, conn_cm = compile_graph(args.checkpoint)

    processed = 0
    try:
        with open(args.ledger, "a", encoding="utf-8") as ledger_f:
            for entry_id, field, field_path, text in pending:
                try:
                    row = run_one(
                        graph,
                        entry_id,
                        field,
                        args.corpus_file,
                        text,
                        field_path,
                        args.language,
                    )
                except (openai.RateLimitError, openai.BadRequestError) as e:
                    if is_rate_limit_error(e):
                        print(
                            f"\nSTOPPED on rate limit/quota after {processed}/{len(pending)} "
                            f"items this run: {e}",
                            file=sys.stderr,
                        )
                        conn_cm.__exit__(None, None, None)
                        return 1
                    raise

                ledger_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                ledger_f.flush()
                processed += 1
                print(
                    f"[{processed}/{len(pending)}] {entry_id}:{field} -> "
                    f"{len(row['applied_indices'])}/{len(row['critic_findings'])} applied, "
                    f"validation_passed={row['validation_passed']}"
                )
    except Exception as e:  # noqa: BLE001 -- deliberate: any unhandled error stops
        # the run cleanly (ledger flushed, checkpoint closed) rather than crashing
        # mid-write; run_review_batch_pipeline.py uses the same top-level catch-all
        # for the same reason. Rate-limit/quota errors are already handled above
        # and never reach here.
        print(
            f"\nSTOPPED on error after {processed}/{len(pending)} items this run: {e}",
            file=sys.stderr,
        )
        conn_cm.__exit__(None, None, None)
        return 1

    conn_cm.__exit__(None, None, None)
    print(
        f"\ndone. {processed} items processed this run, "
        f"{len(done) + processed}/{len(all_items)} total in ledger."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
