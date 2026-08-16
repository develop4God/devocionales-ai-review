"""
Drive the review-collect Finding output through the real graph, one (entry_id,
field) at a time — flag(seeded) -> verify -> prune -> critic -> human_confirm
-> fix -> drift_check -> validate.

Community best practice for LangGraph batch work (see PLAN.md discussion): one
thread_id per item, not one thread for the whole batch, so a crash on item #150
never touches the checkpointed state of items #1-149. A separate ledger file
(JSONL, flushed after every item) is the actual "what's done" record — the
LangGraph checkpointer only guarantees a single item's state survives a
process restart, not cross-item batch progress.

On any unhandled error the run stops immediately (no retry, no skip-and-continue)
and the failing (entry_id, field) is left out of the ledger, so a re-run picks
up from exactly that item.

flag_pass normally calls a live model (domain/flag.run_flag_pass); this run
already has the Fireworks-generated raw findings from review-collect, so
run_flag_pass is monkeypatched per invocation to return those findings instead
of re-flagging from scratch.

Validation-only for this run: validate_pass proves the fix would splice into
the real corpus file cleanly, but nothing here writes to devocionales-json.
The ledger records what *would* be applied for a separate, explicit write step.

human_confirm is not bypassed -- it still runs as a real interrupt(). This
driver supplies the resume decision itself: apply every critic_findings index
where is_valid is True. That decision is logged in the ledger next to the
critic's own reasoning, so a human can audit exactly what the critic approved
and why, after the fact.

If drift_check_pass then flags drift on the applied fix, the graph pauses at
human_confirm a second time (drift's own retry-or-stop gate). This driver
always resumes that second pause with {"apply": []} -- drift is treated as a
signal the item needs a real human look, not something to auto-retry. Those
items land in the ledger with status "drift_needs_review" and
validation_passed left null (validate_pass never ran for them this round).

Usage:
    uv run python scripts/run_review_batch_pipeline.py \
        --review-json data/reviews/review_es_20260815_022100.json \
        --corpus-file ../devocionales-json/Devocional_year_2025.json \
        --language Spanish \
        --checkpoint data/checkpoints/rvr1960_2025_review.sqlite \
        --ledger data/checkpoints/rvr1960_2025_review_ledger.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import content_batch_graph.nodes.flag_pass as flag_pass_module
from content_batch_graph.graph import compile_graph
from content_batch_graph.state import Finding
from langgraph.types import Command


def load_review_findings(review_json_path: str) -> dict[tuple[str, str], list[Finding]]:
    """(entry_id, field) -> list[Finding], skipping entries with zero findings."""
    with open(review_json_path, encoding="utf-8") as f:
        review = json.load(f)

    by_key: dict[tuple[str, str], list[Finding]] = {}
    for item in review["data"]:
        findings = item.get("findings") or []
        if not findings:
            continue
        key = (item["entry_id"], item["field"])
        by_key[key] = [
            Finding(
                quoted_text=f["quoted_text"],
                issue=f["issue"],
                category=f["category"],
                proposed_text=f.get("proposed_text"),
            )
            for f in findings
        ]
    return by_key


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


def resolve_field_path(document: dict, language_key: str, entry_id: str, field: str) -> str | None:
    """Mirrors domain/scan.py's data.{lang}.{date}.{i}.{field} convention."""
    lang_data = document.get("data", {}).get(language_key, {})
    for date, entries in lang_data.items():
        for i, entry in enumerate(entries):
            if entry.get("id") == entry_id:
                if entry.get(field, "") != "":
                    return f"data.{language_key}.{date}.{i}.{field}"
    return None


def run_one(
    graph,
    entry_id: str,
    field: str,
    findings: list[Finding],
    file_path: str,
    file_text: str,
    field_path: str,
    language: str,
) -> dict:
    thread_id = f"{entry_id}:{field}"
    config = {"configurable": {"thread_id": thread_id}}

    flag_pass_module.run_flag_pass = lambda *_a, **_k: findings

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
        raise RuntimeError(f"{thread_id}: graph did not pause at human_confirm as expected")

    critic_findings = result["__interrupt__"][0].value["critic_findings"]
    apply_indices = [i for i, cf in enumerate(critic_findings) if cf["is_valid"]]

    final = graph.invoke(Command(resume={"apply": apply_indices}), config=config)

    status = "validated_not_applied"
    if "__interrupt__" in final:
        # drift_check_pass flagged drift and re-paused at human_confirm asking
        # retry-or-stop. Always stop here -- see module docstring.
        final = graph.invoke(Command(resume={"apply": []}), config=config)
        status = "drift_needs_review"

    return {
        "entry_id": entry_id,
        "field": field,
        "thread_id": thread_id,
        "raw_findings_count": len(findings),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-json", required=True)
    parser.add_argument("--corpus-file", required=True)
    parser.add_argument("--language", required=True, help='e.g. "Spanish"')
    parser.add_argument("--language-key", default="es", help='corpus data.{key} — default "es"')
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    by_key = load_review_findings(args.review_json)
    done = load_ledger_done_keys(args.ledger)
    pending = [k for k in by_key if k not in done]

    print(f"total findings-bearing items: {len(by_key)}")
    print(f"already in ledger: {len(done)}")
    print(f"pending this run: {len(pending)}")

    if not pending:
        print("nothing to do.")
        return 0

    with open(args.corpus_file, encoding="utf-8") as f:
        document = json.load(f)

    graph, conn_cm = compile_graph(args.checkpoint)

    processed = 0
    try:
        with open(args.ledger, "a", encoding="utf-8") as ledger_f:
            for entry_id, field in pending:
                field_path = resolve_field_path(document, args.language_key, entry_id, field)
                if field_path is None:
                    raise RuntimeError(
                        f"{entry_id}:{field}: could not locate a non-empty '{field}' "
                        f"for this entry_id in {args.corpus_file}"
                    )
                parts = field_path.split(".")
                date, i = parts[2], int(parts[3])
                file_text = document["data"][args.language_key][date][i][field]

                row = run_one(
                    graph,
                    entry_id,
                    field,
                    by_key[(entry_id, field)],
                    args.corpus_file,
                    file_text,
                    field_path,
                    args.language,
                )
                ledger_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                ledger_f.flush()
                processed += 1
                print(f"[{processed}/{len(pending)}] {entry_id}:{field} -> "
                      f"{len(row['applied_indices'])}/{len(row['critic_findings'])} applied, "
                      f"validation_passed={row['validation_passed']}")
    except Exception as e:
        print(f"\nSTOPPED on error after {processed}/{len(pending)} items this run: {e}",
              file=sys.stderr)
        conn_cm.__exit__(None, None, None)
        return 1

    conn_cm.__exit__(None, None, None)
    print(f"\ndone. {processed} items processed this run, "
          f"{len(done) + processed}/{len(by_key)} total in ledger.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
