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

human_confirm is not bypassed -- it still runs as a real interrupt(). By default
(no --interactive) this driver supplies the resume decision itself: apply every
critic_findings index where is_valid is True. That decision is logged in the
ledger next to the critic's own reasoning, so a human can audit exactly what the
critic approved and why, after the fact. This is the unattended path, useful for
smoke-testing the pipeline itself.

With --interactive, the auto-decision is replaced by a real terminal prompt per
item: each critic_findings entry is printed (quoted_text, category, the critic's
is_valid verdict and reasoning, proposed replacement_text), and the human types
one of "all", "none", "1,3", or "dismiss 3,5,6" (apply everything except those
indices) to decide what to apply. This matters because the critic's own verdict
isn't always trustworthy on its stated reasoning alone -- see e.g.
marcos1124RVR1960:oracion in an earlier run, where the critic approved a fix
("crea" -> "cree") with grammatically incorrect reasoning even though a native
speaker confirmed the original text was already correct. The human's actual
decision is what gets applied and logged, not the critic's.

If drift_check_pass then flags drift on the applied fix, the graph pauses at
human_confirm a second time (drift's own retry-or-stop gate). In both auto and
--interactive mode this driver always resumes that second pause with
{"apply": []} -- drift is treated as a signal the item needs a real human look
outside this run, not something to auto-retry inline. Those items land in the
ledger with status "drift_needs_review" and validation_passed left null
(validate_pass never ran for them this round).

Usage:
    uv run python scripts/run_review_batch_pipeline.py \
        --review-json data/reviews/review_es_20260815_022100.json \
        --corpus-file ../devocionales-json/Devocional_year_2025.json \
        --language Spanish \
        --checkpoint data/checkpoints/rvr1960_2025_review.sqlite \
        --ledger data/checkpoints/rvr1960_2025_review_ledger.jsonl \
        [--interactive]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import content_batch_graph.nodes.flag_pass as flag_pass_module
from content_batch_graph.domain.prune import prune_findings
from content_batch_graph.domain.verify import verify_findings
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


def auto_decide(critic_findings: list[dict]) -> list[int]:
    """Apply every index the critic itself marked is_valid. Unattended-run default."""
    return [i for i, cf in enumerate(critic_findings) if cf["is_valid"]]


def prompt_decide(entry_id: str, field: str, critic_findings: list[dict]) -> list[int]:
    """
    Print each critic finding and read a real typed decision from the terminal.

    Accepts: "all" (every critic-valid index), "none"/empty (dismiss everything),
    "1,3" (apply exactly those indices), or "dismiss 3,5,6" (apply everything
    except those indices). Invalid input reprompts rather than guessing.
    """
    print(f"\n=== {entry_id}:{field} — {len(critic_findings)} critic finding(s) ===")
    if not critic_findings:
        print("(nothing to review)")
        return []

    for i, cf in enumerate(critic_findings):
        verdict = "VALID" if cf["is_valid"] else "REJECTED"
        print(f"  [{i}] ({cf['category']}, critic says {verdict}) "
              f"{cf['quoted_text']!r} -> {cf.get('replacement_text')!r}")
        print(f"      critic reasoning: {cf['critic_reasoning']}")

    default_apply = auto_decide(critic_findings)
    print(f"  critic-recommended apply set: {default_apply or 'none'}")

    while True:
        raw = input(
            "  apply which? ('all' / 'none' / '1,3' / 'dismiss 3,5,6') > "
        ).strip()
        if raw == "" or raw.lower() == "none":
            return []
        if raw.lower() == "all":
            return list(range(len(critic_findings)))
        if raw.lower().startswith("dismiss"):
            rest = raw[len("dismiss"):].strip()
            try:
                dismiss = {int(x) for x in rest.split(",") if x.strip()}
            except ValueError:
                print("  could not parse indices after 'dismiss' — try again.")
                continue
            if not dismiss.issubset(set(range(len(critic_findings)))):
                print(f"  index out of range 0..{len(critic_findings) - 1} — try again.")
                continue
            return [i for i in range(len(critic_findings)) if i not in dismiss]
        try:
            apply = [int(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            print("  could not parse — try again.")
            continue
        if not set(apply).issubset(set(range(len(critic_findings)))):
            print(f"  index out of range 0..{len(critic_findings) - 1} — try again.")
            continue
        return apply


def pre_filter(
    pending: list[tuple[str, str]],
    by_key: dict[tuple[str, str], list[Finding]],
    document: dict,
    language_key: str,
) -> tuple[list[tuple[str, str]], list[dict]]:
    """
    Runs verify_pass/prune_pass's own logic (no model call, no graph) against every
    pending item upfront, so the real "how many will actually reach critic" number
    is known immediately instead of discovered one slow graph invocation at a time.

    Returns (still_pending, pre_pruned_rows). still_pending is pending items that
    have at least one finding left after verify+prune -- these still go through the
    full graph. pre_pruned_rows are ready-to-write ledger rows for items where
    nothing survived verify+prune -- these never need a critic call, so the full
    graph is skipped for them entirely.
    """
    still_pending: list[tuple[str, str]] = []
    pre_pruned_rows: list[dict] = []

    for entry_id, field in pending:
        field_path = resolve_field_path(document, language_key, entry_id, field)
        if field_path is None:
            still_pending.append((entry_id, field))  # let run_one raise its own error
            continue
        parts = field_path.split(".")
        date, i = parts[2], int(parts[3])
        text = document["data"][language_key][date][i][field]

        verified, _rejected = verify_findings(by_key[(entry_id, field)], text)
        kept, _discarded = prune_findings(verified)

        if kept:
            still_pending.append((entry_id, field))
        else:
            pre_pruned_rows.append(
                {
                    "entry_id": entry_id,
                    "field": field,
                    "thread_id": f"{entry_id}:{field}",
                    "raw_findings_count": len(by_key[(entry_id, field)]),
                    "verified_count": 0,
                    "discarded_count": len(verified),
                    "critic_findings": [],
                    "applied_indices": [],
                    "fix_summary": None,
                    "drift_detected": None,
                    "drift_notes": None,
                    "validation_passed": None,
                    "validation_error": None,
                    "status": "validated_not_applied",
                }
            )

    return still_pending, pre_pruned_rows


def run_one(
    graph,
    entry_id: str,
    field: str,
    findings: list[Finding],
    file_path: str,
    file_text: str,
    field_path: str,
    language: str,
    interactive: bool,
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
    apply_indices = (
        prompt_decide(entry_id, field, critic_findings)
        if interactive
        else auto_decide(critic_findings)
    )

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
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt at the terminal for each item's apply/dismiss decision "
        "instead of auto-applying every critic-valid finding.",
    )
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

    pending, pre_pruned_rows = pre_filter(pending, by_key, document, args.language_key)
    print(f"pre-filtered (verify+prune, no model call): "
          f"{len(pre_pruned_rows)} fully pruned, {len(pending)} will reach critic")

    if pre_pruned_rows:
        with open(args.ledger, "a", encoding="utf-8") as ledger_f:
            for row in pre_pruned_rows:
                ledger_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            ledger_f.flush()
        print(f"wrote {len(pre_pruned_rows)} pre-pruned rows directly to ledger.")

    if not pending:
        print("nothing left to run through the graph.")
        return 0

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
                    args.interactive,
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
