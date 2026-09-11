"""
Convert the ES fine-tune dataset (5-tier, full-text-backfilled) into Alpaca
format (instruction/input/output) for Unsloth SFT training.

Design choices, so a future re-run or a PT/FR port doesn't have to re-derive them:

- `input` is the full field text when available (true_positive/false_positive/
  bad_fix/clean records from the ledger or 2025 review), falling back to the
  wider quoted_text window for commit-diff records where no full corpus lookup
  was done. Training on the whole passage, not an isolated snippet, is the
  point of the full-text backfill step -- see backfill_finetune_dataset_fulltext.py.

- `output` is one consistent JSON shape across all tiers, matching the actual
  fields LangGraph's critic.py/CriticFinding already produces (is_valid,
  category, quoted_text, replacement_text) -- so a model trained on this can
  slot into the existing pipeline's expected output shape without a translation
  layer:
    {"has_error": bool, "category": str, "quoted_text": str|null,
     "proposed_fix": str|null, "reasoning": str}

- hard_negative_bad_fix and hard_negative_partial_fix records teach a second,
  harder lesson beyond has_error: even when has_error is true, the *proposed
  fix itself* must be checked. Their `output` marks the rejected fix explicitly
  in reasoning so the correct chosen behavior (flag it, but with the RIGHT
  fix, or defer to human review) is visible in the training example -- not
  just "yes/no is this an error".

- hold_for_recheck records (7, from the ledger patch) are already excluded
  upstream and never reach this script.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "LangGraph" / "data" / "finetune" / "es_dataset_fulltext.json"
OUTPUT = REPO_ROOT / "LangGraph" / "data" / "finetune" / "es_dataset_alpaca.json"

INSTRUCTION = (
    "Review this Spanish devotional text for typos, grammar errors, or awkward "
    "phrasing. If you find an issue, verify your proposed fix doesn't introduce "
    "a new error elsewhere in the sentence (agreement, missing words, "
    "duplication) before reporting it. If the text is already correct, or a "
    "flagged word is simply uncommon rather than wrong, report no error."
)


def get_input_text(r):
    if r.get("full_text"):
        return r["full_text"]
    return r.get("quoted_text", "")


def build_output(r):
    tier = r["tier"]

    if tier == "true_negative_clean":
        return {
            "has_error": False,
            "category": "none",
            "quoted_text": None,
            "proposed_fix": None,
            "reasoning": "No typo, grammar, or phrasing error found in this text.",
        }

    if tier == "true_positive":
        return {
            "has_error": True,
            "category": r.get("critic_category", "typo"),
            "quoted_text": r.get("quoted_text"),
            "proposed_fix": r.get("proposed_fix"),
            "reasoning": r.get("reasoning") or "Correction verified against the live corpus.",
        }

    if tier == "hard_negative_false_positive":
        return {
            "has_error": False,
            "category": "none",
            "quoted_text": None,
            "proposed_fix": None,
            "reasoning": (
                f"The span {r.get('quoted_text')!r} was flagged as a possible "
                f"{r.get('critic_category')} issue, but it is already correct "
                f"Spanish. {r.get('reasoning') or ''}".strip()
            ),
        }

    if tier == "hard_negative_bad_fix":
        return {
            "has_error": True,
            "category": r.get("critic_category", "grammar"),
            "quoted_text": r.get("quoted_text"),
            # deliberately null: the only fix on record for this span was proven
            # wrong (rejected_fix) -- teach "flag it, but do not propose that fix"
            "proposed_fix": None,
            "reasoning": (
                f"This span may need correction, but the previously proposed fix "
                f"{r.get('rejected_fix')!r} introduced a new error and was "
                f"reverted: {r.get('drift_notes') or r.get('reasoning') or ''}. "
                "Do not repeat that fix; this needs a human-verified correction."
            ).strip(),
        }

    if tier == "hard_negative_partial_fix":
        return {
            "has_error": True,
            "category": r.get("critic_category", "agreement_ripple"),
            "quoted_text": r.get("quoted_text"),
            "proposed_fix": r.get("proposed_fix"),
            "reasoning": (
                f"A partial fix ({r.get('bad_fix')!r}) previously corrected only "
                f"part of this span and left a new agreement error in place. "
                f"{r.get('reasoning') or ''} The correct fix must address the "
                "full affected span, not just the originally flagged word."
            ).strip(),
        }

    raise ValueError(f"Unknown tier: {tier}")


def main():
    records = json.load(open(INPUT, encoding="utf-8"))

    alpaca = []
    skipped = 0
    for r in records:
        input_text = get_input_text(r)
        if not input_text:
            skipped += 1
            continue
        alpaca.append({
            "instruction": INSTRUCTION,
            "input": input_text,
            "output": json.dumps(build_output(r), ensure_ascii=False),
            "_tier": r["tier"],  # kept for dataset auditing; strip before training if the trainer complains
            "_source": r["source"],
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(alpaca, f, ensure_ascii=False, indent=2)

    from collections import Counter
    print(f"Wrote {len(alpaca)} Alpaca records to {OUTPUT} ({skipped} skipped, no input text)")
    print("Tier counts:", dict(Counter(r["_tier"] for r in alpaca)))


if __name__ == "__main__":
    main()
