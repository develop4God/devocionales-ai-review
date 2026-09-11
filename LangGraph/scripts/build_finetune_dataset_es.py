"""
Build the ES fine-tuning dataset from verified sources.

Source of truth for 2027 findings: the patched ledger at
LangGraph/data/checkpoints/2027_es_apply_ledger.jsonl, which carries a
`finding_category` field per finding (see patch_2027_es_ledger_finding_categories.py).
That field is authoritative -- this script does not re-derive categories from
is_valid/drift_detected on its own.

Source of truth for the 2025/2026 base-file findings: hand-reviewed commit diffs,
encoded in finetune_dataset_corrections.py (a separate manual review, since those
commits predate this ledger schema and never went through the graph's ledger at all).

Categories produced (see conversation record for full definitions):
  1. true_positive
  2. true_negative_clean
  3. hard_negative_false_positive
  4. hard_negative_bad_fix
  5. hard_negative_partial_fix

`hold_for_recheck` findings are excluded from the output entirely until a human
resolves them -- they are neither positive nor negative examples yet.

Output: one JSON file with all raw records (tier-labeled, snippet-level).
This is NOT yet the Unsloth/Alpaca training schema -- that conversion, and the
full-text backfill, are separate follow-up steps.
"""
import json
import subprocess
import difflib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_2027 = REPO_ROOT / "LangGraph" / "data" / "checkpoints" / "2027_es_apply_ledger.jsonl"
REVIEW_ES_2025 = REPO_ROOT / "LangGraph" / "data" / "reviews" / "review_es_20260815_022100.json"
DEVOCIONALES_JSON_REPO = REPO_ROOT.parent / "devocionales-json"

OUTPUT = REPO_ROOT / "LangGraph" / "data" / "finetune" / "es_dataset_raw.json"


def from_2027_ledger():
    records = []
    with open(LEDGER_2027, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f if l.strip()]

    for entry in lines:
        for cf in entry.get("critic_findings", []):
            category = cf.get("finding_category")
            if category is None or category == "hold_for_recheck":
                continue

            record = {
                "source": f"ledger_2027:{entry['entry_id']}:{entry['field']}",
                "tier": category,
                "critic_category": cf.get("category"),
                "quoted_text": cf.get("quoted_text"),
                "reasoning": cf.get("critic_reasoning"),
            }
            if category == "true_positive":
                record["proposed_fix"] = cf.get("replacement_text")
            elif category == "hard_negative_bad_fix":
                record["rejected_fix"] = cf.get("replacement_text")
                record["drift_notes"] = entry.get("drift_notes")
            records.append(record)
    return records


def from_2025_2026_base_file_commits():
    """Hand-reviewed commit diffs -- see finetune_dataset_corrections.py for the
    drop/recategorize/hold decisions applied here."""
    from finetune_dataset_corrections import classify

    def get_pairs(commit, files):
        out = subprocess.run(
            ["git", "show", commit, "--"] + files,
            capture_output=True, text=True, cwd=str(DEVOCIONALES_JSON_REPO),
        ).stdout
        lines = out.split("\n")
        pairs = []
        i = 0
        while i < len(lines):
            if lines[i].startswith("-") and not lines[i].startswith("---"):
                old = lines[i][1:]
                if i + 1 < len(lines) and lines[i + 1].startswith("+") and not lines[i + 1].startswith("+++"):
                    new = lines[i + 1][1:]
                    if old != new:
                        pairs.append((old, new))
                    i += 2
                    continue
            i += 1
        return pairs

    def word_diff_span(old, new, pad=4):
        ow, nw = old.split(), new.split()
        sm = difflib.SequenceMatcher(None, ow, nw)
        spans = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                spans.append((" ".join(ow[max(0, i1 - pad):i2 + pad]), " ".join(nw[max(0, j1 - pad):j2 + pad])))
        return spans

    commits = {
        "d5e83763": (["Devocional_year_2025.json", "Devocional_year_2026.json"], "grammar"),
        "1b37632d": (["Devocional_year_2025.json", "Devocional_year_2026.json"], "grammar"),
        "279f74f": (["Devocional_year_2025.json"], "grammar_typo"),
    }

    records = []
    for commit, (files, category) in commits.items():
        for old, new in get_pairs(commit, files):
            for o, n in word_diff_span(old, new):
                base_record = {
                    "source": f"commit:{commit}",
                    "critic_category": category,
                    "quoted_text": o,
                    "proposed_fix": n,
                }
                action, detail = classify(base_record)
                if action == "drop":
                    continue
                if action == "hold":
                    continue
                if action == "recategorize":
                    base_record["critic_category"] = detail
                base_record["tier"] = "true_positive"
                records.append(base_record)

    # the known partial-fix / ripple case, documented by hand (see conversation record)
    records.append({
        "source": "commit:d5e83763_then_279f74f",
        "tier": "hard_negative_partial_fix",
        "critic_category": "agreement_ripple",
        "quoted_text": "a la verdadera discipulado",
        "bad_fix": "a la verdadero discipulado",
        "proposed_fix": "al verdadero discipulado",
        "reasoning": (
            "Changing only the adjective's gender without updating the preceding "
            "article leaves a new agreement error. The full span must be re-checked "
            "after any edit, not just the flagged token."
        ),
    })
    return records


def from_clean_review_sample(n=150, seed=1):
    import random
    d = json.load(open(REVIEW_ES_2025, encoding="utf-8"))
    clean = [r for r in d["data"] if not r.get("findings")]
    random.seed(seed)
    sample = random.sample(clean, min(n, len(clean)))
    return [
        {
            "source": f"review_es_20260815:{r['entry_id']}:{r['field']}",
            "tier": "true_negative_clean",
            "entry_id": r["entry_id"],
            "field": r["field"],
        }
        for r in sample
    ]


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    records = []
    records += from_2027_ledger()
    records += from_2025_2026_base_file_commits()
    records += from_clean_review_sample()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    from collections import Counter
    counts = Counter(r["tier"] for r in records)
    print("Wrote", len(records), "records to", OUTPUT)
    print("Tier counts:", dict(counts))


if __name__ == "__main__":
    main()
