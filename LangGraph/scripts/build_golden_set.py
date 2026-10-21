"""Build the critic golden set from the 2027 human-triage files (read-only).

Labels: positive = critic found a real defect; negative = critic was wrong;
excluded = subjective or undecided. Joins each finding to the run ledgers to
recover which provider(s) raised it.

    python scripts/build_golden_set.py [--out data/golden_set.jsonl]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

CHECKPOINTS = Path(__file__).resolve().parent.parent / "data" / "checkpoints"
LANGS = ("es", "pt", "en", "fr")

POSITIVE = {"applied", "applied_corrected"}
NEGATIVE = {
    "dismissed_false_positive",
    "dismissed_bad_critic_fix",
    "dismissed_critic_fix_wrong",
    # fr generic bucket: every reason recorded is "not a real error / valid style"
    "dismissed",
}


def label_for(disposition: str) -> str:
    if disposition in POSITIVE:
        return "positive"
    if disposition in NEGATIVE:
        return "negative"
    return "excluded"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_provider_index(lang: str, checkpoints: Path = CHECKPOINTS) -> dict:
    """(entry_id, field, quoted_text) -> set of provider_ids across the lang's ledgers."""
    index: dict[tuple, set] = defaultdict(set)
    paths = [*checkpoints.glob(f"2027_{lang}_*ledger.jsonl"),
             *checkpoints.glob(f"{lang}_*_2027_*ledger.jsonl")]  # e.g. pt_arc_2027_review_ledger
    for path in paths:
        for row in load_jsonl(path):
            provider = row.get("provider_id")
            if not provider:
                continue
            for f in row.get("critic_findings", []):
                index[(row["entry_id"], row["field"], f.get("quoted_text"))].add(provider)
    return index


def build_rows(lang: str, checkpoints: Path = CHECKPOINTS) -> list[dict]:
    index = build_provider_index(lang, checkpoints)
    rows = []
    for r in load_jsonl(checkpoints / f"2027_{lang}_findings_review.jsonl"):
        providers = sorted(index.get((r["entry_id"], r["field"], r["quoted_text"]), ()))
        rows.append({
            "lang": lang,
            "entry_id": r["entry_id"],
            "field": r["field"],
            "quoted_text": r["quoted_text"],
            "category": r.get("category"),
            "replacement_text": r.get("replacement_text"),
            "disposition": r["disposition"],
            "reason": r.get("reason"),
            "label": label_for(r["disposition"]),
            "applied_commit": r.get("applied_commit"),
            "providers": providers,
        })
    return rows


def summarize(rows: list[dict]) -> str:
    by_lang: dict[str, Counter] = defaultdict(Counter)
    joined: Counter = Counter()
    for r in rows:
        by_lang[r["lang"]][r["label"]] += 1
        joined[r["lang"]] += bool(r["providers"])
    lines = ["lang  total  positive  negative  excluded  precision  provider_join"]
    for lang, c in by_lang.items():
        total = sum(c.values())
        decided = c["positive"] + c["negative"]
        prec = f"{c['positive'] / decided:.0%}" if decided else "-"
        lines.append(f"{lang:<5} {total:>5} {c['positive']:>9} {c['negative']:>9} "
                     f"{c['excluded']:>9} {prec:>10} {joined[lang]:>6}/{total}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=CHECKPOINTS.parent / "golden_set.jsonl")
    args = ap.parse_args()
    rows = [r for lang in LANGS for r in build_rows(lang)]
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(summarize(rows))
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
