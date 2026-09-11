"""
Backfill full field text into the ES fine-tuning dataset.

The raw dataset (build_finetune_dataset_es.py output) carries only isolated
quoted_text snippets for most records -- useful for the manual review pass, but
not for training: a model trained on isolated fragments learns to react to a
narrow trigger instead of reviewing a whole passage in context (the same
shortcut-learning risk this whole project is trying to avoid).

This script adds a `full_text` field to every record, pulled from the actual
corpus file the record's entry_id/source points to:
  - ledger_2027:<entry_id>:<field>        -> DevocionalesAPI 2027 ES seed file
  - commit:<hash>                          -> already carries a wide quoted_text
                                               window (see build script's pad=4);
                                               left as-is, no corpus lookup needed
  - review_es_20260815:<entry_id>:<field>  -> devocionales-json 2025 base file

Records whose corpus entry can't be found are marked full_text: null and kept
in the output with `full_text_missing: true` so they can be filtered before
training rather than silently dropped.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "LangGraph" / "data" / "finetune" / "es_dataset_raw.json"
OUTPUT = REPO_ROOT / "LangGraph" / "data" / "finetune" / "es_dataset_fulltext.json"

CORPUS_2027 = Path("/home/develop4god/python/DevocionalesAPI/seed_generation/2027/yearly_devotionals/ES/Devocional_year_2027_es.json")
CORPUS_2025 = REPO_ROOT.parent / "devocionales-json" / "Devocional_year_2025.json"
CORPUS_2026 = REPO_ROOT.parent / "devocionales-json" / "Devocional_year_2026.json"


def build_index(path):
    corpus = json.load(open(path, encoding="utf-8"))
    es = corpus["data"]["es"]
    by_id = {}
    for _date, entries in es.items():
        for e in entries:
            by_id[e["id"]] = e
    return by_id


def main():
    index_2027 = build_index(CORPUS_2027)
    index_2025 = build_index(CORPUS_2025)
    index_2026 = build_index(CORPUS_2026)

    records = json.load(open(INPUT, encoding="utf-8"))

    filled, missing, skipped_commit = 0, 0, 0
    for r in records:
        source = r.get("source", "")

        if source.startswith("ledger_2027:"):
            _, entry_id, field = source.split(":")
            entry = index_2027.get(entry_id)
        elif source.startswith("review_es_20260815:"):
            _, entry_id, field = source.split(":")
            entry = index_2025.get(entry_id) or index_2026.get(entry_id)
        elif source.startswith("commit:"):
            # already has a wide word-window in quoted_text from the diff extraction
            skipped_commit += 1
            r["full_text"] = None
            r["full_text_missing"] = False
            r["full_text_note"] = "commit-diff record already carries a wide context window in quoted_text"
            continue
        else:
            entry = None
            field = None

        if entry is not None and field in entry:
            r["full_text"] = entry[field]
            r["full_text_missing"] = False
            filled += 1
        else:
            r["full_text"] = None
            r["full_text_missing"] = True
            missing += 1

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} records to {OUTPUT}")
    print(f"full_text filled: {filled}, missing: {missing}, commit-records (context already wide): {skipped_commit}")


if __name__ == "__main__":
    main()
