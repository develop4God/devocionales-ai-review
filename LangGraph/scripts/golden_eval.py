"""Golden-set eval: recover each labeled field's original text and plan a run.

Dry run only (no model calls): shows how many fields are recoverable, which
candidate providers are usable, and how many flag-pass calls a run would make.

    python scripts/golden_eval.py --lang fr --providers groq_qwen3_8_27b groq_qwen3_8_27b_low
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import openai
import yaml
from dotenv import dotenv_values

from build_golden_set import build_rows
from run_live_validation import classify_rate_limit_error, parse_retry_after_seconds

LANGGRAPH = Path(__file__).resolve().parent.parent
API_REPO = Path.home() / "python" / "DevocionalesAPI"
YEARLY = "seed_generation/2027/yearly_devotionals"
# Current (post-fix) 2027 corpus file per language, relative to API_REPO/YEARLY.
CORPUS_FILES = {
    "es": "ES/Devocional_year_2027_es.json",
    "pt": "PT/Devocional_year_2027_pt_ARC.json",
    "en": "EN/Devocional_year_2027_en_KJV.json",
    "fr": "FR/Devocional_year_2027_fr_LSG1910.json",
}
LANG_NAMES = {"es": "Spanish", "pt": "Portuguese", "en": "English", "fr": "French", "de": "German"}
# Entry ids drift between triage files and corpus ("1Pedro16-7" vs "1Pedro167");
# the trailing YYYYMMDD is stable, so index by it.
DATE_LEN = 8


def index_entries(document: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}

    def walk(o):
        if isinstance(o, dict):
            if "id" in o:
                out[o["id"][-DATE_LEN:]] = o
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(document)
    return out


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def prefix_entries(repo: Path, commit: str, cache: dict) -> dict[str, dict]:
    """Entries of the 2027 corpus file(s) as they were just before `commit` (read-only)."""
    if commit not in cache:
        ents: dict[str, dict] = {}
        names = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit).stdout.split()
        for name in (n for n in names if "2027" in n and n.endswith(".json")):
            shown = _git(repo, "show", f"{commit}^:{name}")
            if shown.returncode == 0:
                for k, v in index_entries(json.loads(shown.stdout)).items():
                    ents.setdefault(k, v)
        cache[commit] = ents
    return cache[commit]


def recover_fields(lang: str, repo: Path = API_REPO) -> tuple[dict, Counter]:
    """(date, field) -> {"text", "rows"} for recoverable labeled rows; plus drop reasons."""
    current = index_entries(json.loads((repo / YEARLY / CORPUS_FILES[lang]).read_text("utf-8")))
    cache: dict = {}
    drops: Counter = Counter()
    by_field: dict[tuple, list] = defaultdict(list)
    for r in build_rows(lang):
        if r["label"] != "excluded":
            by_field[(r["entry_id"][-DATE_LEN:], r["field"])].append(r)

    fields: dict = {}
    for (date, field), rows in by_field.items():
        text = None
        for r in rows:
            if r["label"] == "positive" and r["applied_commit"]:
                entry = prefix_entries(repo, r["applied_commit"], cache).get(date)
                if entry and entry.get(field):
                    text = entry[field]
                    break
        if text is None and date in current:
            text = current[date].get(field)
        if not text:
            drops["no_text"] += len(rows)
            continue
        kept = [r for r in rows if r["quoted_text"] in text]
        drops["quote_not_in_text"] += len(rows) - len(kept)
        if kept:
            fields[(date, field)] = {"text": text, "rows": kept}
    return fields, drops


def check_providers(ids: list[str]) -> list[tuple[str, str]]:
    config = yaml.safe_load((LANGGRAPH / "config" / "providers.yml").read_text("utf-8"))
    by_id = {p["id"]: p for p in config["providers"]}
    env = dotenv_values(LANGGRAPH / ".env")
    out = []
    for pid in ids:
        p = by_id.get(pid)
        if not p:
            out.append((pid, "NOT IN providers.yml"))
        elif not p.get("enabled", True):
            out.append((pid, "disabled"))
        elif p.get("client_type") == "api" and not env.get(p.get("env_var", "")):
            out.append((pid, f"missing key {p.get('env_var')}"))
        else:
            effort = (p.get("extra_body") or {}).get("reasoning_effort", "unset")
            out.append((pid, f"ok (model={p['model']}, reasoning_effort={effort})"))
    return out


def quotes_match(a: str, b: str) -> bool:
    """A finding hits a labeled span if either quote contains the other (whitespace-insensitive)."""
    a, b = " ".join(a.split()), " ".join(b.split())
    return bool(a and b and (a in b or b in a))


def score_field(rows: list[dict], finding_quotes: list[str]) -> dict:
    """Per-field outcome: which labeled rows the findings hit, and findings matching no label."""
    hit_pos = [r for r in rows if r["label"] == "positive" and any(quotes_match(q, r["quoted_text"]) for q in finding_quotes)]
    hit_neg = [r for r in rows if r["label"] == "negative" and any(quotes_match(q, r["quoted_text"]) for q in finding_quotes)]
    unlabeled = [q for q in finding_quotes if not any(quotes_match(q, r["quoted_text"]) for r in rows)]
    return {"tp": len(hit_pos), "fp": len(hit_neg), "unlabeled": unlabeled}


def run_live(fields: dict, lang: str, provider_id: str, role_id: str, out_path: Path) -> None:
    """Flag pass + verify per field (no critic), appending one resumable JSONL row each."""
    from content_batch_graph.domain.flag import run_flag_pass
    from content_batch_graph.domain.verify import verify_findings

    done = set()
    if out_path.exists():
        done = {(r["date"], r["field"]) for r in map(json.loads, out_path.read_text("utf-8").splitlines()) if r.get("provider_id") == provider_id and r.get("role_id") == role_id}
    pending = [(k, v) for k, v in fields.items() if k not in done]
    with out_path.open("a", encoding="utf-8") as out:
        for n, ((date, field), item) in enumerate(pending, 1):
            while True:
                try:
                    raw = run_flag_pass(item["text"], LANG_NAMES[lang], role_id=role_id, provider_id=provider_id)
                    break
                except (openai.RateLimitError, openai.BadRequestError) as e:
                    kind = classify_rate_limit_error(e)
                    if kind is None:
                        raise
                    if kind == "daily_quota":
                        print(f"STOPPED on daily quota after {n - 1}/{len(pending)}: {e}", file=sys.stderr)
                        return
                    time.sleep(parse_retry_after_seconds(e))
            verified, rejected = verify_findings(raw, item["text"])
            quotes = [v["quoted_text"] for v in verified]
            row = {"lang": lang, "date": date, "field": field, "provider_id": provider_id, "role_id": role_id,
                   "raw": len(raw), "verified_quotes": quotes, "rejected": len(rejected),
                   **score_field(item["rows"], quotes)}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{n}/{len(pending)}] {date}:{field} verified={len(quotes)} tp={row['tp']} fp={row['fp']}")


def summarize_run(out_path: Path, fields: dict, provider_id: str, role_id: str) -> str:
    rows = [r for r in map(json.loads, out_path.read_text("utf-8").splitlines())
            if r["provider_id"] == provider_id and r["role_id"] == role_id]
    npos = sum(1 for f in fields.values() for r in f["rows"] if r["label"] == "positive")
    tp, fp = sum(r["tp"] for r in rows), sum(r["fp"] for r in rows)
    unl = sum(len(r["unlabeled"]) for r in rows)
    prec = f"{tp / (tp + fp):.0%}" if tp + fp else "-"
    return (f"{provider_id} / {role_id}: fields={len(rows)} TP={tp} FP={fp} unlabeled={unl} "
            f"precision={prec} recall={tp}/{npos}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=sorted(CORPUS_FILES))
    ap.add_argument("--providers", nargs="+", required=True)
    ap.add_argument("--live", action="store_true",
                    help="make real model calls (default is a dry run); one provider only")
    ap.add_argument("--role", default="native_reader_batch")
    ap.add_argument("--out", type=Path, help="results JSONL (required with --live)")
    args = ap.parse_args()

    fields, drops = recover_fields(args.lang)
    labels = Counter(r["label"] for f in fields.values() for r in f["rows"])
    print(f"[{args.lang}] recoverable fields: {len(fields)}  "
          f"rows: positive={labels['positive']} negative={labels['negative']}")
    print(f"dropped rows: {dict(drops)}")
    print("providers:")
    for pid, status in check_providers(args.providers):
        print(f"  {pid}: {status}")
    print(f"planned flag-pass calls: {len(fields)} x {len(args.providers)} = "
          f"{len(fields) * len(args.providers)} (+ critic calls per finding, unknown until run)")
    if not args.live:
        print("DRY RUN: no API calls made.")
        return
    if len(args.providers) != 1 or not args.out:
        raise SystemExit("--live needs exactly one --providers id and --out")
    run_live(fields, args.lang, args.providers[0], args.role, args.out)
    print(summarize_run(args.out, fields, args.providers[0], args.role))


if __name__ == "__main__":
    main()
