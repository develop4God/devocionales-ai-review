"""Golden-set eval: recover each labeled field's original text and plan a run.

Dry run only (no model calls): shows how many fields are recoverable, which
candidate providers are usable, and how many flag-pass calls a run would make.

    python scripts/golden_eval.py --lang fr --providers groq_qwen3_8_27b groq_qwen3_8_27b_low
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from dotenv import dotenv_values

from build_golden_set import build_rows

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=sorted(CORPUS_FILES))
    ap.add_argument("--providers", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="always on for now: no model calls are implemented yet")
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
    print("DRY RUN: no API calls made.")


if __name__ == "__main__":
    main()
