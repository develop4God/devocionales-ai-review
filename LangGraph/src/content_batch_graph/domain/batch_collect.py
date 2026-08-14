"""
Parse a downloaded batch results JSONL back into per-day devotional records.

A partially-bad batch is normal (one model response truncates, one comes back
without valid JSON), so nothing here raises on a bad line: every failure is
recorded in an errors list and the remaining days still collect. Losing 364 good
days to one malformed response would be the worse outcome.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from content_batch_graph.domain.batch_io import collection_path

# gen_{lang}_{version}_{year}_{MM-DD} — the date is carried in the id itself, so
# no side index from the input file is needed to recover it.
_CUSTOM_ID_RE = re.compile(
    r"^gen_(?P<lang>[^_]+)_(?P<version>[^_]+)_(?P<year>\d{4})_(?P<md>\d{2}-\d{2})$"
)


def date_from_custom_id(custom_id: str) -> date | None:
    """Recover the date encoded in a custom_id, or None if it doesn't match."""
    m = _CUSTOM_ID_RE.match(custom_id or "")
    if not m:
        return None
    try:
        return date.fromisoformat(f"{m['year']}-{m['md']}")
    except ValueError:
        return None


def extract_content(result_line: dict) -> str | None:
    """
    Pull the raw model content out of one batch result line.

    Handles both shapes seen from OpenAI-compatible batch providers:
      response.body.choices[0].message.content   (wrapped)
      response.choices[0].message.content        (unwrapped)
    """
    resp = result_line.get("response") or {}
    body = resp.get("body") or resp
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def _parse_content(raw: str) -> dict | None:
    """Parse the model's JSON content string, tolerating markdown fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_results(results_path: Path) -> tuple[list[dict], list[dict]]:
    """
    Read a results JSONL into (records, errors).

    records: [{date, reflexion, oracion}, ...] sorted by date.
    errors:  [{custom_id, reason}, ...] — one per line that couldn't be used.
    """
    records: list[dict] = []
    errors: list[dict] = []

    with open(Path(results_path), encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                errors.append(
                    {"custom_id": f"<line {lineno}>", "reason": "invalid JSONL line"}
                )
                continue

            custom_id = result.get("custom_id", "")
            day = date_from_custom_id(custom_id)
            if day is None:
                errors.append(
                    {
                        "custom_id": custom_id or f"<line {lineno}>",
                        "reason": "unparseable custom_id",
                    }
                )
                continue

            content = extract_content(result)
            if not content:
                errors.append(
                    {"custom_id": custom_id, "reason": "missing response content"}
                )
                continue

            parsed = _parse_content(content)
            if parsed is None:
                errors.append(
                    {"custom_id": custom_id, "reason": "content is not a JSON object"}
                )
                continue

            reflexion = parsed.get("reflexion")
            oracion = parsed.get("oracion")
            missing = [
                k
                for k, v in (("reflexion", reflexion), ("oracion", oracion))
                if not isinstance(v, str) or not v.strip()
            ]
            if missing:
                errors.append(
                    {
                        "custom_id": custom_id,
                        "reason": f"missing/empty field(s): {', '.join(missing)}",
                    }
                )
                continue

            records.append(
                {
                    "date": day.isoformat(),
                    "reflexion": reflexion.strip(),
                    "oracion": oracion.strip(),
                }
            )

    records.sort(key=lambda r: r["date"])
    return records, errors


def write_year_collection(
    lang: str,
    version: str,
    year: int,
    records: list[dict],
    errors: list[dict] | None = None,
    out_path: Path | None = None,
) -> Path:
    """
    Write one collected year to data/genomes/Devocional_{year}_{lang}_{version}_gen.json.

    The errors summary travels with the data rather than only to stdout, so a
    partial collection is self-describing when someone opens it later.
    """
    out_path = Path(out_path) if out_path else collection_path(lang, version, year)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "lang": lang,
        "version": version,
        "year": year,
        "count": len(records),
        "errors": errors or [],
        "data": sorted(records, key=lambda r: r["date"]),
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return out_path
