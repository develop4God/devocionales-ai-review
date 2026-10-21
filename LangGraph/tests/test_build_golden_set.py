import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "build_golden_set", Path(__file__).resolve().parent.parent / "scripts" / "build_golden_set.py")
bgs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bgs)


def test_label_mapping():
    assert bgs.label_for("applied") == "positive"
    assert bgs.label_for("applied_corrected") == "positive"
    assert bgs.label_for("dismissed_bad_critic_fix") == "negative"
    assert bgs.label_for("dismissed") == "negative"
    for d in ("dismissed_style_preference", "dismissed_ambiguous", "already_resolved",
              "recommend_apply", "left_awkward_not_applied"):
        assert bgs.label_for(d) == "excluded"


def test_build_rows_joins_providers(tmp_path):
    triage = {"entry_id": "E1", "field": "oracion", "quoted_text": "foo",
              "category": "typo", "disposition": "applied", "reason": "r"}
    (tmp_path / "2027_es_findings_review.jsonl").write_text(json.dumps(triage) + "\n")
    ledger = {"entry_id": "E1", "field": "oracion", "provider_id": "p1",
              "critic_findings": [{"quoted_text": "foo"}]}
    (tmp_path / "2027_es_x_ledger.jsonl").write_text(json.dumps(ledger) + "\n")
    [row] = bgs.build_rows("es", tmp_path)
    assert row["label"] == "positive" and row["providers"] == ["p1"]
