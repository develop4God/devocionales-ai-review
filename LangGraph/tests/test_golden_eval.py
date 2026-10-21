import importlib.util
import sys
from pathlib import Path

_scripts = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_scripts))
_spec = importlib.util.spec_from_file_location("golden_eval", _scripts / "golden_eval.py")
ge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ge)


def test_index_entries_keys_by_trailing_date_so_drifted_ids_match():
    doc = {"data": {"es": {"2027-08-03": [{"id": "1Pedro167RVR196020270803", "oracion": "x"}]}}}
    assert "20270803" in ge.index_entries(doc)
    assert ge.index_entries(doc)["20270803"]["oracion"] == "x"


def test_check_providers_flags_unknown_id():
    [(pid, status)] = ge.check_providers(["no_such_provider"])
    assert status == "NOT IN providers.yml"


def test_quotes_match_is_containment_and_whitespace_insensitive():
    assert ge.quotes_match("je marche  jamais", "je marche jamais seul")
    assert not ge.quotes_match("foo", "bar")
    assert not ge.quotes_match("", "bar")


def test_score_field_counts_tp_fp_and_unlabeled():
    rows = [{"label": "positive", "quoted_text": "aaa"}, {"label": "negative", "quoted_text": "bbb"}]
    out = ge.score_field(rows, ["aaa", "bbb", "zzz"])
    assert (out["tp"], out["fp"], out["unlabeled"]) == (1, 1, ["zzz"])


def test_recover_fields_skips_non_reviewed_fields(monkeypatch, tmp_path):
    rows = [
        {"lang": "fr", "entry_id": "X20270101", "field": "tags", "quoted_text": "a", "label": "positive", "applied_commit": None},
        {"lang": "fr", "entry_id": "X20270101", "field": "oracion", "quoted_text": "bon", "label": "negative", "applied_commit": None},
    ]
    monkeypatch.setattr(ge, "build_rows", lambda lang: rows)
    corpus = tmp_path / ge.YEARLY / ge.CORPUS_FILES["fr"]
    corpus.parent.mkdir(parents=True)
    corpus.write_text('{"data": {"fr": {"d": [{"id": "X20270101", "oracion": "un bon texte", "tags": ["a"]}]}}}')
    fields, _ = ge.recover_fields("fr", tmp_path)
    assert list(fields) == [("20270101", "oracion")]
