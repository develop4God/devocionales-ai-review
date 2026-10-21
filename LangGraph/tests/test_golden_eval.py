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
