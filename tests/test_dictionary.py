import httpx
import pytest

from content_batch_graph.domain.dictionary import lookup_word


def test_lookup_word_real_call_finds_a_known_word():
    # Real call against kwfdiksiyonaryo.ph — the specific case this module exists
    # for. "espiritwal" is a confirmed real KWF headword.
    result = lookup_word("espiritwal")
    assert result["found"] is True
    assert result["word"] == "espiritwal"
    assert result["part_of_speech"] is not None
    assert result["definition"] is not None


def test_lookup_word_real_call_reports_not_found_for_a_nonstandard_spelling():
    # "espirituwal" has no KWF entry — this is the exact bug this module fixes:
    # the critic previously trusted its own memory instead of checking this.
    result = lookup_word("espirituwal")
    assert result["found"] is False
    assert result["part_of_speech"] is None
    assert result["definition"] is None


def test_lookup_word_raises_on_http_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise httpx.ConnectError("simulated network failure")

    monkeypatch.setattr(httpx, "get", _raise)
    with pytest.raises(httpx.ConnectError):
        lookup_word("espiritwal")
