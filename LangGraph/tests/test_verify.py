from content_batch_graph.domain.verify import verify_finding, verify_findings
from content_batch_graph.state import Finding


def _finding(quoted_text: str) -> Finding:
    return Finding(quoted_text=quoted_text, issue="test issue", category="typo")


def test_verify_finding_present_in_source_returns_verified():
    result = verify_finding(_finding("teh"), "This has teh typo.")
    assert result is not None
    assert result["verified"] is True
    assert result["quoted_text"] == "teh"


def test_verify_finding_absent_from_source_returns_none():
    result = verify_finding(_finding("nonexistent"), "This text has no such phrase.")
    assert result is None


def test_verify_finding_empty_quote_is_rejected():
    result = verify_finding(_finding(""), "any text")
    assert result is None


def test_verify_findings_splits_verified_and_rejected():
    findings = [_finding("teh"), _finding("hallucinated phrase")]
    verified, rejected = verify_findings(findings, "This has teh typo, nothing else.")
    assert len(verified) == 1
    assert verified[0]["quoted_text"] == "teh"
    assert len(rejected) == 1
    assert rejected[0]["quoted_text"] == "hallucinated phrase"


def test_verify_findings_empty_list_returns_empty_lists():
    verified, rejected = verify_findings([], "any text")
    assert verified == []
    assert rejected == []


def test_verify_finding_rejects_substring_inside_longer_word():
    # Real bug: quoted_text "refleje" is a plain substring of the already-correct
    # word "reflejen" in the source. A naive `in` check passes this, and a
    # downstream str.replace("refleje", "reflejen") then corrupts the correct
    # word into "reflejenn". This must be rejected, not verified.
    result = verify_finding(
        _finding("refleje"), "donde mis acciones reflejen mi confianza en ti."
    )
    assert result is None


def test_verify_finding_accepts_whole_word_even_when_also_a_substring_elsewhere():
    # "refleje" is a real, correctly-bounded word here (followed by a space),
    # even though the same source text also contains "reflejen" elsewhere.
    result = verify_finding(
        _finding("refleje"),
        "que mis acciones refleje mi fe, y que reflejen tu gloria.",
    )
    assert result is not None
    assert result["quoted_text"] == "refleje"


def test_verify_finding_accepts_multi_word_phrase_with_internal_spaces():
    # Word-boundary checking only applies at the two edges of quoted_text; a
    # multi-word phrase's internal spaces must never be treated as required
    # boundaries in a way that breaks matching.
    result = verify_finding(
        _finding("el camino, y la verdad"),
        'Jesús dijo: "Yo soy el camino, y la verdad, y la vida".',
    )
    assert result is not None


def test_verify_finding_rejects_when_preceded_by_word_character():
    # Boundary check applies to the start of quoted_text too, not just the end.
    result = verify_finding(_finding("ejen"), "mis acciones reflejen mi fe.")
    assert result is None
