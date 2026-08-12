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
