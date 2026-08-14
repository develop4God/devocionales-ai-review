from content_batch_graph.domain.prune import prune_findings
from content_batch_graph.state import VerifiedFinding


def _finding(quoted_text: str, proposed_text: str | None) -> VerifiedFinding:
    return VerifiedFinding(
        quoted_text=quoted_text,
        issue="test issue",
        category="typo",
        proposed_text=proposed_text,
        verified=True,
    )


def test_prune_keeps_finding_with_no_proposed_text():
    # Roles that don't ask for proposed_text (e.g. native_reader) must pass through
    # untouched — this layer only judges findings that opted into proposing a fix.
    findings = [_finding("teh", None)]
    kept, discarded = prune_findings(findings)
    assert kept == findings
    assert discarded == []


def test_prune_discards_noop_proposed_text():
    findings = [_finding("teh", "teh")]
    kept, discarded = prune_findings(findings)
    assert kept == []
    assert len(discarded) == 1
    assert discarded[0]["quoted_text"] == "teh"
    assert "no-op" in discarded[0]["reason"]


def test_prune_discard_is_case_and_whitespace_insensitive():
    findings = [_finding("Teh", " teh ")]
    kept, discarded = prune_findings(findings)
    assert kept == []
    assert len(discarded) == 1


def test_prune_keeps_finding_with_real_proposed_text():
    findings = [_finding("teh", "the")]
    kept, discarded = prune_findings(findings)
    assert kept == findings
    assert discarded == []


def test_prune_findings_empty_list_returns_empty_lists():
    kept, discarded = prune_findings([])
    assert kept == []
    assert discarded == []


def test_prune_findings_mixed_batch_splits_correctly():
    findings = [
        _finding("teh", "the"),  # real fix, keep
        _finding("recieve", "recieve"),  # no-op, discard
        _finding("definately", None),  # no proposal, keep (not this layer's job)
    ]
    kept, discarded = prune_findings(findings)
    assert [f["quoted_text"] for f in kept] == ["teh", "definately"]
    assert [d["quoted_text"] for d in discarded] == ["recieve"]
