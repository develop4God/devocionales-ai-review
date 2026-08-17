from content_batch_graph.domain.review_prefilter import classify_item
from content_batch_graph.state import Finding


def _finding(quoted_text: str, proposed_text: str | None = "the") -> Finding:
    return Finding(
        quoted_text=quoted_text,
        issue="test issue",
        category="typo",
        proposed_text=proposed_text,
    )


def test_classify_item_kept_when_finding_survives_verify_and_prune():
    result = classify_item("entry1", "body", [_finding("teh")], "this is teh text")
    assert result.kept is True
    assert [f["quoted_text"] for f in result.kept_findings] == ["teh"]
    assert result.pre_pruned_row is None


def test_classify_item_rejected_by_verify_when_quote_not_in_text():
    result = classify_item("entry1", "body", [_finding("xyzzy")], "this is teh text")
    assert result.kept is False
    assert result.verified_count == 0
    assert result.rejected_count == 1
    assert result.pre_pruned_row["status"] == "rejected_by_verify"
    assert result.pre_pruned_row["entry_id"] == "entry1"
    assert result.pre_pruned_row["field"] == "body"
    assert result.pre_pruned_row["thread_id"] == "entry1:body"


def test_classify_item_pruned_after_verify_when_proposed_text_is_noop():
    result = classify_item(
        "entry1", "body", [_finding("teh", "teh")], "this is teh text"
    )
    assert result.kept is False
    assert result.verified_count == 1
    assert result.discarded_count == 1
    assert result.pre_pruned_row["status"] == "pruned_after_verify"


def test_classify_item_pre_pruned_row_is_ready_to_write_ledger_shape():
    result = classify_item("entry1", "body", [_finding("xyzzy")], "this is teh text")
    row = result.pre_pruned_row
    assert row["critic_findings"] == []
    assert row["applied_indices"] == []
    assert row["fix_summary"] is None
    assert row["validation_passed"] is None
