from content_batch_graph.domain.review_report import (
    build_review_report,
    format_report_text,
)


def _finding(
    quoted_text,
    replacement_text,
    is_valid=True,
    category="typo",
    reasoning="stub reasoning",
):
    return {
        "category": category,
        "quoted_text": quoted_text,
        "replacement_text": replacement_text,
        "is_valid": is_valid,
        "critic_reasoning": reasoning,
    }


def test_build_review_report_identifies_exact_word_changes():
    before = "This is teh correct answer, and it is quite clear."
    after = "This is the correct answer, and it is quite clear."
    findings = [_finding("teh", "the")]

    report = build_review_report(before, after, findings, "entry1", "English")

    assert len(report["word_changes"]) == 1
    assert report["word_changes"][0]["before"] == "teh"
    assert report["word_changes"][0]["after"] == "the"
    assert report["changed_word_count"] == 1
    assert report["unchanged_word_count"] == 9


def test_build_review_report_no_changes_when_text_unchanged():
    text = "This text has no issues."
    report = build_review_report(text, text, [], "entry1", "English")

    assert report["word_changes"] == []
    assert report["changed_word_count"] == 0
    assert report["unchanged_word_count"] == len(text.split())


def test_build_review_report_marks_kwf_grounded_findings():
    before = "May espirituwal na kahulugan ito."
    after = "May espiritwal na kahulugan ito."
    findings = [
        _finding(
            "espirituwal",
            "espiritwal",
            reasoning="'espirituwal' is a real word in the KWF dictionary — dismissed.",
        )
    ]

    report = build_review_report(before, after, findings, "entry1", "Filipino")

    assert report["findings"][0]["grounded"] is True


def test_build_review_report_marks_ungrounded_findings():
    before = "This is teh answer."
    after = "This is the answer."
    findings = [_finding("teh", "the", reasoning="Common English typo.")]

    report = build_review_report(before, after, findings, "entry1", "English")

    assert report["findings"][0]["grounded"] is False


def test_build_review_report_includes_context_windows():
    before = "A" * 100 + "teh" + "B" * 100
    after = "A" * 100 + "the" + "B" * 100
    findings = [_finding("teh", "the")]

    report = build_review_report(before, after, findings, "entry1", "English")

    finding = report["findings"][0]
    assert "teh" in finding["context_before"] or finding["context_before"] == ""
    assert finding["context_before"].startswith("...")
    assert finding["context_after"].startswith("...")


def test_build_review_report_rejected_finding_has_no_replacement():
    before = "This phrase is fine."
    findings = [
        _finding(
            "is fine",
            None,
            is_valid=False,
            reasoning="Not actually an error.",
        )
    ]

    report = build_review_report(before, before, findings, "entry1", "English")

    assert report["findings"][0]["is_valid"] is False
    assert report["findings"][0]["replacement_text"] is None


def test_format_report_text_produces_readable_output():
    before = "This is teh answer."
    after = "This is the answer."
    findings = [_finding("teh", "the", reasoning="Common typo.")]

    report = build_review_report(
        before,
        after,
        findings,
        "entry1",
        "English",
        drift_detected=False,
        drift_notes="Clean.",
        validation_passed=True,
    )
    text = format_report_text(report)

    assert "entry1" in text
    assert "'teh' -> 'the'" in text
    assert "APPLIED" in text
    assert "model judgment only" in text
    assert "drift_detected: False" in text
    assert "Validation: True" in text
