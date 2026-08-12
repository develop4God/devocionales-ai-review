from content_batch_graph.domain.review_report import (
    build_pending_review,
    build_review_report,
    format_pending_review_text,
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


def test_build_pending_review_splits_typo_from_stylistic():
    findings = [
        _finding("teh", "the", category="typo"),
        _finding("quite clear", "very clear", category="awkward_phrasing"),
    ]

    review = build_pending_review(findings, "entry1", "English")

    assert len(review["mechanical"]) == 1
    assert review["mechanical"][0]["index"] == 0
    assert review["mechanical"][0]["quoted_text"] == "teh"

    assert len(review["stylistic"]) == 1
    assert review["stylistic"][0]["index"] == 1
    assert review["stylistic"][0]["category"] == "awkward_phrasing"


def test_build_pending_review_skips_rejected_findings():
    findings = [
        _finding("teh", "the", category="typo"),
        _finding("fine text", None, is_valid=False, category="typo"),
    ]

    review = build_pending_review(findings, "entry1", "English")

    assert len(review["mechanical"]) == 1
    assert review["mechanical"][0]["quoted_text"] == "teh"


def test_build_pending_review_preserves_original_indices_across_tiers():
    # A rejected finding sits between two applicable ones — indices in the report
    # must still match their real position in critic_findings, not a re-numbered
    # position within mechanical/stylistic.
    findings = [
        _finding("teh", "the", category="typo"),
        _finding("fine text", None, is_valid=False, category="typo"),
        _finding("quite clear", "very clear", category="awkward_phrasing"),
    ]

    review = build_pending_review(findings, "entry1", "English")

    assert review["mechanical"][0]["index"] == 0
    assert review["stylistic"][0]["index"] == 2


def test_build_pending_review_includes_field_path():
    findings = [_finding("teh", "the", category="typo")]

    review = build_pending_review(
        findings, "entry1", "English", field_path="data.en.2025-08-01.0.reflexion"
    )

    assert review["field_path"] == "data.en.2025-08-01.0.reflexion"
    assert review["mechanical"][0]["field_path"] == "data.en.2025-08-01.0.reflexion"


def test_build_pending_review_marks_kwf_grounded():
    findings = [
        _finding(
            "espirituwal",
            "espiritwal",
            category="typo",
            reasoning="real word in the KWF dictionary — dismissed.",
        )
    ]

    review = build_pending_review(findings, "entry1", "Filipino")

    assert review["mechanical"][0]["grounded"] is True


def test_format_pending_review_text_produces_readable_output():
    findings = [
        _finding("teh", "the", category="typo", reasoning="Common typo."),
        _finding(
            "quite clear",
            "very clear",
            category="awkward_phrasing",
            reasoning="Reads a bit stiff.",
        ),
    ]

    review = build_pending_review(
        findings, "entry1", "English", field_path="data.en.2025-08-01.0.reflexion"
    )
    text = format_pending_review_text(review)

    assert "entry1" in text
    assert "data.en.2025-08-01.0.reflexion" in text
    assert "[0] 'teh' -> 'the'" in text
    assert "[1] [awkward_phrasing] 'quite clear' -> 'very clear'" in text
    assert "Reads a bit stiff." in text
