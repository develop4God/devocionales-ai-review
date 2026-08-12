from content_batch_graph.domain.fix import run_fix_pass
from content_batch_graph.domain.validate import validate_json
from content_batch_graph.state import CriticFinding


def _critic_finding(
    quoted_text: str,
    issue: str,
    replacement_text: str | None,
    is_valid: bool = True,
    category: str = "typo",
) -> CriticFinding:
    return CriticFinding(
        quoted_text=quoted_text,
        issue=issue,
        category=category,
        verified=True,
        is_valid=is_valid,
        replacement_text=replacement_text,
        critic_reasoning="test reasoning",
    )


def test_fix_pass_returns_unchanged_text_and_note_when_no_findings():
    fixed_text, summary = run_fix_pass("Some text.", [])
    assert fixed_text == "Some text."
    assert "No approved findings" in summary


def test_fix_pass_applies_surgical_replacement_for_valid_finding():
    source = "This is teh correct answer, and it is quite clear."
    findings = [_critic_finding("teh", "Likely typo.", "the")]
    fixed_text, summary = run_fix_pass(source, findings)

    assert fixed_text == "This is the correct answer, and it is quite clear."
    assert "teh" in summary and "the" in summary


def test_fix_pass_skips_findings_the_critic_marked_invalid():
    source = "This is teh correct answer."
    findings = [_critic_finding("teh", "Claimed typo.", None, is_valid=False)]
    fixed_text, summary = run_fix_pass(source, findings)

    assert fixed_text == source
    assert "No approved findings" in summary


def test_fix_pass_only_changes_the_flagged_span():
    source = '{"reflexion": "This is teh correct answer."}'
    findings = [_critic_finding("teh", "Likely typo.", "the")]
    fixed_text, _summary = run_fix_pass(source, findings)

    assert fixed_text == '{"reflexion": "This is the correct answer."}'
    passed, error = validate_json(fixed_text)
    assert passed, f"surgical fix broke JSON structure: {error}"
