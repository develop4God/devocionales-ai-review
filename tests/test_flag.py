from content_batch_graph.domain.flag import run_flag_pass


def test_flag_pass_finds_each_occurrence_of_known_typo():
    findings = run_flag_pass("teh first and teh second occurrence")
    assert len(findings) == 2
    assert all(f["quoted_text"] == "teh" for f in findings)
    assert all(f["category"] == "typo" for f in findings)


def test_flag_pass_returns_empty_for_clean_text():
    findings = run_flag_pass("This text has no known issues in it.")
    assert findings == []


def test_flag_pass_empty_input_returns_empty():
    assert run_flag_pass("") == []
