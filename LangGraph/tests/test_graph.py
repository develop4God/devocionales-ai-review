import tempfile

from langgraph.types import Command

import content_batch_graph.nodes.critic_pass as critic_pass_module
import content_batch_graph.nodes.drift_check_pass as drift_check_pass_module
import content_batch_graph.nodes.fix_pass as fix_pass_module
import content_batch_graph.nodes.flag_pass as flag_pass_module
from content_batch_graph.graph import compile_graph
from content_batch_graph.state import CriticFinding, Finding


def _fresh_db_path() -> str:
    return tempfile.mktemp(suffix=".sqlite")


def _stub_one_finding(source_text: str, language: str) -> list[Finding]:
    if "teh" not in source_text:
        return []
    return [Finding(quoted_text="teh", issue="Likely typo.", category="typo")]


def _stub_two_findings(source_text: str, language: str) -> list[Finding]:
    return [
        Finding(quoted_text="teh", issue="Likely typo.", category="typo"),
        Finding(
            quoted_text="quite clear",
            issue="Awkward phrasing.",
            category="awkward_phrasing",
        ),
    ]


def _stub_critic_valid(source_text, findings, language):
    return [
        CriticFinding(
            quoted_text=f["quoted_text"],
            issue=f["issue"],
            category=f["category"],
            verified=True,
            is_valid=True,
            replacement_text="the" if f["quoted_text"] == "teh" else "very clear",
            critic_reasoning="Confirmed.",
        )
        for f in findings
    ]


def _stub_fix(file_text, findings):
    applicable = [f for f in findings if f["is_valid"] and f["replacement_text"]]
    if not applicable:
        return file_text, "No approved findings to fix."
    fixed = file_text
    for f in applicable:
        fixed = fixed.replace(f["quoted_text"], f["replacement_text"])
    return fixed, "Fixed 1 typo: 'teh' -> 'the'."


def _stub_fix_selected(file_text, findings):
    applicable = [f for f in findings if f["is_valid"] and f["replacement_text"]]
    if not applicable:
        return file_text, "No approved findings to fix."
    fixed = file_text
    for f in applicable:
        fixed = fixed.replace(f["quoted_text"], f["replacement_text"])
    return fixed, f"Fixed {len(applicable)} finding(s)."


def _stub_fix_produces_invalid_json(file_text, findings):
    return "{not valid json", "Broke it on purpose."


def _stub_no_drift(fixed_text, applied_findings, language):
    return False, "No drift found."


def test_graph_pauses_at_human_confirm_with_verified_findings(monkeypatch):
    monkeypatch.setattr(flag_pass_module, "run_flag_pass", _stub_one_finding)
    monkeypatch.setattr(critic_pass_module, "run_critic_pass_batch", _stub_critic_valid)
    graph, _ = compile_graph(_fresh_db_path())
    config = {"configurable": {"thread_id": "test-pause"}}

    result = graph.invoke(
        {
            "file_path": "fake.txt",
            "file_text": "This has teh typo in it.",
            "language": "English",
        },
        config=config,
    )

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["question"].startswith("Which critic-reviewed findings")
    assert len(payload["verified_findings"]) == 1
    assert payload["verified_findings"][0]["quoted_text"] == "teh"
    assert payload["rejected_findings"] == []
    assert len(payload["critic_findings"]) == 1
    assert payload["critic_findings"][0]["is_valid"] is True


def test_graph_resumes_and_records_human_decision(monkeypatch):
    monkeypatch.setattr(flag_pass_module, "run_flag_pass", _stub_one_finding)
    monkeypatch.setattr(critic_pass_module, "run_critic_pass_batch", _stub_critic_valid)
    monkeypatch.setattr(fix_pass_module, "run_fix_pass", _stub_fix)
    monkeypatch.setattr(drift_check_pass_module, "run_drift_check", _stub_no_drift)
    graph, _ = compile_graph(_fresh_db_path())
    config = {"configurable": {"thread_id": "test-resume"}}

    graph.invoke(
        {"file_path": "fake.txt", "file_text": "teh typo here", "language": "English"},
        config=config,
    )
    final = graph.invoke(Command(resume={"apply": [0]}), config=config)

    assert final["human_decision"] == {"apply": [0]}
    assert len(final["verified_findings"]) == 1


def test_graph_checkpoint_survives_separate_graph_instance(monkeypatch):
    """The checkpointer, not in-process memory, is what makes resume work — verified
    by compiling a SECOND graph instance against the same db path to resume, rather
    than reusing the original in-process graph object."""
    monkeypatch.setattr(flag_pass_module, "run_flag_pass", _stub_one_finding)
    monkeypatch.setattr(critic_pass_module, "run_critic_pass_batch", _stub_critic_valid)
    monkeypatch.setattr(fix_pass_module, "run_fix_pass", _stub_fix)
    monkeypatch.setattr(drift_check_pass_module, "run_drift_check", _stub_no_drift)
    db_path = _fresh_db_path()
    config = {"configurable": {"thread_id": "test-cross-instance"}}

    graph_a, _ = compile_graph(db_path)
    result = graph_a.invoke(
        {"file_path": "fake.txt", "file_text": "teh error", "language": "English"},
        config=config,
    )
    assert "__interrupt__" in result

    graph_b, _ = compile_graph(db_path)
    final = graph_b.invoke(Command(resume={"apply": [0]}), config=config)

    assert final["human_decision"] == {"apply": [0]}
    assert final["verified_findings"][0]["quoted_text"] == "teh"


def test_graph_clean_text_produces_no_findings_but_still_pauses(monkeypatch):
    monkeypatch.setattr(flag_pass_module, "run_flag_pass", _stub_one_finding)
    monkeypatch.setattr(critic_pass_module, "run_critic_pass_batch", _stub_critic_valid)
    graph, _ = compile_graph(_fresh_db_path())
    config = {"configurable": {"thread_id": "test-clean"}}

    result = graph.invoke(
        {
            "file_path": "fake.txt",
            "file_text": "This text is entirely clean.",
            "language": "English",
        },
        config=config,
    )

    payload = result["__interrupt__"][0].value
    assert payload["verified_findings"] == []
    assert payload["rejected_findings"] == []


def test_graph_approved_decision_runs_fix_and_validate(monkeypatch):
    monkeypatch.setattr(flag_pass_module, "run_flag_pass", _stub_one_finding)
    monkeypatch.setattr(critic_pass_module, "run_critic_pass_batch", _stub_critic_valid)
    monkeypatch.setattr(fix_pass_module, "run_fix_pass", _stub_fix)
    monkeypatch.setattr(drift_check_pass_module, "run_drift_check", _stub_no_drift)
    graph, _ = compile_graph(_fresh_db_path())
    config = {"configurable": {"thread_id": "test-approved-fix"}}

    graph.invoke(
        {
            "file_path": "fake.txt",
            "file_text": '{"text": "teh typo here"}',
            "language": "English",
        },
        config=config,
    )
    final = graph.invoke(Command(resume={"apply": [0]}), config=config)

    assert final["fixed_text"] == '{"text": "the typo here"}'
    assert "Fixed 1 typo" in final["fix_summary"]
    assert final["validation_passed"] is True
    assert final["fix_attempts"] == 1
    assert final["drift_detected"] is False


def test_graph_rejected_decision_skips_fix_and_validate(monkeypatch):
    monkeypatch.setattr(flag_pass_module, "run_flag_pass", _stub_one_finding)
    monkeypatch.setattr(critic_pass_module, "run_critic_pass_batch", _stub_critic_valid)
    monkeypatch.setattr(fix_pass_module, "run_fix_pass", _stub_fix)
    monkeypatch.setattr(drift_check_pass_module, "run_drift_check", _stub_no_drift)
    graph, _ = compile_graph(_fresh_db_path())
    config = {"configurable": {"thread_id": "test-rejected-skips-fix"}}

    graph.invoke(
        {"file_path": "fake.txt", "file_text": "teh typo here", "language": "English"},
        config=config,
    )
    final = graph.invoke(Command(resume={"apply": []}), config=config)

    assert final["human_decision"] == {"apply": []}
    assert final.get("fixed_text") is None
    assert final.get("validation_passed") is None


def test_graph_validate_failure_loops_back_to_fix_then_stops_at_cap(monkeypatch):
    # A fix_pass that always produces invalid JSON should loop back through
    # fix_pass/validate_pass up to MAX_FIX_ATTEMPTS, then stop rather than loop
    # forever — proving the loop-cap guard actually terminates the graph.
    monkeypatch.setattr(flag_pass_module, "run_flag_pass", _stub_one_finding)
    monkeypatch.setattr(critic_pass_module, "run_critic_pass_batch", _stub_critic_valid)
    monkeypatch.setattr(
        fix_pass_module, "run_fix_pass", _stub_fix_produces_invalid_json
    )
    monkeypatch.setattr(drift_check_pass_module, "run_drift_check", _stub_no_drift)
    graph, _ = compile_graph(_fresh_db_path())
    config = {"configurable": {"thread_id": "test-validate-loop-cap"}}

    graph.invoke(
        {"file_path": "fake.txt", "file_text": "teh typo here", "language": "English"},
        config=config,
    )
    final = graph.invoke(Command(resume={"apply": [0]}), config=config)

    assert final["validation_passed"] is False
    assert final["fix_attempts"] == 3


def test_graph_drift_detected_pauses_at_human_confirm_again(monkeypatch):
    def _stub_drift(fixed_text, applied_findings, language):
        return True, "Replacement reads awkwardly in context."

    monkeypatch.setattr(flag_pass_module, "run_flag_pass", _stub_one_finding)
    monkeypatch.setattr(critic_pass_module, "run_critic_pass_batch", _stub_critic_valid)
    monkeypatch.setattr(fix_pass_module, "run_fix_pass", _stub_fix)
    monkeypatch.setattr(drift_check_pass_module, "run_drift_check", _stub_drift)
    graph, _ = compile_graph(_fresh_db_path())
    config = {"configurable": {"thread_id": "test-drift-pause"}}

    graph.invoke(
        {
            "file_path": "fake.txt",
            "file_text": '{"text": "teh typo here"}',
            "language": "English",
        },
        config=config,
    )
    result = graph.invoke(Command(resume={"apply": [0]}), config=config)

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["question"].startswith("Drift detected")
    assert payload["drift_detected"] is True
    assert result.get("validation_passed") is None

    final = graph.invoke(Command(resume={"apply": []}), config=config)
    assert final["human_decision"] == {"apply": []}


def test_graph_applies_only_selected_finding_indices(monkeypatch):
    # Two findings surface; the human approves only index 0 ("teh" -> "the"),
    # leaving index 1 ("quite clear" -> "very clear") untouched — proving fix_pass
    # applies exactly the selected findings, not all-or-nothing.
    monkeypatch.setattr(flag_pass_module, "run_flag_pass", _stub_two_findings)
    monkeypatch.setattr(critic_pass_module, "run_critic_pass_batch", _stub_critic_valid)
    monkeypatch.setattr(fix_pass_module, "run_fix_pass", _stub_fix_selected)
    monkeypatch.setattr(drift_check_pass_module, "run_drift_check", _stub_no_drift)
    graph, _ = compile_graph(_fresh_db_path())
    config = {"configurable": {"thread_id": "test-partial-apply"}}

    graph.invoke(
        {
            "file_path": "fake.txt",
            "file_text": "This has teh typo, and it is quite clear too.",
            "language": "English",
        },
        config=config,
    )
    final = graph.invoke(Command(resume={"apply": [0]}), config=config)

    assert final["human_decision"] == {"apply": [0]}
    assert "the typo" in final["fixed_text"]
    assert "quite clear" in final["fixed_text"]  # index 1 was NOT applied
    assert "very clear" not in final["fixed_text"]


def _stub_findings_one_noop_one_real(source_text: str, language: str) -> list[Finding]:
    return [
        Finding(
            quoted_text="teh",
            issue="Likely typo.",
            category="typo",
            proposed_text="the",
        ),
        Finding(
            quoted_text="recieve",
            issue="Model flagged this but proposed no change.",
            category="typo",
            proposed_text="recieve",  # no-op — must be pruned before critic_pass
        ),
    ]


def test_graph_prune_pass_cuts_noop_finding_before_critic(monkeypatch):
    # native_reader_batch-style findings carry proposed_text. One is a real fix
    # ("teh" -> "the"), the other is a no-op the flag pass shouldn't have raised.
    # prune_pass must cut the no-op for free, before critic_pass ever sees it, and
    # surface it in discarded_findings for calibration — not silently vanish it.
    critic_calls = []

    def _stub_critic_records_calls(source_text, findings, language):
        critic_calls.extend(findings)
        return [
            CriticFinding(
                quoted_text=f["quoted_text"],
                issue=f["issue"],
                category=f["category"],
                proposed_text=f.get("proposed_text"),
                verified=True,
                is_valid=True,
                replacement_text="the",
                critic_reasoning="Confirmed.",
            )
            for f in findings
        ]

    monkeypatch.setattr(
        flag_pass_module, "run_flag_pass", _stub_findings_one_noop_one_real
    )
    monkeypatch.setattr(
        critic_pass_module, "run_critic_pass_batch", _stub_critic_records_calls
    )
    graph, _ = compile_graph(_fresh_db_path())
    config = {"configurable": {"thread_id": "test-prune"}}

    result = graph.invoke(
        {
            "file_path": "fake.txt",
            "file_text": "This has teh typo, and did not recieve the memo.",
            "language": "English",
        },
        config=config,
    )

    # Only the real fix reached critic_pass — the no-op was cut by prune_pass.
    assert [f["quoted_text"] for f in critic_calls] == ["teh"]

    payload = result["__interrupt__"][0].value
    assert len(payload["discarded_findings"]) == 1
    assert payload["discarded_findings"][0]["quoted_text"] == "recieve"
    assert "no-op" in payload["discarded_findings"][0]["reason"]
