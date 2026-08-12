import tempfile

from langgraph.types import Command

import content_batch_graph.nodes.flag_pass as flag_pass_module
from content_batch_graph.graph import compile_graph
from content_batch_graph.state import Finding


def _fresh_db_path() -> str:
    return tempfile.mktemp(suffix=".sqlite")


def _stub_one_finding(source_text: str, language: str) -> list[Finding]:
    if "teh" not in source_text:
        return []
    return [Finding(quoted_text="teh", issue="Likely typo.", category="typo")]


def test_graph_pauses_at_human_confirm_with_verified_findings(monkeypatch):
    monkeypatch.setattr(flag_pass_module, "run_flag_pass", _stub_one_finding)
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
    assert payload["question"] == "Approve these verified findings?"
    assert len(payload["verified_findings"]) == 1
    assert payload["verified_findings"][0]["quoted_text"] == "teh"
    assert payload["rejected_findings"] == []


def test_graph_resumes_and_records_human_decision(monkeypatch):
    monkeypatch.setattr(flag_pass_module, "run_flag_pass", _stub_one_finding)
    graph, _ = compile_graph(_fresh_db_path())
    config = {"configurable": {"thread_id": "test-resume"}}

    graph.invoke(
        {"file_path": "fake.txt", "file_text": "teh typo here", "language": "English"},
        config=config,
    )
    final = graph.invoke(Command(resume="approved"), config=config)

    assert final["human_decision"] == "approved"
    assert len(final["verified_findings"]) == 1


def test_graph_checkpoint_survives_separate_graph_instance(monkeypatch):
    """The checkpointer, not in-process memory, is what makes resume work — verified
    by compiling a SECOND graph instance against the same db path to resume, rather
    than reusing the original in-process graph object."""
    monkeypatch.setattr(flag_pass_module, "run_flag_pass", _stub_one_finding)
    db_path = _fresh_db_path()
    config = {"configurable": {"thread_id": "test-cross-instance"}}

    graph_a, _ = compile_graph(db_path)
    result = graph_a.invoke(
        {"file_path": "fake.txt", "file_text": "teh error", "language": "English"},
        config=config,
    )
    assert "__interrupt__" in result

    graph_b, _ = compile_graph(db_path)
    final = graph_b.invoke(Command(resume="approved"), config=config)

    assert final["human_decision"] == "approved"
    assert final["verified_findings"][0]["quoted_text"] == "teh"


def test_graph_clean_text_produces_no_findings_but_still_pauses(monkeypatch):
    monkeypatch.setattr(flag_pass_module, "run_flag_pass", _stub_one_finding)
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
