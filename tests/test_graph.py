import tempfile

from langgraph.types import Command

from content_batch_graph.graph import compile_graph


def _fresh_db_path() -> str:
    return tempfile.mktemp(suffix=".sqlite")


def test_graph_pauses_at_human_confirm_with_verified_findings():
    graph, _ = compile_graph(_fresh_db_path())
    config = {"configurable": {"thread_id": "test-pause"}}

    result = graph.invoke(
        {"file_path": "fake.txt", "file_text": "This has teh typo in it."},
        config=config,
    )

    assert "__interrupt__" in result
    payload = result["__interrupt__"][0].value
    assert payload["question"] == "Approve these verified findings?"
    assert len(payload["verified_findings"]) == 1
    assert payload["verified_findings"][0]["quoted_text"] == "teh"
    assert payload["rejected_findings"] == []


def test_graph_resumes_and_records_human_decision():
    graph, _ = compile_graph(_fresh_db_path())
    config = {"configurable": {"thread_id": "test-resume"}}

    graph.invoke({"file_path": "fake.txt", "file_text": "teh typo here"}, config=config)
    final = graph.invoke(Command(resume="approved"), config=config)

    assert final["human_decision"] == "approved"
    assert len(final["verified_findings"]) == 1


def test_graph_checkpoint_survives_separate_graph_instance():
    """The checkpointer, not in-process memory, is what makes resume work — verified
    by compiling a SECOND graph instance against the same db path to resume, rather
    than reusing the original in-process graph object."""
    db_path = _fresh_db_path()
    config = {"configurable": {"thread_id": "test-cross-instance"}}

    graph_a, _ = compile_graph(db_path)
    result = graph_a.invoke(
        {"file_path": "fake.txt", "file_text": "teh error"}, config=config
    )
    assert "__interrupt__" in result

    graph_b, _ = compile_graph(db_path)
    final = graph_b.invoke(Command(resume="approved"), config=config)

    assert final["human_decision"] == "approved"
    assert final["verified_findings"][0]["quoted_text"] == "teh"


def test_graph_clean_text_produces_no_findings_but_still_pauses():
    graph, _ = compile_graph(_fresh_db_path())
    config = {"configurable": {"thread_id": "test-clean"}}

    result = graph.invoke(
        {"file_path": "fake.txt", "file_text": "This text is entirely clean."},
        config=config,
    )

    payload = result["__interrupt__"][0].value
    assert payload["verified_findings"] == []
    assert payload["rejected_findings"] == []
