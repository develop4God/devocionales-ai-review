"""
Shared test isolation: no test in this suite should write real files into
data/batch_input or data/batch_output.

content_batch_graph.domain.batch_io.PATHS is a module-level BatchPaths built
once at import time from LANGGRAPH_DATA_DIR (see batch_io.py) -- setting that
env var from inside a test is too late, the singleton is already frozen. So
this replaces the singleton itself, per test, with one pointed at a pytest
tmp_path instead.
"""

from __future__ import annotations

import pytest
from batch_common import BatchPaths

from content_batch_graph.domain import batch_io


@pytest.fixture(autouse=True)
def _isolated_batch_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_io, "PATHS", BatchPaths(tmp_path, env_prefix="LANGGRAPH"))
    monkeypatch.setattr(batch_io, "GENOMES_DIR", tmp_path / "data" / "genomes")
