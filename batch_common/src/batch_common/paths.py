"""
Generic batch data-directory helper.

Generalizes GEP/paths.py's fixed module-level constants + resolve_batch_input /
resolve_batch_output into an instantiable object, so each consuming project
points it at its own data root and its own env-var prefix rather than importing
another project's hardcoded paths.
"""

from __future__ import annotations

import os
from pathlib import Path


class BatchPaths:
    """
    Batch input/output directories under one project's data root.

    root:       the project directory that owns `data/` (e.g. LangGraph/).
    env_prefix: prefix for the optional override env var, e.g. "LANGGRAPH"
                makes LANGGRAPH_DATA_DIR override the data subdirectory name.
    """

    def __init__(
        self, root: Path, env_prefix: str, data_dir_name: str = "data"
    ) -> None:
        self.root = Path(root)
        self.env_prefix = env_prefix
        self.data_dir = self.root / os.environ.get(
            f"{env_prefix}_DATA_DIR", data_dir_name
        )

    @property
    def batch_input_dir(self) -> Path:
        return self.data_dir / "batch_input"

    @property
    def batch_output_dir(self) -> Path:
        return self.data_dir / "batch_output"

    def ensure_dirs(self) -> None:
        """Create the batch data directories if they don't exist (idempotent)."""
        self.batch_input_dir.mkdir(parents=True, exist_ok=True)
        self.batch_output_dir.mkdir(parents=True, exist_ok=True)

    def resolve_batch_input(self, file_arg: str | Path) -> Path:
        """Resolve an --input argument, falling back to batch_input_dir/<name>."""
        return self._resolve(file_arg, self.batch_input_dir)

    def resolve_batch_output(self, file_arg: str | Path) -> Path:
        """Resolve a --results argument, falling back to batch_output_dir/<name>."""
        return self._resolve(file_arg, self.batch_output_dir)

    @staticmethod
    def _resolve(file_arg: str | Path, fallback_dir: Path) -> Path:
        p = Path(file_arg)
        if p.exists():
            return p
        candidate = fallback_dir / p.name
        if candidate.exists():
            return candidate
        # Return the original path so the caller gets a clear FileNotFoundError
        # naming what they actually asked for.
        return p
