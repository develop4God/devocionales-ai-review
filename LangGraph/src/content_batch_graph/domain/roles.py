"""
Resolve a configured role (config/roles.yml) into its persona/instructions data.

No role is hardcoded here — the flag pass is a role-agnostic engine fed whatever
role data this module resolves. Adding a new role means adding an entry to
roles.yml, not writing a new function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _REPO_ROOT / "config" / "roles.yml"

_config: dict[str, Any] | None = None


class Role(TypedDict):
    id: str
    name: str
    persona: str
    categories: list[str]


def _load_config() -> dict[str, Any]:
    global _config
    if _config is None:
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    return _config


def get_role(role_id: str | None = None) -> Role:
    """
    Returns the role data for the given role id, or the configured default_role if
    none is given. Raises if the role id isn't found.
    """
    config = _load_config()
    role_id = role_id or config["settings"]["default_role"]
    roles = {r["id"]: r for r in config["roles"]}
    if role_id not in roles:
        raise ValueError(f"Role '{role_id}' not found.")
    return roles[role_id]
