"""Tiny flat YAML config reader used by the model starter scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_config(path: str | Path) -> dict[str, Any]:
    """Load the flat `key: value` config files used in personal sandboxes.

    This intentionally avoids adding a YAML dependency. Keep configs flat; if a
    model needs richer config later, that owner can add PyYAML in their branch.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    config: dict[str, Any] = {}
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-") or raw_line[:1].isspace():
            raise ValueError(f"{path}:{line_number}: only flat `key: value` entries are supported")
        if ":" not in line:
            raise ValueError(f"{path}:{line_number}: expected `key: value`")
        key, value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{path}:{line_number}: empty config key")
        config[key] = _parse_scalar(value)
    return config
