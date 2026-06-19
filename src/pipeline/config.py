"""Pipeline config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = Path("configs/default_pipeline.yaml")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a small YAML config file.

    PyYAML is used when installed. A minimal parser is kept here so the batch
    wrapper works with the repository's current dependency list.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        return _parse_simple_yaml(text)

    loaded = yaml.safe_load(text)
    return loaded if isinstance(loaded, dict) else {}


def get_config_value(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Return a nested config value using dot notation."""
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the subset of YAML used by configs/default_pipeline.yaml."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    pending_list_key: tuple[int, dict[str, Any], str] | None = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if line.startswith("- "):
            if pending_list_key is None:
                raise ValueError("List item without a parent key in config")
            parent_indent, parent, key = pending_list_key
            if indent <= parent_indent:
                raise ValueError("Invalid list indentation in config")
            if not isinstance(parent.get(key), list):
                parent[key] = []
            parent[key].append(_parse_scalar(line[2:].strip()))
            continue

        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"Invalid config line: {raw_line}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        key = key.strip()
        value = value.strip()

        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
            pending_list_key = (indent, parent, key)
        else:
            parent[key] = _parse_scalar(value)
            pending_list_key = None

    return root


def _parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")
