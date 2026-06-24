"""I/O and schema helpers for the isolated temporal module."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = (PROJECT_ROOT / "outputs").resolve()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def reject_outputs_path(path: str | Path) -> None:
    resolved = resolve_path(path)
    try:
        resolved.relative_to(OUTPUTS_ROOT)
    except ValueError:
        return
    raise ValueError(f"Refusing to write inside outputs/: {resolved}")


def ensure_output_parent(path: str | Path) -> Path:
    resolved = resolve_path(path)
    reject_outputs_path(resolved)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def read_csv_with_header(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Input CSV not found: {resolved}")
    header = list(pd.read_csv(resolved, nrows=0).columns)
    df = pd.read_csv(resolved)
    return df, header


def write_json(path: str | Path, payload: dict) -> Path:
    resolved = ensure_output_parent(path)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return resolved


def print_header_report(headers: dict[str, list[str]]) -> None:
    print("Detected source headers:")
    for name, columns in headers.items():
        print(f"- {name}: {', '.join(columns)}")


def print_row_counts(row_counts: dict[str, int]) -> None:
    print("Input row counts:")
    for name, count in row_counts.items():
        print(f"- {name}: {count}")


def format_missing_inputs(paths: dict[str, Path]) -> list[str]:
    return [name for name, path in paths.items() if not Path(path).exists()]


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result

