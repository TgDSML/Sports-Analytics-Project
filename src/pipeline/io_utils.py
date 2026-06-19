"""Filesystem helpers for per-clip pipeline outputs."""

from __future__ import annotations

from pathlib import Path


CLIP_OUTPUT_FOLDERS = [
    "detections",
    "tracks",
    "teams",
    "possession",
    "carries",
    "interceptions",
    "tactical",
    "visualizations",
    "logs",
    "preprocessed",
]


def ensure_clip_output_dirs(outputs_root: Path, clip_id: str) -> dict[str, Path]:
    """Create and return standard per-clip output directories."""
    clip_root = outputs_root / clip_id
    directories = {"root": clip_root}
    for folder in CLIP_OUTPUT_FOLDERS:
        path = clip_root / folder
        path.mkdir(parents=True, exist_ok=True)
        directories[folder] = path
    return directories


def resolve_repo_path(path: Path) -> Path:
    """Resolve a path relative to the repository root."""
    return path if path.is_absolute() else Path.cwd() / path
