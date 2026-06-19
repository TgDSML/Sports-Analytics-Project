"""CSV manifest helpers for SoccerNet batch clips."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


MANIFEST_COLUMNS = [
    "clip_id",
    "split",
    "competition",
    "source",
    "game_id",
    "half",
    "video_path",
    "enabled",
    "notes",
]


@dataclass(frozen=True)
class ManifestRow:
    """One dataset manifest row."""

    clip_id: str
    split: str
    competition: str
    source: str
    game_id: str
    half: int
    video_path: Path
    enabled: bool
    notes: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, str]) -> "ManifestRow":
        """Build a typed manifest row from CSV text values."""
        return cls(
            clip_id=row["clip_id"],
            split=row.get("split", ""),
            competition=row.get("competition", ""),
            source=row.get("source", "SoccerNet"),
            game_id=row.get("game_id", ""),
            half=int(row.get("half") or 0),
            video_path=Path(row.get("video_path", "")),
            enabled=str(row.get("enabled", "true")).strip().lower()
            in {"1", "true", "yes", "y"},
            notes=row.get("notes", ""),
        )

    def to_dict(self) -> dict[str, str]:
        """Return a CSV-ready dict."""
        return {
            "clip_id": self.clip_id,
            "split": self.split,
            "competition": self.competition,
            "source": self.source,
            "game_id": self.game_id,
            "half": str(self.half),
            "video_path": str(self.video_path),
            "enabled": "true" if self.enabled else "false",
            "notes": self.notes,
        }


def read_manifest(path: Path, enabled_only: bool = False) -> list[ManifestRow]:
    """Read manifest rows from a CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Manifest missing column(s): {', '.join(sorted(missing))}"
            )
        rows = [ManifestRow.from_dict(row) for row in reader if row.get("clip_id")]

    if enabled_only:
        rows = [row for row in rows if row.enabled]
    return rows


def write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    """Write manifest rows to CSV with deterministic ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item.clip_id):
            writer.writerow(row.to_dict())


def find_manifest_row(path: Path, clip_id: str) -> ManifestRow:
    """Return one row by clip ID."""
    matches = [row for row in read_manifest(path) if row.clip_id == clip_id]
    if not matches:
        raise ValueError(f"Clip ID not found in manifest: {clip_id}")
    if len(matches) > 1:
        raise ValueError(f"Duplicate clip ID in manifest: {clip_id}")
    return matches[0]


def make_clip_id(game_id: str, half: int, resolution: str) -> str:
    """Build a deterministic filesystem-safe clip ID."""
    safe_game = (
        game_id.replace("/", "__")
        .replace("\\", "__")
        .replace(" ", "_")
        .replace("-", "_")
    )
    return f"{safe_game}__h{half}_{resolution}"
