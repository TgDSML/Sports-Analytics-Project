"""Input/output helpers for generated analysis artifacts."""

import csv
from pathlib import Path


DETECTION_CSV_COLUMNS = [
    "frame",
    "timestamp",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "width",
    "height",
]

TRACK_CSV_COLUMNS = [
    "frame",
    "timestamp",
    "track_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
]


def write_detections_csv(rows: list[dict], csv_path: str | Path) -> None:
    """Write detection rows to a CSV file with a stable header."""
    _write_csv(rows, csv_path, DETECTION_CSV_COLUMNS)


def write_tracks_csv(rows: list[dict], csv_path: str | Path) -> None:
    """Write tracked player rows to a CSV file with a stable header."""
    _write_csv(rows, csv_path, TRACK_CSV_COLUMNS)


def _write_csv(rows: list[dict], csv_path: str | Path, columns: list[str]) -> None:
    """Write rows to CSV using explicit columns."""
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
