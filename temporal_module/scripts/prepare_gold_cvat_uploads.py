"""Prepare unique CVAT upload copies for ready gold clips."""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST = Path("temporal_module") / "data" / "gold_event_project" / "manifests" / "gold_clip_manifest.csv"
DEFAULT_UPLOAD_DIR = Path("temporal_module") / "data" / "gold_event_project" / "cvat_uploads"
DEFAULT_INVENTORY = Path("temporal_module") / "data" / "gold_event_project" / "manifests" / "gold_cvat_upload_inventory.csv"
HEADER = [
    "clip_id",
    "video_filename",
    "source_clip_path",
    "cvat_upload_path",
    "source_exists",
    "upload_copy_exists",
    "status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy ready gold clips into the CVAT upload folder.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--upload-dir", type=Path, default=DEFAULT_UPLOAD_DIR)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


def write_inventory(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=HEADER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def main() -> int:
    args = parse_args()
    manifest_rows = read_manifest(args.manifest)
    filename_counts = Counter(row.get("video_filename", "") for row in manifest_rows)
    inventory: list[dict[str, Any]] = []
    fatal = False

    for row in manifest_rows:
        clip_id = row.get("clip_id", "")
        video_filename = row.get("video_filename", "")
        source_clip = Path(row.get("source_clip_path", ""))
        destination = args.upload_dir / video_filename
        annotation_status = row.get("annotation_status", "")
        source_exists = source_clip.exists()
        destination_exists = destination.exists()
        status = "ready"
        error = ""

        if annotation_status != "ready_for_cvat":
            status = "not_ready_for_cvat"
        elif filename_counts[video_filename] > 1:
            status = "duplicate_destination"
            error = "duplicate video_filename in manifest"
            fatal = True
        elif not source_exists:
            status = "missing_source_clip"
            error = "source_clip_path does not exist"
            fatal = True
        elif destination_exists and not args.overwrite:
            status = "already_exists"
        elif not args.dry_run:
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_clip, destination)
                status = "copied"
                destination_exists = True
            except OSError as exc:
                status = "failed"
                error = str(exc)
                fatal = True

        inventory.append(
            {
                "clip_id": clip_id,
                "video_filename": video_filename,
                "source_clip_path": str(source_clip),
                "cvat_upload_path": str(destination),
                "source_exists": bool_text(source_exists),
                "upload_copy_exists": bool_text(destination_exists or (status == "copied")),
                "status": status,
                "error": error,
            }
        )

    write_inventory(args.inventory, inventory)
    counts = Counter(row["status"] for row in inventory)
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")
    print(f"Inventory: {args.inventory}")
    return 1 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
