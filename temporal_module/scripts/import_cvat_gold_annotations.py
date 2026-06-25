"""Import native CVAT-for-video gold event tracks into canonical intervals."""

from __future__ import annotations

import argparse
import csv
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

VALID_LABELS = {"carry", "pass", "turnover", "shot", "uncertain"}
DEFAULT_GOLD_ROOT = Path("temporal_module") / "data" / "gold_event_project"
DEFAULT_MANIFEST = DEFAULT_GOLD_ROOT / "manifests" / "gold_clip_manifest.csv"
DEFAULT_EXPORT_DIR = DEFAULT_GOLD_ROOT / "cvat_exports"
DEFAULT_OUTPUT = DEFAULT_GOLD_ROOT / "annotations" / "gold_event_intervals.csv"
HEADER = [
    "clip_id",
    "event_id",
    "event_type",
    "start_frame",
    "end_frame",
    "start_seconds",
    "end_seconds",
    "cvat_task_name",
    "annotator",
    "confidence",
    "uncertain",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CVAT native video exports into gold event intervals.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        return {row["clip_id"]: row for row in csv.DictReader(csv_file)}


def find_annotation_xml(zip_path: Path) -> bytes:
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".xml")]
        preferred = [name for name in names if Path(name).name.lower() == "annotations.xml"]
        if not names:
            raise ValueError("no XML file found in CVAT export")
        return archive.read((preferred or names)[0])


def task_name_from_root(root: ET.Element) -> str:
    name_node = root.find("./meta/task/name")
    return (name_node.text or "").strip() if name_node is not None else ""


def task_size_from_root(root: ET.Element) -> int:
    size_node = root.find("./meta/task/size")
    try:
        return int(size_node.text or 0) if size_node is not None else 0
    except ValueError:
        return 0


def resolve_fps(root: ET.Element, manifest_row: dict[str, str], warnings: list[str]) -> float:
    try:
        fps = float(manifest_row.get("fps") or 0.0)
    except ValueError:
        fps = 0.0
    if fps > 0:
        return fps

    try:
        duration = float(manifest_row.get("duration_seconds") or 0.0)
    except ValueError:
        duration = 0.0
    task_size = task_size_from_root(root)
    if duration > 0 and task_size > 0:
        derived_fps = task_size / duration
        warnings.append(f"manifest fps missing; derived fps={derived_fps:.6f} from CVAT task size {task_size} and duration {duration:.3f}s")
        return derived_fps

    warnings.append("manifest fps missing; using fallback fps=25.0")
    return 25.0


def task_name_is_compatible(task_name: str, expected_task: str, clip_id: str) -> bool:
    if not task_name or not expected_task:
        return True
    if task_name == expected_task:
        return True
    if task_name == "gold_v1__{{file_name}}":
        return True
    if task_name == f"{clip_id}.mp4":
        return True
    if task_name == f"gold_v1__{clip_id}.mp4":
        return True
    return False


def active_intervals(track: ET.Element) -> list[tuple[int, int]]:
    frames: list[int] = []
    for box in track.findall("box"):
        outside = box.attrib.get("outside", "0") in {"1", "true", "True"}
        if outside:
            continue
        frame_text = box.attrib.get("frame")
        if frame_text is None:
            continue
        frames.append(int(frame_text))
    if not frames:
        return []
    frames = sorted(set(frames))
    intervals: list[tuple[int, int]] = []
    start = prev = frames[0]
    for frame in frames[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        intervals.append((start, prev))
        start = prev = frame
    intervals.append((start, prev))
    return intervals


def convert_export(zip_path: Path, clip_id: str, manifest_row: dict[str, str]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    root = ET.fromstring(find_annotation_xml(zip_path))
    task_name = task_name_from_root(root)
    expected_task = manifest_row.get("cvat_task_name", "")
    errors: list[str] = []
    warnings: list[str] = []
    if not task_name_is_compatible(task_name, expected_task, clip_id):
        warnings.append(f"task name mismatch: export={task_name}, manifest={expected_task}")
    fps = resolve_fps(root, manifest_row, warnings)

    rows: list[dict[str, Any]] = []
    seen = set()
    interval_by_clip: list[tuple[int, int, str]] = []
    event_counter = 1
    for track in root.findall("track"):
        label = track.attrib.get("label", "")
        if label not in VALID_LABELS:
            continue
        for start_frame, end_frame in active_intervals(track):
            if end_frame < start_frame or start_frame < 0:
                errors.append(f"invalid bounds for {label}: {start_frame}-{end_frame}")
                continue
            duplicate_key = (label, start_frame, end_frame)
            if duplicate_key in seen:
                errors.append(f"duplicate event: {label} {start_frame}-{end_frame}")
                continue
            seen.add(duplicate_key)
            for prev_start, prev_end, prev_label in interval_by_clip:
                if not (end_frame < prev_start or start_frame > prev_end):
                    warnings.append(f"overlap: {label} {start_frame}-{end_frame} with {prev_label} {prev_start}-{prev_end}")
            interval_by_clip.append((start_frame, end_frame, label))
            uncertain = label == "uncertain"
            rows.append(
                {
                    "clip_id": clip_id,
                    "event_id": f"{clip_id}__event_{event_counter:04d}",
                    "event_type": label,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_seconds": f"{(start_frame / fps):.6f}" if fps > 0 else "",
                    "end_seconds": f"{(end_frame / fps):.6f}" if fps > 0 else "",
                    "cvat_task_name": expected_task,
                    "annotator": "",
                    "confidence": "",
                    "uncertain": "true" if uncertain else "false",
                    "notes": "imported_from=" + zip_path.name,
                }
            )
            event_counter += 1
    return rows, warnings, errors


def write_output(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=HEADER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    manifest = read_manifest(args.manifest)
    exports = sorted(args.export_dir.glob("cvat_export__*.zip")) if args.export_dir.exists() else []
    if not exports:
        print("no exports found")
        if not args.dry_run:
            write_output(args.output, [])
        return 0

    all_rows: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    all_errors: list[str] = []
    for export in exports:
        clip_id = export.stem.removeprefix("cvat_export__")
        if clip_id not in manifest:
            all_errors.append(f"unknown clip export: {export.name}")
            continue
        rows, warnings, errors = convert_export(export, clip_id, manifest[clip_id])
        all_rows.extend(rows)
        all_warnings.extend(f"{clip_id}: {warning}" for warning in warnings)
        all_errors.extend(f"{clip_id}: {error}" for error in errors)

    print(f"exports found: {len(exports)}")
    print(f"intervals parsed: {len(all_rows)}")
    if all_warnings:
        print("validation warnings:")
        for warning in all_warnings:
            print("- " + warning)
    if all_errors:
        print("validation errors:")
        for error in all_errors:
            print("- " + error)
    if not args.dry_run and not all_errors:
        write_output(args.output, all_rows)
        print(f"wrote: {args.output}")
    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
