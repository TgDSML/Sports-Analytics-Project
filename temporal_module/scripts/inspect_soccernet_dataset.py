"""Inspect a local SoccerNet folder and build a read-only clip inventory."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}
ANNOTATION_EXTENSIONS = {".json", ".csv", ".txt"}
TRACKING_EXTENSIONS = {".csv", ".json", ".parquet", ".pkl", ".npy", ".npz"}
METADATA_EXTENSIONS = {".md", ".yaml", ".yml", ".txt", ".json"}

FILE_INVENTORY_COLUMNS = [
    "absolute_path",
    "relative_path",
    "filename",
    "extension",
    "parent_relative_dir",
    "is_video",
    "is_annotation",
    "is_tracking_like",
    "is_metadata",
    "file_size_bytes",
]

CLIP_INVENTORY_COLUMNS = [
    "inventory_clip_id",
    "absolute_path",
    "relative_video_path",
    "filename",
    "extension",
    "parent_relative_dir",
    "width",
    "height",
    "fps",
    "frame_count",
    "duration_seconds",
    "readable_status",
    "error_message",
    "annotation_paths",
    "tracking_paths",
    "metadata_paths",
    "period_token",
]

SCHEMA_COLUMNS = [
    "file_path",
    "relative_path",
    "extension",
    "schema_kind",
    "schema_summary",
    "error_message",
]

ERROR_COLUMNS = [
    "path",
    "relative_path",
    "error_stage",
    "error_message",
]

PILOT_COLUMNS = [
    "inventory_clip_id",
    "video_path",
    "relative_video_path",
    "fps",
    "frame_count",
    "duration_seconds",
    "width",
    "height",
    "annotation_paths",
    "tracking_paths",
    "readiness_status",
    "selection_reason",
    "manual_include",
    "manual_exclude_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a local SoccerNet folder and create candidate inventory files."
    )
    parser.add_argument("--soccernet-root", required=True)
    parser.add_argument("--output-root", default=str(Path("temporal_module") / "data" / "soccernet_inventory"))
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--include-hidden", action="store_true")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_hidden_path(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return any(part.startswith(".") for part in relative.parts)


def safe_relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def parent_relative_dir(path: Path, root: Path) -> str:
    parent = path.parent
    if parent == root:
        return "."
    return safe_relative(parent, root)


def inventory_clip_id(relative_dir: str, filename: str) -> str:
    raw = f"{relative_dir}/{filename}" if relative_dir != "." else filename
    stem = str(Path(raw).with_suffix(""))
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return normalized or "clip"


def period_token(path_text: str) -> str:
    normalized = path_text.replace("\\", "/").casefold()
    matches = re.findall(r"(?<![a-z0-9])(h[12]|half[_ -]?[12]|[12](?:st|nd)?[_ -]?half)(?![a-z0-9])", normalized)
    if matches:
        return matches[-1].replace(" ", "_").replace("-", "_")
    filename = Path(path_text).name.casefold()
    if filename.startswith("1_") or filename in {"1.mp4", "1.mkv", "1.avi", "1.mov"}:
        return "h1"
    if filename.startswith("2_") or filename in {"2.mp4", "2.mkv", "2.avi", "2.mov"}:
        return "h2"
    return "unknown_period"


def scan_files(root: Path, include_hidden: bool) -> tuple[list[Path], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    files: list[Path] = []
    try:
        iterator = root.rglob("*")
        for path in iterator:
            if not include_hidden and is_hidden_path(path, root):
                continue
            try:
                if path.is_file():
                    files.append(path)
            except OSError as error:
                errors.append(error_row(path, root, "stat", str(error)))
    except OSError as error:
        errors.append(error_row(root, root, "scan", str(error)))
    return sorted(files, key=lambda value: value.as_posix().casefold()), errors


def classify_file(path: Path, root: Path) -> dict[str, Any]:
    extension = path.suffix.casefold()
    try:
        size = path.stat().st_size
    except OSError:
        size = ""
    return {
        "absolute_path": str(path.resolve()),
        "relative_path": safe_relative(path, root),
        "filename": path.name,
        "extension": extension,
        "parent_relative_dir": parent_relative_dir(path, root),
        "is_video": int(extension in VIDEO_EXTENSIONS),
        "is_annotation": int(extension in ANNOTATION_EXTENSIONS),
        "is_tracking_like": int(extension in TRACKING_EXTENSIONS),
        "is_metadata": int(extension in METADATA_EXTENSIONS),
        "file_size_bytes": size,
    }


def read_video_metadata(path: Path) -> tuple[dict[str, Any], str]:
    try:
        import cv2  # type: ignore
    except Exception as error:
        return empty_video_metadata(), f"OpenCV import failed: {error}"

    capture = None
    try:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return empty_video_metadata(), "OpenCV could not open video"
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps and fps > 0 and frame_count and frame_count > 0 else ""
        metadata = {
            "width": numeric_or_blank(width),
            "height": numeric_or_blank(height),
            "fps": numeric_or_blank(fps),
            "frame_count": numeric_or_blank(frame_count),
            "duration_seconds": numeric_or_blank(duration),
        }
        return metadata, ""
    except Exception as error:
        return empty_video_metadata(), str(error)
    finally:
        if capture is not None:
            capture.release()


def numeric_or_blank(value: Any) -> Any:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if number <= 0:
        return ""
    if number.is_integer():
        return int(number)
    return round(number, 6)


def empty_video_metadata() -> dict[str, str]:
    return {
        "width": "",
        "height": "",
        "fps": "",
        "frame_count": "",
        "duration_seconds": "",
    }


def files_by_parent(file_rows: list[dict[str, Any]], key: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in file_rows:
        if int(row[key]) == 1:
            grouped[str(row["parent_relative_dir"])].append(str(row["relative_path"]))
    for paths in grouped.values():
        paths.sort()
    return grouped


def build_clip_rows(root: Path, file_rows: list[dict[str, Any]], max_clips: int | None) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    annotation_by_parent = files_by_parent(file_rows, "is_annotation")
    tracking_by_parent = files_by_parent(file_rows, "is_tracking_like")
    metadata_by_parent = files_by_parent(file_rows, "is_metadata")
    video_rows = [row for row in file_rows if int(row["is_video"]) == 1]
    if max_clips is not None:
        video_rows = video_rows[: max(0, max_clips)]

    clip_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for row in video_rows:
        path = Path(row["absolute_path"])
        metadata, error_message = read_video_metadata(path)
        readable = error_message == ""
        relative_path = str(row["relative_path"])
        parent = str(row["parent_relative_dir"])
        if error_message:
            errors.append(error_row(path, root, "video_read", error_message))
        clip_rows.append(
            {
                "inventory_clip_id": inventory_clip_id(parent, str(row["filename"])),
                "absolute_path": str(path),
                "relative_video_path": relative_path,
                "filename": row["filename"],
                "extension": row["extension"],
                "parent_relative_dir": parent,
                "width": metadata["width"],
                "height": metadata["height"],
                "fps": metadata["fps"],
                "frame_count": metadata["frame_count"],
                "duration_seconds": metadata["duration_seconds"],
                "readable_status": "readable" if readable else "unreadable",
                "error_message": error_message,
                "annotation_paths": ";".join(annotation_by_parent.get(parent, [])),
                "tracking_paths": ";".join(tracking_by_parent.get(parent, [])),
                "metadata_paths": ";".join(metadata_by_parent.get(parent, [])),
                "period_token": period_token(relative_path),
            }
        )
    return clip_rows, errors


def schema_summary(path: Path) -> tuple[str, str, str]:
    extension = path.suffix.casefold()
    try:
        if extension == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as csv_file:
                reader = csv.reader(csv_file)
                header = next(reader, [])
            return "csv_header", ",".join(header), ""
        if extension == ".json":
            text = path.read_text(encoding="utf-8", errors="replace")
            payload = json.loads(text)
            if isinstance(payload, dict):
                return "json_object_keys", ",".join(sorted(str(key) for key in payload.keys())[:50]), ""
            if isinstance(payload, list):
                return "json_list", f"items={len(payload)}", ""
            return "json_scalar", type(payload).__name__, ""
        if extension == ".txt":
            with path.open(encoding="utf-8", errors="replace") as text_file:
                lines = [line.strip() for _, line in zip(range(5), text_file)]
            return "txt_preview", " | ".join(line for line in lines if line), ""
    except Exception as error:
        return extension.lstrip(".") or "unknown", "", str(error)
    return extension.lstrip(".") or "unknown", "", ""


def build_schema_rows(root: Path, file_rows: list[dict[str, Any]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for row in file_rows:
        if int(row["is_annotation"]) != 1:
            continue
        path = Path(row["absolute_path"])
        kind, summary, error_message = schema_summary(path)
        if error_message:
            errors.append(error_row(path, root, "schema_read", error_message))
        rows.append(
            {
                "file_path": str(path),
                "relative_path": str(row["relative_path"]),
                "extension": str(row["extension"]),
                "schema_kind": kind,
                "schema_summary": summary,
                "error_message": error_message,
            }
        )
    return rows, errors


def duration_bucket(row: dict[str, Any]) -> str:
    try:
        duration = float(row.get("duration_seconds", ""))
    except (TypeError, ValueError):
        return "unknown_duration"
    if duration < 300:
        return "short"
    if duration < 1800:
        return "medium"
    return "long"


def candidate_score(row: dict[str, Any]) -> int:
    score = 0
    score += 100 if row.get("readable_status") == "readable" else 0
    score += 20 if row.get("annotation_paths") else 0
    score += 10 if row.get("tracking_paths") else 0
    return score


def diversity_group(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("parent_relative_dir", "unknown_parent")),
            duration_bucket(row),
            str(row.get("period_token", "unknown_period")),
        ]
    )


def selection_reason(row: dict[str, Any]) -> str:
    reasons = ["pilot_candidate_only", "readable_video"]
    if row.get("annotation_paths"):
        reasons.append("nearby_annotation_files")
    if row.get("tracking_paths"):
        reasons.append("nearby_tracking_files")
    reasons.append(f"parent_folder={row.get('parent_relative_dir', '')}")
    reasons.append(f"duration_bucket={duration_bucket(row)}")
    reasons.append(f"period_token={row.get('period_token', 'unknown_period')}")
    return ";".join(reasons)


def select_pilot_candidates(clip_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    readable = [row for row in clip_rows if row.get("readable_status") == "readable"]
    sorted_rows = sorted(
        readable,
        key=lambda row: (
            -candidate_score(row),
            str(row.get("parent_relative_dir", "")).casefold(),
            duration_bucket(row),
            str(row.get("period_token", "")).casefold(),
            str(row.get("relative_video_path", "")).casefold(),
        ),
    )
    groups: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in sorted_rows:
        groups[diversity_group(row)].append(row)
    selected: list[dict[str, Any]] = []
    while groups and len(selected) < 30:
        for group in sorted(list(groups)):
            if len(selected) >= 30:
                break
            if not groups[group]:
                groups.pop(group, None)
                continue
            selected.append(groups[group].popleft())
            if not groups.get(group):
                groups.pop(group, None)
    return [
        {
            "inventory_clip_id": row["inventory_clip_id"],
            "video_path": row["absolute_path"],
            "relative_video_path": row["relative_video_path"],
            "fps": row["fps"],
            "frame_count": row["frame_count"],
            "duration_seconds": row["duration_seconds"],
            "width": row["width"],
            "height": row["height"],
            "annotation_paths": row["annotation_paths"],
            "tracking_paths": row["tracking_paths"],
            "readiness_status": "ready_for_video_pilot",
            "selection_reason": selection_reason(row),
            "manual_include": "",
            "manual_exclude_reason": "",
        }
        for row in selected
    ]


def error_row(path: Path, root: Path, stage: str, message: str) -> dict[str, str]:
    try:
        relative = safe_relative(path, root)
    except ValueError:
        relative = str(path)
    return {
        "path": str(path),
        "relative_path": relative,
        "error_stage": stage,
        "error_message": message,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def summary_payload(
    root: Path,
    file_rows: list[dict[str, Any]],
    clip_rows: list[dict[str, Any]],
    pilot_rows: list[dict[str, Any]],
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    extensions = sorted({str(row["extension"]) for row in file_rows if row.get("extension")})
    total_videos = sum(int(row["is_video"]) == 1 for row in file_rows)
    annotation_count = sum(int(row["is_annotation"]) == 1 for row in file_rows)
    tracking_count = sum(int(row["is_tracking_like"]) == 1 for row in file_rows)
    readable = sum(row.get("readable_status") == "readable" for row in clip_rows)
    unreadable = sum(row.get("readable_status") != "readable" for row in clip_rows)
    return {
        "supplied_soccernet_root": str(root.resolve()),
        "discovery_timestamp": utc_now_iso(),
        "total_files_scanned": len(file_rows),
        "total_videos": total_videos,
        "readable_videos": readable,
        "unreadable_videos": unreadable,
        "grouped_clips": len(clip_rows),
        "candidate_clips_selected": len(pilot_rows),
        "annotation_file_count": annotation_count,
        "tracking_file_count": tracking_count,
        "detected_extensions": extensions,
        "error_count": len(errors),
    }


def inspect_dataset(args: argparse.Namespace) -> int:
    root = Path(args.soccernet_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"SoccerNet root is not a directory: {root}")
    output_root = Path(args.output_root)

    files, scan_errors = scan_files(root, args.include_hidden)
    file_rows = [classify_file(path, root) for path in files]
    clip_rows, video_errors = build_clip_rows(root, file_rows, args.max_clips)
    schema_rows, schema_errors = build_schema_rows(root, file_rows)
    pilot_rows = select_pilot_candidates(clip_rows)
    errors = scan_errors + video_errors + schema_errors

    write_csv(output_root / "soccernet_file_inventory.csv", file_rows, FILE_INVENTORY_COLUMNS)
    write_csv(output_root / "soccernet_clip_inventory.csv", clip_rows, CLIP_INVENTORY_COLUMNS)
    write_csv(output_root / "soccernet_annotation_schema_inventory.csv", schema_rows, SCHEMA_COLUMNS)
    write_csv(output_root / "soccernet_inventory_errors.csv", errors, ERROR_COLUMNS)
    write_csv(output_root / "soccernet_pilot_candidates.csv", pilot_rows, PILOT_COLUMNS)
    write_json(
        output_root / "soccernet_inventory_summary.json",
        summary_payload(root, file_rows, clip_rows, pilot_rows, errors),
    )

    print(f"SoccerNet inventory written to: {output_root}")
    print(f"Videos inspected: {len(clip_rows)}")
    print(f"Pilot candidates selected: {len(pilot_rows)}")
    print("No downloads were performed.")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return inspect_dataset(args)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
