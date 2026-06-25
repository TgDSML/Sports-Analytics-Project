"""Build CVAT-readiness manifests for the gold event project."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2

from src.pipeline.config import DEFAULT_CONFIG_PATH, get_config_value, load_config
from src.pipeline.manifest import MANIFEST_COLUMNS, read_manifest

GOLD_ROOT = Path("temporal_module") / "data" / "gold_event_project"
MANIFEST_DIR = GOLD_ROOT / "manifests"
DERIVED_ROOT = Path("temporal_module") / "data" / "derived"
OUTPUTS_ROOT = Path("outputs")

GOLD_CLIP_HEADER = [
    "clip_id",
    "video_filename",
    "source_video",
    "source_clip_path",
    "season",
    "game_id",
    "half",
    "source_start_seconds",
    "duration_seconds",
    "fps",
    "cvat_task_name",
    "annotation_status",
    "source_manifest_split",
    "notes",
]
ASSIGNMENT_HEADER = [
    "clip_id",
    "cvat_task_name",
    "video_filename",
    "annotator",
    "reviewer",
    "annotation_status",
    "review_status",
]
READINESS_HEADER = [
    "clip_id",
    "source_video_exists",
    "clip_mp4_exists",
    "tracks_exists",
    "ball_tracks_exists",
    "teams_exists",
    "possession_exists",
    "possession_debug_exists",
    "carries_exists",
    "interceptions_exists",
    "temporal_frames_exists",
    "clip_duration_seconds",
    "clip_fps",
    "status",
    "missing_artifacts",
    "error",
]
ALIGNMENT_HEADER = [
    "clip_id",
    "clip_path",
    "duration_seconds",
    "fps",
    "video_frame_count",
    "temporal_first_frame",
    "temporal_last_frame",
    "temporal_frame_count",
    "alignment_status",
    "notes",
]

REQUIRED_OUTPUTS = {
    "clip_mp4_exists": Path("preprocessed") / "clip.mp4",
    "tracks_exists": Path("tracks") / "tracks.csv",
    "ball_tracks_exists": Path("tracks") / "ball_tracks.csv",
    "teams_exists": Path("teams") / "player_teams.csv",
    "possession_exists": Path("possession") / "possession.csv",
    "possession_debug_exists": Path("possession") / "possession_debug.csv",
    "carries_exists": Path("carries") / "carries.csv",
    "interceptions_exists": Path("interceptions") / "interceptions.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gold CVAT readiness manifests.")
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--outputs-root", type=Path, default=OUTPUTS_ROOT)
    parser.add_argument("--derived-root", type=Path, default=DERIVED_ROOT)
    parser.add_argument("--gold-root", type=Path, default=GOLD_ROOT)
    return parser.parse_args()


def find_source_manifest(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidates: list[Path] = []
    roots = [Path("data") / "manifests", Path("temporal_module") / "data"]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            try:
                with path.open(newline="", encoding="utf-8-sig") as csv_file:
                    reader = csv.reader(csv_file)
                    header = next(reader, [])
            except (OSError, UnicodeDecodeError):
                continue
            if header == MANIFEST_COLUMNS:
                candidates.append(path)
    if not candidates:
        raise FileNotFoundError("No compatible source manifest found.")
    candidates.sort(key=lambda item: (0 if item.as_posix() == "data/manifests/dataset_manifest.csv" else 1, len(item.parts), item.as_posix()))
    return candidates[0]


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def write_csv(path: Path, rows: list[dict[str, Any]], header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def video_metadata(path: Path) -> tuple[float, float, int, str]:
    if not path.exists():
        return 0.0, 0.0, 0, "missing clip"
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0.0, 0.0, 0, "could not open clip"
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return duration, fps, frame_count, ""


def temporal_frame_bounds(path: Path) -> tuple[int | None, int | None, int, str]:
    if not path.exists():
        return None, None, 0, "missing temporal_frames.csv"
    try:
        with path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None or "frame" not in reader.fieldnames:
                return None, None, 0, "temporal_frames.csv missing frame column"
            frames = [int(float(row["frame"])) for row in reader if row.get("frame") not in {None, ""}]
    except (OSError, ValueError) as error:
        return None, None, 0, str(error)
    if not frames:
        return None, None, 0, "temporal_frames.csv has no frame rows"
    return min(frames), max(frames), len(frames), ""


def season_from_game_id(game_id: str) -> str:
    parts = game_id.replace("\\", "/").split("/")
    return parts[1] if len(parts) > 1 else ""


def validate_source_rows(rows: list[Any]) -> dict[str, str]:
    errors: dict[str, str] = {}
    counts = Counter(row.clip_id for row in rows)
    for row in rows:
        messages: list[str] = []
        if counts[row.clip_id] > 1:
            messages.append("duplicate clip_id")
        if row.half not in {1, 2}:
            messages.append("half is not 1 or 2")
        name = row.video_path.name
        if name not in {"1_720p.mkv", "2_720p.mkv"}:
            messages.append("source is not 1_720p.mkv or 2_720p.mkv")
        elif name[0] != str(row.half):
            messages.append("half does not match source filename")
        if not row.video_path.exists():
            messages.append("source video missing")
        normalized = row.video_path.as_posix().replace("\\", "/")
        if row.game_id and row.game_id.replace("\\", "/") not in normalized:
            messages.append("video_path does not contain game_id")
        if messages:
            errors[row.clip_id] = "; ".join(messages)
    return errors


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    duration_expected = float(get_config_value(config, "video.clip_duration_seconds", 30))
    source_manifest = find_source_manifest(args.source_manifest)
    source_rows = read_manifest(source_manifest, enabled_only=True)
    source_errors = validate_source_rows(source_rows)

    outputs_root = args.outputs_root
    derived_root = args.derived_root
    manifest_dir = args.gold_root / "manifests"
    upload_dir = args.gold_root / "cvat_uploads"
    export_dir = args.gold_root / "cvat_exports"

    readiness_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    blocked: dict[str, str] = {}

    for row in source_rows:
        clip_root = outputs_root / row.clip_id
        clip_path = clip_root / "preprocessed" / "clip.mp4"
        temporal_path = derived_root / row.clip_id / "temporal_frames.csv"
        duration, fps, frame_count, video_error = video_metadata(clip_path)
        first_frame, last_frame, temporal_count, temporal_error = temporal_frame_bounds(temporal_path)

        exists_map = {name: (clip_root / rel_path).exists() for name, rel_path in REQUIRED_OUTPUTS.items()}
        source_exists = row.video_path.exists()
        temporal_exists = temporal_path.exists()
        missing = [name for name, exists in exists_map.items() if not exists]
        if not source_exists:
            missing.append("source_video")
        if row.clip_id in source_errors:
            missing.append("source_manifest_validation")

        if row.clip_id in source_errors or not source_exists:
            status = "blocked"
            error = source_errors.get(row.clip_id, "source video missing")
        elif missing:
            status = "needs_original_pipeline"
            error = ""
        elif not temporal_exists:
            status = "needs_temporal_sweep"
            error = ""
        else:
            status = "ready_for_cvat"
            error = ""

        if not clip_path.exists() or video_error:
            alignment_status = "blocked"
            alignment_notes = video_error or "missing clip"
        elif not temporal_exists:
            alignment_status = "missing_temporal_frames"
            alignment_notes = temporal_error
        else:
            notes: list[str] = []
            alignment_status = "aligned"
            if abs(duration - duration_expected) > 1.0:
                alignment_status = "duration_mismatch"
                notes.append(f"expected about {duration_expected:.3f}s")
            if fps <= 0:
                alignment_status = "fps_mismatch"
                notes.append("fps unavailable")
            if first_frame not in {0, 1}:
                alignment_status = "frame_count_mismatch"
                notes.append("temporal frames do not start at 0 or 1")
            if frame_count and temporal_count and abs(frame_count - temporal_count) > max(2, int(round(fps)) if fps > 0 else 2):
                alignment_status = "frame_count_mismatch"
                notes.append("temporal frame count differs from video frame count")
            alignment_notes = "; ".join(notes)

        annotation_status = "ready_for_cvat" if status == "ready_for_cvat" and alignment_status == "aligned" else ("planned" if status in {"needs_original_pipeline", "needs_temporal_sweep"} else "blocked")
        if annotation_status != "ready_for_cvat":
            reason = error or "; ".join(missing) or alignment_notes or status
            blocked[row.clip_id] = reason

        readiness_rows.append(
            {
                "clip_id": row.clip_id,
                "source_video_exists": bool_text(source_exists),
                "clip_mp4_exists": bool_text(exists_map["clip_mp4_exists"]),
                "tracks_exists": bool_text(exists_map["tracks_exists"]),
                "ball_tracks_exists": bool_text(exists_map["ball_tracks_exists"]),
                "teams_exists": bool_text(exists_map["teams_exists"]),
                "possession_exists": bool_text(exists_map["possession_exists"]),
                "possession_debug_exists": bool_text(exists_map["possession_debug_exists"]),
                "carries_exists": bool_text(exists_map["carries_exists"]),
                "interceptions_exists": bool_text(exists_map["interceptions_exists"]),
                "temporal_frames_exists": bool_text(temporal_exists),
                "clip_duration_seconds": f"{duration:.6f}" if clip_path.exists() else "",
                "clip_fps": f"{fps:.6f}" if clip_path.exists() else "",
                "status": status,
                "missing_artifacts": ";".join(missing + ([] if temporal_exists else ["temporal_frames"])),
                "error": error,
            }
        )
        alignment_rows.append(
            {
                "clip_id": row.clip_id,
                "clip_path": str(clip_path),
                "duration_seconds": f"{duration:.6f}" if clip_path.exists() else "",
                "fps": f"{fps:.6f}" if clip_path.exists() else "",
                "video_frame_count": frame_count if clip_path.exists() else "",
                "temporal_first_frame": "" if first_frame is None else first_frame,
                "temporal_last_frame": "" if last_frame is None else last_frame,
                "temporal_frame_count": temporal_count if temporal_exists else "",
                "alignment_status": alignment_status,
                "notes": alignment_notes,
            }
        )
        video_filename = f"{row.clip_id}.mp4"
        gold_rows.append(
            {
                "clip_id": row.clip_id,
                "video_filename": video_filename,
                "source_video": str(row.video_path),
                "source_clip_path": str(clip_path),
                "season": season_from_game_id(row.game_id),
                "game_id": row.game_id,
                "half": row.half,
                "source_start_seconds": "0",
                "duration_seconds": f"{duration_expected:.6f}".rstrip("0").rstrip("."),
                "fps": f"{fps:.6f}" if clip_path.exists() else "",
                "cvat_task_name": f"gold_v1__{video_filename}",
                "annotation_status": annotation_status,
                "source_manifest_split": row.split,
                "notes": "source_manifest=" + source_manifest.as_posix(),
            }
        )
        assignment_rows.append(
            {
                "clip_id": row.clip_id,
                "cvat_task_name": f"gold_v1__{video_filename}",
                "video_filename": video_filename,
                "annotator": "",
                "reviewer": "",
                "annotation_status": annotation_status if annotation_status == "ready_for_cvat" else "blocked",
                "review_status": "not_started",
            }
        )

    write_csv(manifest_dir / "gold_clip_manifest.csv", gold_rows, GOLD_CLIP_HEADER)
    write_csv(manifest_dir / "cvat_assignment_manifest.csv", assignment_rows, ASSIGNMENT_HEADER)
    write_csv(manifest_dir / "gold_pipeline_readiness_inventory.csv", readiness_rows, READINESS_HEADER)
    write_csv(manifest_dir / "gold_frame_alignment_report.csv", alignment_rows, ALIGNMENT_HEADER)

    status_counts = Counter(row["status"] for row in readiness_rows)
    aligned_count = sum(1 for row in alignment_rows if row["alignment_status"] == "aligned")
    temporal_ready = sum(1 for row in readiness_rows if row["temporal_frames_exists"] == "true")
    processed_original = sum(1 for row in readiness_rows if row["status"] in {"ready_for_cvat", "needs_temporal_sweep"})
    upload_copies = 0
    upload_inventory = manifest_dir / "gold_cvat_upload_inventory.csv"
    if upload_inventory.exists():
        with upload_inventory.open(newline="", encoding="utf-8") as csv_file:
            upload_copies = sum(1 for item in csv.DictReader(csv_file) if item.get("upload_copy_exists") == "true")

    ready = len(source_rows) > 0 and len(blocked) == 0 and aligned_count == len(source_rows)
    lines = [
        "# CVAT Readiness Report",
        "",
        f"- Source manifest: `{source_manifest}`",
        f"- Selected source-manifest row count: {len(source_rows)}",
        f"- Processed original-pipeline clip count: {processed_original}",
        f"- Temporal-frame-ready clip count: {temporal_ready}",
        f"- Frame-aligned clip count: {aligned_count}",
        f"- CVAT-upload-copy count: {upload_copies}",
        f"- Blocked clip count: {len(blocked)}",
        f"- Corpus status: {'READY FOR CVAT' if ready else 'NOT READY FOR CVAT'}",
        f"- CVAT upload folder: `{args.gold_root / 'cvat_uploads'}`",
        "- CVAT Create Multi Tasks name template: `gold_v1__{{file_name}}`",
        f"- Future export folder: `{export_dir}`",
        "- Future import dry-run command: `python -m temporal_module.scripts.import_cvat_gold_annotations --dry-run`",
        "- Future import command: `python -m temporal_module.scripts.import_cvat_gold_annotations`",
        "",
        "## Status Counts",
        "",
    ]
    for key in ["ready_for_cvat", "needs_original_pipeline", "needs_temporal_sweep", "blocked"]:
        lines.append(f"- {key}: {status_counts.get(key, 0)}")
    lines.extend(["", "## Blocked Clips", ""])
    if blocked:
        for clip_id, reason in sorted(blocked.items()):
            lines.append(f"- `{clip_id}`: {reason}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## One-To-One Mapping Confirmation",
            "",
            "All ready clips are required to map as:",
            "",
            "`clip_id` <-> `outputs/<clip_id>/preprocessed/clip.mp4` <-> `temporal_frames.csv` <-> `<clip_id>.mp4` <-> `gold_v1__<clip_id>.mp4` <-> future `cvat_export__<clip_id>.zip`",
        ]
    )
    (manifest_dir / "CVAT_READINESS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Source manifest: {source_manifest}")
    print(f"Selected rows: {len(source_rows)}")
    print(f"Original processed: {processed_original}")
    print(f"Temporal ready: {temporal_ready}")
    print(f"Aligned: {aligned_count}")
    print("READY FOR CVAT" if ready else "NOT READY FOR CVAT")
    return 0 if not source_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
