"""Batch-cut SoccerNet 720p halves and run the analytics pipeline safely.

Default behavior:
- Recursively discovers 1_720p.mkv and 2_720p.mkv under:
  data/SoccerNet/england_epl/
- Cuts a 30-second segment from each source video.
- Runs the existing analytics stages for every discovered source.
- Writes isolated outputs per source video.
- Never writes batch results under the existing outputs/ directory.

Example:
    python run_720p_clip_pipeline.py --dry-run --max-videos 2
    python run_720p_clip_pipeline.py --max-videos 1 --detect-ball
    python run_720p_clip_pipeline.py --detect-ball --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SOCCERNET_DIR = Path("data/SoccerNet/england_epl")
DEFAULT_BATCH_CLIPS_ROOT = Path("temporal_module/data/soccernet_batch_clips")
DEFAULT_BATCH_OUTPUT_ROOT = Path("temporal_module/data/soccernet_batch_analytics")
LOCAL_FFMPEG_ROOT = Path("tools/ffmpeg")

SUPPORTED_SOURCE_NAMES = {"1_720p.mkv", "2_720p.mkv"}


def utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def find_ffmpeg() -> str:
    """Find FFmpeg from PATH or the project-local tools directory."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    local_matches = sorted(LOCAL_FFMPEG_ROOT.rglob("ffmpeg.exe"))
    if local_matches:
        return str(local_matches[0])

    raise RuntimeError(
        "ffmpeg was not found on PATH or under tools/ffmpeg. "
        "Install FFmpeg or place ffmpeg.exe under tools/ffmpeg."
    )


def run_command(command: list[str]) -> None:
    """Print and run one command, raising on failure."""
    print("\nRunning:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    subprocess.run(command, check=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with stable readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def safe_component(value: str) -> str:
    """Convert arbitrary text into a safe deterministic path component."""
    value = value.replace("\\", "/")
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "unknown"

def source_half(source_video: Path) -> str:
    """Return SoccerNet half number from the source filename."""
    if source_video.name.startswith("1_"):
        return "1"
    if source_video.name.startswith("2_"):
        return "2"
    return "unknown"

def make_clip_id(source_video: Path, soccernet_root: Path) -> str:
    """Create a deterministic clip ID from relative SoccerNet path and half."""
    relative = source_video.relative_to(soccernet_root).as_posix()
    relative_without_suffix = str(Path(relative).with_suffix(""))
    relative_without_video_name = relative_without_suffix.replace("1_720p", "h1_720p")
    relative_without_video_name = relative_without_video_name.replace("2_720p", "h2_720p")
    return safe_component(relative_without_video_name)


def discover_source_videos(
    soccernet_root: Path,
    seasons: set[str] | None,
    halves: set[str],
) -> list[Path]:
    """Recursively discover requested SoccerNet half-videos."""
    if not soccernet_root.exists():
        raise FileNotFoundError(f"SoccerNet root not found: {soccernet_root}")

    videos: list[Path] = []

    for path in sorted(soccernet_root.rglob("*_720p.mkv")):
        if not path.is_file() or path.name not in SUPPORTED_SOURCE_NAMES:
            continue

        try:
            relative = path.relative_to(soccernet_root)
        except ValueError:
            continue

        if not relative.parts:
            continue

        season = relative.parts[0]
        half = source_half(path)

        if seasons and season not in seasons:
            continue
        if half not in halves:
            continue

        videos.append(path)

    return videos


def write_top_tracks_csv(tracks_csv_path: Path, output_path: Path, top_n: int = 5) -> None:
    """Write rows for the longest-lived tracks to a separate CSV."""
    with tracks_csv_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or "track_id" not in reader.fieldnames:
            raise RuntimeError(f"Tracks CSV missing track_id column: {tracks_csv_path}")
        rows = list(reader)

    track_counts = Counter(row["track_id"] for row in rows if row.get("track_id"))
    top_track_ids = {track_id for track_id, _ in track_counts.most_common(top_n)}
    top_rows = [row for row in rows if row.get("track_id") in top_track_ids]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(top_rows)

    print(f"Top {top_n} tracks CSV saved to: {output_path}")


def cut_clip(
    source_video: Path,
    clip_path: Path,
    start_seconds: float,
    duration_seconds: float,
    overwrite: bool,
) -> None:
    """Cut one MP4 segment from a SoccerNet MKV source video."""
    ffmpeg = find_ffmpeg()
    clip_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-ss",
        str(start_seconds),
        "-i",
        str(source_video),
        "-t",
        str(duration_seconds),
        "-vf",
        "scale=-2:720",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        str(clip_path),
    ]
    run_command(command)


def build_output_paths(run_output_dir: Path) -> dict[str, Path]:
    """Build isolated output locations for one source clip."""
    return {
        "tracked_video": run_output_dir / "tracked.mp4",
        "detections_csv": run_output_dir / "detections.csv",
        "tracks_csv": run_output_dir / "tracks.csv",
        "tracks_top5_csv": run_output_dir / "tracks_top5.csv",
        "heatmap": run_output_dir / "heatmap_all.png",
        "trajectories": run_output_dir / "trajectories_all.png",
        "trajectories_top5": run_output_dir / "trajectories_top5.png",
        "player_stats": run_output_dir / "player_stats.csv",
        "player_stats_xlsx": run_output_dir / "player_stats.xlsx",
        "player_teams": run_output_dir / "player_teams.csv",
        "team_video": run_output_dir / "team_tracked.mp4",
        "team_a_heatmap": run_output_dir / "heatmap_team_a.png",
        "team_b_heatmap": run_output_dir / "heatmap_team_b.png",
        "team_debug": run_output_dir / "team_debug",
        "ball_raw_detections_csv": run_output_dir / "ball_detections_raw.csv",
        "ball_filtered_detections_csv": run_output_dir / "ball_detections_filtered.csv",
        "ball_filtered_video": run_output_dir / "ball_detected_filtered.mp4",
        "ball_debug": run_output_dir / "ball_debug",
        "ball_summary_csv": run_output_dir / "ball_debug" / "ball_detection_summary.csv",
        "ball_summary_md": run_output_dir / "ball_debug" / "ball_detection_summary.md",
        "ball_tracks_csv": run_output_dir / "ball_tracks.csv",
        "ball_tracked_video": run_output_dir / "ball_tracked.mp4",
        "ball_tracking_summary_csv": run_output_dir / "ball_debug" / "ball_tracking_summary.csv",
        "ball_tracking_summary_md": run_output_dir / "ball_debug" / "ball_tracking_summary.md",
        "possession_csv": run_output_dir / "possession.csv",
        "possession_debug_csv": run_output_dir / "possession_debug.csv",
        "possession_summary_csv": run_output_dir / "possession_summary.csv",
        "possession_summary_md": run_output_dir / "possession_summary.md",
        "possession_video": run_output_dir / "possession.mp4",
        "possession_debug_video": run_output_dir / "possession_debug.mp4",
        "possession_qa_summary": run_output_dir / "possession_qa_summary.md",
    }


def run_analytics(
    clip_path: Path,
    run_output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Run tracking, team analysis, optional ball analysis, and possession."""
    python = sys.executable
    outputs = build_output_paths(run_output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    main_command = [
        python,
        "main.py",
        "--video",
        str(clip_path),
        "--output",
        str(outputs["tracked_video"]),
        "--model",
        args.model,
        "--conf",
        str(args.conf),
        "--imgsz",
        str(args.imgsz),
        "--csv-output",
        str(outputs["detections_csv"]),
        "--enable-tracking",
        "--tracker-type",
        "bytetrack",
        "--tracks-csv",
        str(outputs["tracks_csv"]),
        "--generate-heatmap",
        "--heatmap-output",
        str(outputs["heatmap"]),
        "--generate-trajectories",
        "--trajectory-output",
        str(outputs["trajectories"]),
        "--generate-player-stats",
        "--player-stats-output",
        str(outputs["player_stats"]),
    ]

    if args.detect_ball:
        main_command.extend(
            [
                "--detect-ball",
                "--ball-model",
                args.ball_model,
                "--ball-conf",
                str(args.ball_conf),
                "--ball-imgsz",
                str(args.ball_imgsz),
                "--ball-raw-output-csv",
                str(outputs["ball_raw_detections_csv"]),
                "--ball-output-csv",
                str(outputs["ball_filtered_detections_csv"]),
                "--ball-video-output",
                str(outputs["ball_filtered_video"]),
                "--ball-summary-csv",
                str(outputs["ball_summary_csv"]),
                "--ball-summary-md",
                str(outputs["ball_summary_md"]),
                "--ball-debug-dir",
                str(outputs["ball_debug"]),
                "--ball-debug-frame-stride",
                str(args.ball_debug_frame_stride),
                "--ball-min-area",
                str(args.ball_min_area),
                "--ball-max-area",
                str(args.ball_max_area),
                "--ball-min-width",
                str(args.ball_min_width),
                "--ball-max-width",
                str(args.ball_max_width),
                "--ball-min-height",
                str(args.ball_min_height),
                "--ball-max-height",
                str(args.ball_max_height),
                "--ball-max-detections-per-frame",
                str(args.ball_max_detections_per_frame),
                "--ball-exclude-top-ratio",
                str(args.ball_exclude_top_ratio),
                "--ball-exclude-bottom-ratio",
                str(args.ball_exclude_bottom_ratio),
            ]
        )

    run_command(main_command)

    if args.detect_ball and args.ball_model != "yolov8n.pt":
        run_command(
            [
                python,
                "-m",
                "src.detection.compare_ball_models",
                "--video",
                str(clip_path),
                "--candidate-model",
                args.ball_model,
                "--generic-model",
                "yolov8n.pt",
                "--generic-conf",
                "0.10",
                "--candidate-conf",
                str(args.ball_conf),
                "--imgsz",
                str(args.ball_imgsz),
                "--ball-min-area",
                str(args.ball_min_area),
                "--ball-max-area",
                str(args.ball_max_area),
                "--ball-min-width",
                str(args.ball_min_width),
                "--ball-max-width",
                str(args.ball_max_width),
                "--ball-min-height",
                str(args.ball_min_height),
                "--ball-max-height",
                str(args.ball_max_height),
                "--ball-max-detections-per-frame",
                str(args.ball_max_detections_per_frame),
                "--ball-exclude-top-ratio",
                str(args.ball_exclude_top_ratio),
                "--ball-exclude-bottom-ratio",
                str(args.ball_exclude_bottom_ratio),
            ]
        )

    if args.detect_ball:
        run_command(
            [
                python,
                "-m",
                "src.tracking.ball_tracker",
                "--video",
                str(clip_path),
                "--detections-csv",
                str(outputs["ball_filtered_detections_csv"]),
                "--output-csv",
                str(outputs["ball_tracks_csv"]),
                "--output-video",
                str(outputs["ball_tracked_video"]),
                "--summary-csv",
                str(outputs["ball_tracking_summary_csv"]),
                "--summary-md",
                str(outputs["ball_tracking_summary_md"]),
                "--max-distance",
                str(args.ball_track_max_distance),
                "--max-gap",
                str(args.ball_track_max_gap),
            ]
        )

    write_top_tracks_csv(outputs["tracks_csv"], outputs["tracks_top5_csv"])

    run_command(
        [
            python,
            "-m",
            "src.analytics.trajectories",
            "--tracks-csv",
            str(outputs["tracks_top5_csv"]),
            "--output",
            str(outputs["trajectories_top5"]),
            "--frame-width",
            "1280",
            "--frame-height",
            "720",
        ]
    )

    run_command(
        [
            python,
            "-m",
            "src.analytics.player_stats",
            "--tracks-csv",
            str(outputs["tracks_csv"]),
            "--output",
            str(outputs["player_stats"]),
            "--excel-output",
            str(outputs["player_stats_xlsx"]),
        ]
    )

    run_command(
        [
            python,
            "-m",
            "src.analytics.team_classifier",
            "--video",
            str(clip_path),
            "--tracks-csv",
            str(outputs["tracks_csv"]),
            "--output",
            str(outputs["player_teams"]),
            "--team-video-output",
            str(outputs["team_video"]),
            "--team-a-heatmap",
            str(outputs["team_a_heatmap"]),
            "--team-b-heatmap",
            str(outputs["team_b_heatmap"]),
            "--num-clusters",
            "2",
            "--detect-roles",
            "--role-clusters",
            "5",
            "--debug-dir",
            str(outputs["team_debug"]),
            "--frame-width",
            "1280",
            "--frame-height",
            "720",
        ]
    )

    if args.detect_ball:
        possession_command = [
            python,
            "-m",
            "src.analytics.possession",
            "--video",
            str(clip_path),
            "--player-tracks-csv",
            str(outputs["tracks_csv"]),
            "--teams-csv",
            str(outputs["player_teams"]),
            "--ball-tracks-csv",
            str(outputs["ball_tracks_csv"]),
            "--output-csv",
            str(outputs["possession_csv"]),
            "--debug-csv",
            str(outputs["possession_debug_csv"]),
            "--summary-csv",
            str(outputs["possession_summary_csv"]),
            "--summary-md",
            str(outputs["possession_summary_md"]),
            "--qa-summary-md",
            str(outputs["possession_qa_summary"]),
            "--output-video",
            str(outputs["possession_video"]),
            "--debug-video",
            str(outputs["possession_debug_video"]),
            "--max-player-ball-distance",
            str(args.possession_max_distance),
            "--min-track-confidence",
            str(args.possession_min_track_confidence),
            "--min-ball-confidence",
            str(args.possession_min_ball_confidence),
            "--switch-confirmation-frames",
            str(args.possession_switch_confirmation_frames),
        ]

        if args.possession_assign_interpolated:
            possession_command.append("--assign-interpolated")

        run_command(possession_command)


def parse_csv_set(value: str) -> set[str]:
    """Parse comma-separated CLI values into a normalized set."""
    return {part.strip() for part in value.split(",") if part.strip()}


def parse_args() -> argparse.Namespace:
    """Parse batch pipeline arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Recursively process SoccerNet EPL 720p half-videos into isolated "
            "30-second clip and analytics folders."
        )
    )

    parser.add_argument(
        "--soccernet-dir",
        type=Path,
        default=DEFAULT_SOCCERNET_DIR,
        help="Root containing season folders and SoccerNet game directories.",
    )
    parser.add_argument(
        "--seasons",
        default="",
        help="Optional comma-separated season filter, e.g. 2014-2015,2015-2016.",
    )
    parser.add_argument(
        "--halves",
        default="1,2",
        help="Comma-separated SoccerNet halves to process: 1, 2, or 1,2.",
    )
    parser.add_argument(
        "--batch-clips-root",
        type=Path,
        default=DEFAULT_BATCH_CLIPS_ROOT,
        help="Root directory for exported MP4 clips.",
    )
    parser.add_argument(
        "--batch-output-root",
        type=Path,
        default=DEFAULT_BATCH_OUTPUT_ROOT,
        help="Root directory for isolated analytics results.",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="Start time within each SoccerNet half.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Duration in seconds for each exported clip.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Maximum number of discovered videos to process; 0 means all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned inputs and outputs only. No files are created.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip sources with a successful existing run_status.json.",
    )
    parser.add_argument(
        "--overwrite-batch-output",
        action="store_true",
        help="Allow replacement of an existing exported clip/output directory.",
    )

    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640)

    parser.add_argument("--detect-ball", action="store_true")
    parser.add_argument("--ball-model", default="yolov8n.pt")
    parser.add_argument("--ball-conf", type=float, default=0.10)
    parser.add_argument("--ball-imgsz", type=int, default=1280)
    parser.add_argument("--ball-min-area", type=int, default=20)
    parser.add_argument("--ball-max-area", type=int, default=500)
    parser.add_argument("--ball-min-width", type=int, default=4)
    parser.add_argument("--ball-max-width", type=int, default=30)
    parser.add_argument("--ball-min-height", type=int, default=4)
    parser.add_argument("--ball-max-height", type=int, default=30)
    parser.add_argument("--ball-max-detections-per-frame", type=int, default=1)
    parser.add_argument("--ball-exclude-top-ratio", type=float, default=0.08)
    parser.add_argument("--ball-exclude-bottom-ratio", type=float, default=0.0)
    parser.add_argument("--ball-debug-frame-stride", type=int, default=0)
    parser.add_argument("--ball-track-max-distance", type=float, default=90.0)
    parser.add_argument("--ball-track-max-gap", type=int, default=8)
    parser.add_argument("--possession-max-distance", type=float, default=80.0)
    parser.add_argument("--possession-min-track-confidence", type=float, default=0.10)
    parser.add_argument("--possession-min-ball-confidence", type=float, default=0.25)
    parser.add_argument("--possession-switch-confirmation-frames", type=int, default=3)
    parser.add_argument("--possession-assign-interpolated", action="store_true")

    return parser.parse_args()


def was_successfully_completed(status_path: Path) -> bool:
    """Return True only for an existing successful batch status file."""
    if not status_path.exists():
        return False

    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    return payload.get("status") == "success"


def process_one_source(
    source_video: Path,
    soccernet_root: Path,
    args: argparse.Namespace,
) -> str:
    """Process one SoccerNet half and return success, failed, or skipped."""
    clip_id = make_clip_id(source_video, soccernet_root)
    relative_source = source_video.relative_to(soccernet_root)

    clip_dir = args.batch_clips_root / clip_id
    run_output_dir = args.batch_output_root / clip_id
    clip_path = clip_dir / "clip_30s.mp4"
    manifest_path = run_output_dir / "run_manifest.json"
    status_path = run_output_dir / "run_status.json"

    season = relative_source.parts[0] if relative_source.parts else "unknown"
    half = source_half(source_video)

    manifest = {
        "clip_id": clip_id,
        "source_video": str(source_video),
        "relative_source_video": relative_source.as_posix(),
        "season": season,
        "half": half,
        "start_seconds": args.start_seconds,
        "duration_seconds": args.duration,
        "clip_path": str(clip_path),
        "analytics_output_dir": str(run_output_dir),
        "model": args.model,
        "detect_ball": args.detect_ball,
        "created_at": utc_now(),
    }

    if args.resume and was_successfully_completed(status_path):
        print(f"\nSkipping completed source: {source_video}")
        return "skipped"

    if not args.overwrite_batch_output and (
        clip_path.exists() or run_output_dir.exists()
    ):
        print(
            f"\nSkipping existing batch output: {clip_id}\n"
            "Use --resume for successful runs or --overwrite-batch-output "
            "to intentionally replace this batch output."
        )
        return "skipped"

    if args.dry_run:
        print("\n[DRY RUN]")
        print(f"Source: {source_video}")
        print(f"Clip ID: {clip_id}")
        print(f"Clip output: {clip_path}")
        print(f"Analytics output: {run_output_dir}")
        return "skipped"

    clip_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)

    try:
        cut_clip(
            source_video=source_video,
            clip_path=clip_path,
            start_seconds=args.start_seconds,
            duration_seconds=args.duration,
            overwrite=args.overwrite_batch_output,
        )

        run_analytics(
            clip_path=clip_path,
            run_output_dir=run_output_dir,
            args=args,
        )

        write_json(
            status_path,
            {
                "status": "success",
                "clip_id": clip_id,
                "source_video": str(source_video),
                "completed_at": utc_now(),
            },
        )
        print(f"\nSUCCESS: {clip_id}")
        return "success"

    except subprocess.CalledProcessError as error:
        write_json(
            status_path,
            {
                "status": "failed",
                "clip_id": clip_id,
                "source_video": str(source_video),
                "failed_stage": "subprocess",
                "error": f"Command failed with exit code {error.returncode}",
                "completed_at": utc_now(),
            },
        )
        print(f"\nFAILED: {clip_id} - subprocess exit code {error.returncode}")
        return "failed"

    except Exception as error:
        write_json(
            status_path,
            {
                "status": "failed",
                "clip_id": clip_id,
                "source_video": str(source_video),
                "failed_stage": "python",
                "error": str(error),
                "completed_at": utc_now(),
            },
        )
        print(f"\nFAILED: {clip_id} - {error}")
        return "failed"


def main() -> int:
    """Run the batch SoccerNet clip and analytics workflow."""
    args = parse_args()

    requested_halves = parse_csv_set(args.halves)
    invalid_halves = requested_halves - {"1", "2"}
    if invalid_halves:
        print(f"Invalid --halves values: {sorted(invalid_halves)}", file=sys.stderr)
        return 2

    seasons = parse_csv_set(args.seasons) or None
    videos = discover_source_videos(
        soccernet_root=args.soccernet_dir,
        seasons=seasons,
        halves=requested_halves,
    )

    if args.max_videos > 0:
        videos = videos[: args.max_videos]

    print(f"SoccerNet root: {args.soccernet_dir}")
    print(f"Discovered videos: {len(videos)}")

    if not videos:
        print("No matching 1_720p.mkv or 2_720p.mkv files found.")
        return 0

    if args.dry_run:
        for source_video in videos:
            process_one_source(source_video, args.soccernet_dir, args)

        print("\nDry run complete. No files were created.")
        return 0

    summary = Counter()

    for index, source_video in enumerate(videos, start=1):
        print(f"\n{'=' * 80}")
        print(f"[{index}/{len(videos)}] Processing: {source_video}")
        result = process_one_source(source_video, args.soccernet_dir, args)
        summary[result] += 1

    print("\n" + "=" * 80)
    print("Batch summary")
    print(f"Discovered: {len(videos)}")
    print(f"Succeeded:  {summary['success']}")
    print(f"Failed:     {summary['failed']}")
    print(f"Skipped:    {summary['skipped']}")
    print(f"Clip root:  {args.batch_clips_root}")
    print(f"Output root:{args.batch_output_root}")

    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())