"""Run the existing single-clip analytics workflow for one manifest row."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from src.pipeline.config import DEFAULT_CONFIG_PATH, get_config_value, load_config
from src.pipeline.io_utils import ensure_clip_output_dirs
from src.pipeline.logging_utils import configure_logger
from src.pipeline.manifest import ManifestRow, find_manifest_row


LOCAL_FFMPEG_ROOT = Path("tools/ffmpeg")


@dataclass
class StageResult:
    """Result for one pipeline stage."""

    stage: str
    success: bool
    command: str
    log_path: Path | None = None
    error: str = ""


@dataclass
class ClipRunResult:
    """Result for a clip pipeline run."""

    clip_id: str
    success: bool
    output_root: Path
    stages: list[StageResult]


def find_ffmpeg() -> str | None:
    """Return ffmpeg from PATH or the project-local tools directory."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        return ffmpeg

    local_matches = sorted(LOCAL_FFMPEG_ROOT.rglob("ffmpeg.exe"))
    return str(local_matches[0]) if local_matches else None


def run_command(command: list[str], log_path: Path) -> StageResult:
    """Run a command and capture output to a stage log."""
    stage = log_path.stem
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("Running: " + " ".join(command) + "\n\n")
        completed = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if completed.returncode != 0:
        return StageResult(
            stage=stage,
            success=False,
            command=" ".join(command),
            log_path=log_path,
            error=f"exit code {completed.returncode}",
        )
    return StageResult(stage=stage, success=True, command=" ".join(command), log_path=log_path)


def get_video_size(video_path: Path) -> tuple[int, int]:
    """Return video width and height."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    return width, height


def cut_clip(
    source_video: Path,
    clip_path: Path,
    duration_seconds: int,
    target_height: int,
    overwrite: bool,
) -> StageResult:
    """Create the standardized short MP4 clip used by downstream stages."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return StageResult(
            stage="preprocess",
            success=False,
            command="ffmpeg",
            error="ffmpeg was not found on PATH or under tools/ffmpeg",
        )
    if clip_path.exists() and not overwrite:
        return StageResult(stage="preprocess", success=True, command="skip existing clip")

    command = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-i",
        str(source_video),
        "-t",
        str(duration_seconds),
        "-vf",
        f"scale=-2:{target_height}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        str(clip_path),
    ]
    return run_command(command, clip_path.parent.parent / "logs" / "preprocess.log")


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


def build_output_paths(dirs: dict[str, Path]) -> dict[str, Path]:
    """Return named output paths for one clip."""
    return {
        "clip": dirs["preprocessed"] / "clip.mp4",
        "tracked_video": dirs["visualizations"] / "tracked.mp4",
        "processed_video": dirs["visualizations"] / "processed.mp4",
        "detections_csv": dirs["detections"] / "detections.csv",
        "tracks_csv": dirs["tracks"] / "tracks.csv",
        "tracks_top_csv": dirs["tracks"] / "tracks_top5.csv",
        "heatmap": dirs["visualizations"] / "heatmap_all.png",
        "trajectories": dirs["visualizations"] / "trajectories_all.png",
        "trajectories_top": dirs["tactical"] / "trajectories_top5.png",
        "player_stats": dirs["tactical"] / "player_stats.csv",
        "player_stats_xlsx": dirs["tactical"] / "player_stats.xlsx",
        "player_teams": dirs["teams"] / "player_teams.csv",
        "team_video": dirs["visualizations"] / "team_tracked.mp4",
        "team_a_heatmap": dirs["teams"] / "heatmap_team_a.png",
        "team_b_heatmap": dirs["teams"] / "heatmap_team_b.png",
        "team_debug": dirs["teams"] / "debug",
        "ball_raw_csv": dirs["detections"] / "ball_detections_raw.csv",
        "ball_filtered_csv": dirs["detections"] / "ball_detections_filtered.csv",
        "ball_video": dirs["visualizations"] / "ball_detected_filtered.mp4",
        "ball_debug": dirs["detections"] / "ball_debug",
        "ball_summary_csv": dirs["detections"] / "ball_detection_summary.csv",
        "ball_summary_md": dirs["detections"] / "ball_detection_summary.md",
        "ball_tracks_csv": dirs["tracks"] / "ball_tracks.csv",
        "ball_tracked_video": dirs["visualizations"] / "ball_tracked.mp4",
        "ball_tracking_summary_csv": dirs["tracks"] / "ball_tracking_summary.csv",
        "ball_tracking_summary_md": dirs["tracks"] / "ball_tracking_summary.md",
        "possession_csv": dirs["possession"] / "possession.csv",
        "possession_debug_csv": dirs["possession"] / "possession_debug.csv",
        "possession_summary_csv": dirs["possession"] / "possession_summary.csv",
        "possession_summary_md": dirs["possession"] / "possession_summary.md",
        "possession_video": dirs["visualizations"] / "possession.mp4",
        "possession_debug_video": dirs["visualizations"] / "possession_debug.mp4",
        "possession_qa_summary": dirs["possession"] / "possession_qa_summary.md",
        "carries_csv": dirs["carries"] / "carries.csv",
        "carries_summary_csv": dirs["carries"] / "carry_summary.csv",
        "carries_summary_md": dirs["carries"] / "carry_summary.md",
        "carries_maps": dirs["carries"] / "maps",
        "interceptions_csv": dirs["interceptions"] / "interceptions.csv",
        "interceptions_summary_csv": dirs["interceptions"] / "interceptions_summary.csv",
        "interceptions_summary_md": dirs["interceptions"] / "interceptions_summary.md",
        "interceptions_maps": dirs["interceptions"] / "maps",
        "passing_summary_csv": dirs["tactical"] / "passing_summary.csv",
        "passing_summary_md": dirs["tactical"] / "passing_summary.md",
        "passing_maps": dirs["tactical"] / "passing_maps",
    }


def process_clip(row: ManifestRow, config: dict[str, Any]) -> ClipRunResult:
    """Run the current analytics workflow for one manifest row."""
    outputs_root = Path(get_config_value(config, "runtime.outputs_dir", "outputs"))
    dirs = ensure_clip_output_dirs(outputs_root, row.clip_id)
    paths = build_output_paths(dirs)
    logger = configure_logger(f"clip.{row.clip_id}", dirs["logs"] / "clip.log")
    stages: list[StageResult] = []
    python = sys.executable

    source_video = row.video_path
    if not source_video.exists():
        stage = StageResult(
            stage="input",
            success=False,
            command="resolve input",
            error=f"video not found: {source_video}",
        )
        stages.append(stage)
        logger.error(stage.error)
        return ClipRunResult(row.clip_id, False, dirs["root"], stages)

    logger.info("Processing %s from %s", row.clip_id, source_video)
    preprocess = cut_clip(
        source_video=source_video,
        clip_path=paths["clip"],
        duration_seconds=int(get_config_value(config, "video.clip_duration_seconds", 30)),
        target_height=int(get_config_value(config, "video.target_height", 720)),
        overwrite=bool(get_config_value(config, "runtime.overwrite_outputs", True)),
    )
    stages.append(preprocess)
    if not preprocess.success:
        logger.error("Preprocess failed: %s", preprocess.error)
        return ClipRunResult(row.clip_id, False, dirs["root"], stages)

    main_command = [
        python,
        "main.py",
        "--video",
        str(paths["clip"]),
        "--output",
        str(paths["tracked_video"]),
        "--model",
        str(get_config_value(config, "models.player_model", "yolov8n.pt")),
        "--conf",
        str(get_config_value(config, "detection.conf", 0.15)),
        "--imgsz",
        str(get_config_value(config, "detection.imgsz", 640)),
        "--csv-output",
        str(paths["detections_csv"]),
        "--enable-tracking",
        "--tracker-type",
        str(get_config_value(config, "detection.tracker_type", "bytetrack")),
        "--tracks-csv",
        str(paths["tracks_csv"]),
        "--generate-heatmap",
        "--heatmap-output",
        str(paths["heatmap"]),
        "--generate-trajectories",
        "--trajectory-output",
        str(paths["trajectories"]),
        "--generate-player-stats",
        "--player-stats-output",
        str(paths["player_stats"]),
    ]
    if bool(get_config_value(config, "ball_detection.enabled", False)):
        main_command.extend(
            [
                "--detect-ball",
                "--ball-model",
                str(get_config_value(config, "models.ball_model", "yolov8n.pt")),
                "--ball-conf",
                str(get_config_value(config, "ball_detection.conf", 0.10)),
                "--ball-imgsz",
                str(get_config_value(config, "ball_detection.imgsz", 1280)),
                "--ball-raw-output-csv",
                str(paths["ball_raw_csv"]),
                "--ball-output-csv",
                str(paths["ball_filtered_csv"]),
                "--ball-video-output",
                str(paths["ball_video"]),
                "--ball-summary-csv",
                str(paths["ball_summary_csv"]),
                "--ball-summary-md",
                str(paths["ball_summary_md"]),
                "--ball-debug-dir",
                str(paths["ball_debug"]),
                "--ball-debug-frame-stride",
                str(get_config_value(config, "ball_detection.debug_frame_stride", 0)),
                "--ball-min-area",
                str(get_config_value(config, "ball_detection.min_area", 20)),
                "--ball-max-area",
                str(get_config_value(config, "ball_detection.max_area", 500)),
                "--ball-min-width",
                str(get_config_value(config, "ball_detection.min_width", 4)),
                "--ball-max-width",
                str(get_config_value(config, "ball_detection.max_width", 30)),
                "--ball-min-height",
                str(get_config_value(config, "ball_detection.min_height", 4)),
                "--ball-max-height",
                str(get_config_value(config, "ball_detection.max_height", 30)),
                "--ball-max-detections-per-frame",
                str(get_config_value(config, "ball_detection.max_detections_per_frame", 1)),
                "--ball-exclude-top-ratio",
                str(get_config_value(config, "ball_detection.exclude_top_ratio", 0.08)),
                "--ball-exclude-bottom-ratio",
                str(get_config_value(config, "ball_detection.exclude_bottom_ratio", 0.0)),
            ]
        )

    stages.append(run_command(main_command, dirs["logs"] / "detection_tracking.log"))
    if not stages[-1].success:
        logger.error("Detection/tracking failed: %s", stages[-1].error)
        return ClipRunResult(row.clip_id, False, dirs["root"], stages)
    shutil.copy2(paths["tracked_video"], paths["processed_video"])

    remaining_ball_commands: list[tuple[str, list[str]]] = []
    if bool(get_config_value(config, "ball_detection.enabled", False)):
        ball_commands = _ball_stage_commands(python, paths, config)
        ball_tracking_name, ball_tracking_command = ball_commands[0]
        result = run_command(
            ball_tracking_command,
            dirs["logs"] / f"{ball_tracking_name}.log",
        )
        result.stage = ball_tracking_name
        stages.append(result)
        if not result.success:
            logger.error("%s failed: %s", ball_tracking_name, result.error)
            return ClipRunResult(row.clip_id, False, dirs["root"], stages)
        remaining_ball_commands = ball_commands[1:]

    try:
        frame_width, frame_height = get_video_size(paths["clip"])
        write_top_tracks_csv(
            paths["tracks_csv"],
            paths["tracks_top_csv"],
            top_n=int(get_config_value(config, "tactical.top_tracks", 5)),
        )
        stages.append(StageResult("top_tracks", True, "write_top_tracks_csv"))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        stages.append(StageResult("top_tracks", False, "write_top_tracks_csv", error=str(error)))
        logger.error("Top tracks failed: %s", error)
        return ClipRunResult(row.clip_id, False, dirs["root"], stages)

    stage_commands = [
        (
            "trajectories_top5",
            [
                python,
                "-m",
                "src.analytics.trajectories",
                "--tracks-csv",
                str(paths["tracks_top_csv"]),
                "--output",
                str(paths["trajectories_top"]),
                "--frame-width",
                str(frame_width),
                "--frame-height",
                str(frame_height),
            ],
        ),
        (
            "player_stats_excel",
            [
                python,
                "-m",
                "src.analytics.player_stats",
                "--tracks-csv",
                str(paths["tracks_csv"]),
                "--output",
                str(paths["player_stats"]),
                "--excel-output",
                str(paths["player_stats_xlsx"]),
            ],
        ),
        (
            "team_assignment",
            [
                python,
                "-m",
                "src.analytics.team_classifier",
                "--video",
                str(paths["clip"]),
                "--tracks-csv",
                str(paths["tracks_csv"]),
                "--output",
                str(paths["player_teams"]),
                "--team-video-output",
                str(paths["team_video"]),
                "--team-a-heatmap",
                str(paths["team_a_heatmap"]),
                "--team-b-heatmap",
                str(paths["team_b_heatmap"]),
                "--num-clusters",
                "2",
                "--detect-roles",
                "--role-clusters",
                "5",
                "--debug-dir",
                str(paths["team_debug"]),
                "--frame-width",
                str(frame_width),
                "--frame-height",
                str(frame_height),
            ],
        ),
    ]

    stage_commands.extend(remaining_ball_commands)

    for stage_name, command in stage_commands:
        result = run_command(command, dirs["logs"] / f"{stage_name}.log")
        result.stage = stage_name
        stages.append(result)
        if not result.success:
            logger.error("%s failed: %s", stage_name, result.error)
            return ClipRunResult(row.clip_id, False, dirs["root"], stages)

    if bool(get_config_value(config, "ball_detection.enabled", False)):
        if bool(get_config_value(config, "carries.enabled", True)):
            for stage_name, command in _carry_stage_commands(python, paths):
                result = run_command(command, dirs["logs"] / f"{stage_name}.log")
                result.stage = stage_name
                stages.append(result)
                if not result.success:
                    return ClipRunResult(row.clip_id, False, dirs["root"], stages)

        if bool(get_config_value(config, "interceptions.enabled", True)):
            for stage_name, command in _interception_stage_commands(python, paths):
                result = run_command(command, dirs["logs"] / f"{stage_name}.log")
                result.stage = stage_name
                stages.append(result)
                if not result.success:
                    return ClipRunResult(row.clip_id, False, dirs["root"], stages)

        if bool(get_config_value(config, "passes.enabled", True)):
            for stage_name, command in _passing_stage_commands(python, paths):
                result = run_command(command, dirs["logs"] / f"{stage_name}.log")
                result.stage = stage_name
                stages.append(result)
                if not result.success:
                    return ClipRunResult(row.clip_id, False, dirs["root"], stages)

    logger.info("Completed %s", row.clip_id)
    return ClipRunResult(row.clip_id, all(stage.success for stage in stages), dirs["root"], stages)


def _ball_stage_commands(
    python: str,
    paths: dict[str, Path],
    config: dict[str, Any],
) -> list[tuple[str, list[str]]]:
    commands = [
        (
            "ball_tracking",
            [
                python,
                "-m",
                "src.tracking.ball_tracker",
                "--video",
                str(paths["clip"]),
                "--detections-csv",
                str(paths["ball_filtered_csv"]),
                "--output-csv",
                str(paths["ball_tracks_csv"]),
                "--output-video",
                str(paths["ball_tracked_video"]),
                "--summary-csv",
                str(paths["ball_tracking_summary_csv"]),
                "--summary-md",
                str(paths["ball_tracking_summary_md"]),
                "--max-distance",
                str(get_config_value(config, "ball_tracking.max_distance", 90.0)),
                "--max-gap",
                str(get_config_value(config, "ball_tracking.max_gap", 8)),
            ],
        ),
        (
            "possession",
            [
                python,
                "-m",
                "src.analytics.possession",
                "--video",
                str(paths["clip"]),
                "--player-tracks-csv",
                str(paths["tracks_csv"]),
                "--teams-csv",
                str(paths["player_teams"]),
                "--ball-tracks-csv",
                str(paths["ball_tracks_csv"]),
                "--output-csv",
                str(paths["possession_csv"]),
                "--debug-csv",
                str(paths["possession_debug_csv"]),
                "--summary-csv",
                str(paths["possession_summary_csv"]),
                "--summary-md",
                str(paths["possession_summary_md"]),
                "--qa-summary-md",
                str(paths["possession_qa_summary"]),
                "--output-video",
                str(paths["possession_video"]),
                "--debug-video",
                str(paths["possession_debug_video"]),
                "--max-player-ball-distance",
                str(get_config_value(config, "possession.max_player_ball_distance", 80.0)),
                "--min-track-confidence",
                str(get_config_value(config, "possession.min_track_confidence", 0.10)),
                "--min-ball-confidence",
                str(get_config_value(config, "possession.min_ball_confidence", 0.25)),
                "--switch-confirmation-frames",
                str(get_config_value(config, "possession.switch_confirmation_frames", 3)),
            ],
        ),
    ]
    if bool(get_config_value(config, "possession.assign_interpolated", False)):
        commands[-1][1].append("--assign-interpolated")
    return commands


def _carry_stage_commands(
    python: str,
    paths: dict[str, Path],
) -> list[tuple[str, list[str]]]:
    return [
        (
            "carries",
            [
                python,
                "-m",
                "src.analytics.carries",
                "--possession-csv",
                str(paths["possession_csv"]),
                "--possession-debug-csv",
                str(paths["possession_debug_csv"]),
                "--player-tracks-csv",
                str(paths["tracks_csv"]),
                "--output-csv",
                str(paths["carries_csv"]),
                "--summary-csv",
                str(paths["carries_summary_csv"]),
                "--summary-md",
                str(paths["carries_summary_md"]),
            ],
        ),
        (
            "carry_maps",
            [
                python,
                "-m",
                "src.analytics.carry_map",
                "--carry-events-csv",
                str(paths["carries_csv"]),
                "--output-dir",
                str(paths["carries_maps"]),
            ],
        ),
    ]


def _interception_stage_commands(
    python: str,
    paths: dict[str, Path],
) -> list[tuple[str, list[str]]]:
    return [
        (
            "interceptions",
            [
                python,
                "-m",
                "src.analytics.interceptions",
                "--possession-csv",
                str(paths["possession_csv"]),
                "--possession-debug-csv",
                str(paths["possession_debug_csv"]),
                "--output-csv",
                str(paths["interceptions_csv"]),
                "--summary-csv",
                str(paths["interceptions_summary_csv"]),
                "--summary-md",
                str(paths["interceptions_summary_md"]),
            ],
        ),
        (
            "interception_maps",
            [
                python,
                "-m",
                "src.analytics.interception_map",
                "--interceptions-csv",
                str(paths["interceptions_csv"]),
                "--output-dir",
                str(paths["interceptions_maps"]),
            ],
        ),
    ]


def _passing_stage_commands(
    python: str,
    paths: dict[str, Path],
) -> list[tuple[str, list[str]]]:
    return [
        (
            "passing_stats",
            [
                python,
                "-m",
                "src.analytics.passing_stats",
                "--possession-csv",
                str(paths["possession_csv"]),
                "--output-csv",
                str(paths["passing_summary_csv"]),
                "--output-md",
                str(paths["passing_summary_md"]),
            ],
        ),
        (
            "passing_maps",
            [
                python,
                "-m",
                "src.analytics.generate_passing_network",
                "--possession-csv",
                str(paths["possession_csv"]),
                "--possession-debug-csv",
                str(paths["possession_debug_csv"]),
                "--output-dir",
                str(paths["passing_maps"]),
            ],
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one manifest clip through the analytics pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--clip-id")
    parser.add_argument("--video", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    manifest_path = args.manifest or Path(get_config_value(config, "data.manifest_path"))
    if args.video:
        row = ManifestRow(
            clip_id=args.clip_id or args.video.stem,
            split="manual",
            competition="unknown",
            source="manual",
            game_id=args.video.stem,
            half=0,
            video_path=args.video,
            enabled=True,
        )
    elif args.clip_id:
        row = find_manifest_row(manifest_path, args.clip_id)
    else:
        raise SystemExit("Provide --clip-id or --video")

    result = process_clip(row, config)
    print(f"Clip {result.clip_id}: {'succeeded' if result.success else 'failed'}")
    print(f"Output root: {result.output_root}")
    for stage in result.stages:
        status = "ok" if stage.success else "failed"
        print(f"- {stage.stage}: {status}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
