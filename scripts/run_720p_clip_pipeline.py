"""Cut a 30-second SoccerNet 720p clip and run the full analytics pipeline."""

import argparse
import csv
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


DEFAULT_SOCCERNET_DIR = Path("data/SoccerNet")
DEFAULT_SOURCE_NAME = "1_720p.mkv"
DEFAULT_CLIP_PATH = Path("data/sample_30s_720p.mp4")
LOCAL_FFMPEG_ROOT = Path("tools/ffmpeg")


def find_source_video(soccernet_dir: Path, source_name: str) -> Path | None:
    """Return the first matching SoccerNet source video, if available."""
    direct_path = soccernet_dir / source_name
    if direct_path.is_file():
        return direct_path

    matches = sorted(path for path in soccernet_dir.rglob(source_name) if path.is_file())
    return matches[0] if matches else None


def run_command(command: list[str]) -> None:
    """Print and run a subprocess command, failing on non-zero exit."""
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def find_ffmpeg() -> str | None:
    """Return ffmpeg from PATH or the project-local tools directory."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        return ffmpeg

    local_matches = sorted(LOCAL_FFMPEG_ROOT.rglob("ffmpeg.exe"))
    if local_matches:
        return str(local_matches[0])
    return None


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


def cut_clip(source_video: Path, clip_path: Path, duration_seconds: int, overwrite: bool) -> None:
    """Create a short MP4 clip from the 720p SoccerNet MKV."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH or under tools/ffmpeg")

    clip_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y" if overwrite else "-n",
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


def run_analytics(
    clip_path: Path,
    model_path: str,
    conf: float,
    imgsz: int,
    detect_ball: bool,
    ball_model_path: str,
    ball_conf: float,
    ball_imgsz: int,
    ball_min_area: int,
    ball_max_area: int,
    ball_min_width: int,
    ball_max_width: int,
    ball_min_height: int,
    ball_max_height: int,
    ball_max_detections_per_frame: int,
    ball_exclude_top_ratio: float,
    ball_exclude_bottom_ratio: float,
    ball_debug_frame_stride: int,
    ball_track_max_distance: float,
    ball_track_max_gap: int,
    possession_max_distance: float,
    possession_min_track_confidence: float,
    possession_min_ball_confidence: float,
    possession_switch_confirmation_frames: int,
    possession_assign_interpolated: bool,
) -> None:
    """Run detection, tracking, movement analytics, and team analytics."""
    python = sys.executable
    outputs = {
        "tracked_video": Path("outputs/tracked_30s_720p.mp4"),
        "detections_csv": Path("outputs/detections_30s_720p.csv"),
        "tracks_csv": Path("outputs/tracks_30s_720p.csv"),
        "tracks_top5_csv": Path("outputs/tracks_top5_720p.csv"),
        "heatmap": Path("outputs/heatmap_all_720p.png"),
        "trajectories": Path("outputs/trajectories_all_720p.png"),
        "trajectories_top5": Path("outputs/trajectories_top5_720p.png"),
        "player_stats": Path("outputs/player_stats_30s_720p.csv"),
        "player_stats_xlsx": Path("outputs/player_stats_30s_720p.xlsx"),
        "player_teams": Path("outputs/player_teams_30s_720p.csv"),
        "team_video": Path("outputs/team_tracked_30s_720p.mp4"),
        "team_a_heatmap": Path("outputs/heatmap_team_a_720p.png"),
        "team_b_heatmap": Path("outputs/heatmap_team_b_720p.png"),
        "team_debug": Path("outputs/team_debug"),
        "ball_raw_detections_csv": Path("outputs/ball_detections_raw_30s_720p.csv"),
        "ball_filtered_detections_csv": Path("outputs/ball_detections_filtered_30s_720p.csv"),
        "ball_filtered_video": Path("outputs/ball_detected_filtered_30s_720p.mp4"),
        "ball_debug": Path("outputs/ball_debug"),
        "ball_summary_csv": Path("outputs/ball_debug/ball_detection_summary.csv"),
        "ball_summary_md": Path("outputs/ball_debug/ball_detection_summary.md"),
        "ball_tracks_csv": Path("outputs/ball_tracks_30s_720p.csv"),
        "ball_tracked_video": Path("outputs/ball_tracked_30s_720p.mp4"),
        "ball_tracking_summary_csv": Path("outputs/ball_debug/ball_tracking_summary.csv"),
        "ball_tracking_summary_md": Path("outputs/ball_debug/ball_tracking_summary.md"),
        "possession_csv": Path("outputs/possession_30s_720p.csv"),
        "possession_debug_csv": Path("outputs/possession_debug_30s_720p.csv"),
        "possession_summary_csv": Path("outputs/possession_summary_30s_720p.csv"),
        "possession_summary_md": Path("outputs/possession_summary_30s_720p.md"),
        "possession_video": Path("outputs/possession_30s_720p.mp4"),
        "possession_debug_video": Path("outputs/possession_debug_30s_720p.mp4"),
        "possession_qa_summary": Path("outputs/possession_qa_summary.md"),
        "carry_events_csv": Path("outputs/carry_events_30s_720p.csv"),
        "carry_summary_csv": Path("outputs/carry_summary_30s_720p.csv"),
        "carry_summary_md": Path("outputs/carry_summary_30s_720p.md"),
        "carry_maps_dir": Path("outputs/carry_maps"),
        "interceptions_csv": Path("outputs/interceptions_30s_720p.csv"),
        "interceptions_summary_csv": Path("outputs/interceptions_summary_30s_720p.csv"),
        "interceptions_summary_md": Path("outputs/interceptions_summary_30s_720p.md"),
        "interception_maps_dir": Path("outputs/interception_maps"),
        "passing_summary_csv": Path("outputs/passing_summary_30s_720p.csv"),
        "passing_summary_md": Path("outputs/passing_summary_30s_720p.md"),
        "passing_maps_dir": Path("outputs/passing_maps"),
    }

    main_command = [
        python,
        "main.py",
        "--video",
        str(clip_path),
        "--output",
        str(outputs["tracked_video"]),
        "--model",
        model_path,
        "--conf",
        str(conf),
        "--imgsz",
        str(imgsz),
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
    if detect_ball:
        main_command.extend(
            [
                "--detect-ball",
                "--ball-model",
                ball_model_path,
                "--ball-conf",
                str(ball_conf),
                "--ball-imgsz",
                str(ball_imgsz),
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
                str(ball_debug_frame_stride),
                "--ball-min-area",
                str(ball_min_area),
                "--ball-max-area",
                str(ball_max_area),
                "--ball-min-width",
                str(ball_min_width),
                "--ball-max-width",
                str(ball_max_width),
                "--ball-min-height",
                str(ball_min_height),
                "--ball-max-height",
                str(ball_max_height),
                "--ball-max-detections-per-frame",
                str(ball_max_detections_per_frame),
                "--ball-exclude-top-ratio",
                str(ball_exclude_top_ratio),
                "--ball-exclude-bottom-ratio",
                str(ball_exclude_bottom_ratio),
            ]
        )

    run_command(main_command)

    if detect_ball and ball_model_path != "yolov8n.pt":
        run_command(
            [
                python,
                "-m",
                "src.detection.compare_ball_models",
                "--video",
                str(clip_path),
                "--candidate-model",
                ball_model_path,
                "--generic-model",
                "yolov8n.pt",
                "--generic-conf",
                "0.10",
                "--candidate-conf",
                str(ball_conf),
                "--imgsz",
                str(ball_imgsz),
                "--ball-min-area",
                str(ball_min_area),
                "--ball-max-area",
                str(ball_max_area),
                "--ball-min-width",
                str(ball_min_width),
                "--ball-max-width",
                str(ball_max_width),
                "--ball-min-height",
                str(ball_min_height),
                "--ball-max-height",
                str(ball_max_height),
                "--ball-max-detections-per-frame",
                str(ball_max_detections_per_frame),
                "--ball-exclude-top-ratio",
                str(ball_exclude_top_ratio),
                "--ball-exclude-bottom-ratio",
                str(ball_exclude_bottom_ratio),
            ]
        )

    if detect_ball:
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
                str(ball_track_max_distance),
                "--max-gap",
                str(ball_track_max_gap),
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

    if detect_ball:
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
            str(possession_max_distance),
            "--min-track-confidence",
            str(possession_min_track_confidence),
            "--min-ball-confidence",
            str(possession_min_ball_confidence),
            "--switch-confirmation-frames",
            str(possession_switch_confirmation_frames),
        ]
        if possession_assign_interpolated:
            possession_command.append("--assign-interpolated")
        run_command(possession_command)

        run_command(
            [
                python,
                "-m",
                "src.analytics.carries",
                "--possession-csv",
                str(outputs["possession_csv"]),
                "--possession-debug-csv",
                str(outputs["possession_debug_csv"]),
                "--player-tracks-csv",
                str(outputs["tracks_csv"]),
                "--output-csv",
                str(outputs["carry_events_csv"]),
                "--summary-csv",
                str(outputs["carry_summary_csv"]),
                "--summary-md",
                str(outputs["carry_summary_md"]),
            ]
        )
        run_command(
            [
                python,
                "-m",
                "src.analytics.carry_map",
                "--carry-events-csv",
                str(outputs["carry_events_csv"]),
                "--output-dir",
                str(outputs["carry_maps_dir"]),
            ]
        )
        run_command(
            [
                python,
                "-m",
                "src.analytics.interceptions",
                "--possession-csv",
                str(outputs["possession_csv"]),
                "--possession-debug-csv",
                str(outputs["possession_debug_csv"]),
                "--output-csv",
                str(outputs["interceptions_csv"]),
                "--summary-csv",
                str(outputs["interceptions_summary_csv"]),
                "--summary-md",
                str(outputs["interceptions_summary_md"]),
            ]
        )
        run_command(
            [
                python,
                "-m",
                "src.analytics.interception_map",
                "--interceptions-csv",
                str(outputs["interceptions_csv"]),
                "--output-dir",
                str(outputs["interception_maps_dir"]),
            ]
        )
        run_command(
            [
                python,
                "-m",
                "src.analytics.passing_stats",
                "--possession-csv",
                str(outputs["possession_csv"]),
                "--output-csv",
                str(outputs["passing_summary_csv"]),
                "--output-md",
                str(outputs["passing_summary_md"]),
            ]
        )
        run_command(
            [
                python,
                "-m",
                "src.analytics.generate_passing_network",
                "--possession-csv",
                str(outputs["possession_csv"]),
                "--possession-debug-csv",
                str(outputs["possession_debug_csv"]),
                "--output-dir",
                str(outputs["passing_maps_dir"]),
            ]
        )

    print("720p analytics outputs:")
    for output_name, output_path in outputs.items():
        if (
            output_name.startswith("ball_")
            or output_name.startswith("possession_")
            or output_name.startswith("carry_")
            or output_name.startswith("interception")
            or output_name.startswith("passing_")
        ) and not detect_ball:
            continue
        print(f"- {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "If SoccerNet 1_720p.mkv exists, cut a 30-second 720p clip and "
            "run all analytics outputs with a _720p suffix."
        )
    )
    parser.add_argument(
        "--soccernet-dir",
        type=Path,
        default=DEFAULT_SOCCERNET_DIR,
        help="Directory containing the local SoccerNet files",
    )
    parser.add_argument(
        "--source-name",
        default=DEFAULT_SOURCE_NAME,
        help="SoccerNet source filename to search for",
    )
    parser.add_argument(
        "--clip-output",
        type=Path,
        default=DEFAULT_CLIP_PATH,
        help="Path to save the 30-second 720p clip",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Clip duration in seconds",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path or name")
    parser.add_argument("--conf", type=float, default=0.15, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument(
        "--detect-ball",
        action="store_true",
        help="Also run the ball detection baseline on the 720p clip",
    )
    parser.add_argument(
        "--ball-model",
        default="yolov8n.pt",
        help="YOLO model path or name to use for ball detection",
    )
    parser.add_argument(
        "--ball-conf",
        type=float,
        default=0.10,
        help="Ball detection confidence threshold",
    )
    parser.add_argument(
        "--ball-imgsz",
        type=int,
        default=1280,
        help="Ball detection inference image size",
    )
    parser.add_argument("--ball-min-area", type=int, default=20)
    parser.add_argument("--ball-max-area", type=int, default=500)
    parser.add_argument("--ball-min-width", type=int, default=4)
    parser.add_argument("--ball-max-width", type=int, default=30)
    parser.add_argument("--ball-min-height", type=int, default=4)
    parser.add_argument("--ball-max-height", type=int, default=30)
    parser.add_argument("--ball-max-detections-per-frame", type=int, default=1)
    parser.add_argument("--ball-exclude-top-ratio", type=float, default=0.08)
    parser.add_argument("--ball-exclude-bottom-ratio", type=float, default=0.0)
    parser.add_argument(
        "--ball-debug-frame-stride",
        type=int,
        default=0,
        help="Save one ball debug frame every N frames; 0 disables frame export",
    )
    parser.add_argument("--ball-track-max-distance", type=float, default=90.0)
    parser.add_argument("--ball-track-max-gap", type=int, default=8)
    parser.add_argument("--possession-max-distance", type=float, default=80.0)
    parser.add_argument("--possession-min-track-confidence", type=float, default=0.10)
    parser.add_argument("--possession-min-ball-confidence", type=float, default=0.25)
    parser.add_argument("--possession-switch-confirmation-frames", type=int, default=3)
    parser.add_argument(
        "--possession-assign-interpolated",
        action="store_true",
        help="Allow interpolated ball points to assign possession. Default skips them.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite an existing 30-second clip",
    )
    return parser.parse_args()


def main() -> int:
    """Run the 720p clip workflow."""
    args = parse_args()
    source_video = find_source_video(args.soccernet_dir, args.source_name)
    if source_video is None:
        print(f"Skipping: {args.source_name} was not found under {args.soccernet_dir}")
        return 0

    try:
        cut_clip(
            source_video=source_video,
            clip_path=args.clip_output,
            duration_seconds=args.duration,
            overwrite=not args.no_overwrite,
        )
        run_analytics(
            clip_path=args.clip_output,
            model_path=args.model,
            conf=args.conf,
            imgsz=args.imgsz,
            detect_ball=args.detect_ball,
            ball_model_path=args.ball_model,
            ball_conf=args.ball_conf,
            ball_imgsz=args.ball_imgsz,
            ball_min_area=args.ball_min_area,
            ball_max_area=args.ball_max_area,
            ball_min_width=args.ball_min_width,
            ball_max_width=args.ball_max_width,
            ball_min_height=args.ball_min_height,
            ball_max_height=args.ball_max_height,
            ball_max_detections_per_frame=args.ball_max_detections_per_frame,
            ball_exclude_top_ratio=args.ball_exclude_top_ratio,
            ball_exclude_bottom_ratio=args.ball_exclude_bottom_ratio,
            ball_debug_frame_stride=args.ball_debug_frame_stride,
            ball_track_max_distance=args.ball_track_max_distance,
            ball_track_max_gap=args.ball_track_max_gap,
            possession_max_distance=args.possession_max_distance,
            possession_min_track_confidence=args.possession_min_track_confidence,
            possession_min_ball_confidence=args.possession_min_ball_confidence,
            possession_switch_confirmation_frames=args.possession_switch_confirmation_frames,
            possession_assign_interpolated=args.possession_assign_interpolated,
        )
    except subprocess.CalledProcessError as error:
        print(f"Error: command failed with exit code {error.returncode}")
        return error.returncode
    except RuntimeError as error:
        print(f"Error: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
