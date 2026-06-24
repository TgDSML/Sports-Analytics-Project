"""Build player movement analytics and temporal movement features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import PROJECT_ROOT, ensure_output_parent, reject_outputs_path, utc_now_iso  # noqa: E402


INCLUDED_CLASSES = {"person", "player"}
LOW_CONFIDENCE_THRESHOLD = 0.30
HIGH_MOTION_PERCENTILE = 90.0

SUMMARY_COLUMNS = [
    "clip_id",
    "track_id",
    "team",
    "role",
    "first_frame",
    "last_frame",
    "first_timestamp",
    "last_timestamp",
    "observed_frames",
    "time_on_screen_seconds",
    "consecutive_motion_pairs",
    "gap_break_count",
    "max_frame_gap",
    "consecutive_segment_count",
    "total_distance_px",
    "mean_speed_px_per_second",
    "median_speed_px_per_second",
    "max_speed_px_per_second",
    "mean_acceleration_px_per_second2",
    "max_acceleration_px_per_second2",
    "mean_center_x",
    "mean_center_y",
    "min_center_x",
    "max_center_x",
    "min_center_y",
    "max_center_y",
    "mean_detection_confidence",
    "low_confidence_fraction",
    "track_fragmentation_risk",
    "movement_quality_flag",
]

TRAJECTORY_COLUMNS = [
    "clip_id",
    "track_id",
    "team",
    "role",
    "frame",
    "timestamp",
    "center_x",
    "center_y",
    "confidence",
    "is_consecutive_from_previous",
    "gap_from_previous_frame",
    "step_distance_px",
    "speed_px_per_second",
    "acceleration_px_per_second2",
    "motion_valid",
    "trajectory_segment_id",
]

HEATMAP_COLUMNS = [
    "clip_id",
    "track_id",
    "team",
    "role",
    "grid_col",
    "grid_row",
    "frame_count",
    "fraction_of_track_frames",
    "image_width_inferred",
    "image_height_inferred",
    "coordinate_space",
]

TEMPORAL_FEATURE_COLUMNS = [
    "clip_id",
    "frame",
    "timestamp",
    "valid_player_count",
    "mean_player_speed_px_per_second",
    "max_player_speed_px_per_second",
    "mean_player_acceleration_px_per_second2",
    "max_player_acceleration_px_per_second2",
    "high_motion_player_count",
    "trajectory_gap_player_count",
    "team_a_mean_speed_px_per_second",
    "team_b_mean_speed_px_per_second",
    "team_a_valid_player_count",
    "team_b_valid_player_count",
    "movement_feature_valid",
]

SUMMARY_BUILD_COLUMNS = [
    "clip_id",
    "status",
    "tracks_input_rows",
    "player_team_input_rows",
    "player_tracks_written",
    "trajectory_rows_written",
    "heatmap_rows_written",
    "temporal_feature_rows_written",
    "output_directory",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build gap-aware player movement analytics for clips with tracks and team assignments."
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--derived-root", default=str(Path("temporal_module") / "data" / "derived"))
    parser.add_argument("--grid-cols", type=int, default=12)
    parser.add_argument("--grid-rows", type=int, default=8)
    return parser.parse_args()


def enforce_derived_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = (PROJECT_ROOT / "temporal_module" / "data" / "derived").resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"Derived outputs must be under {allowed}: {resolved}") from error
    reject_outputs_path(resolved)
    return resolved


def candidate_clips(outputs_root: Path) -> list[str]:
    if not outputs_root.exists():
        raise FileNotFoundError(f"Outputs root not found: {outputs_root}")
    clips = []
    for clip_dir in sorted(path for path in outputs_root.iterdir() if path.is_dir()):
        if (clip_dir / "tracks" / "tracks.csv").exists() and (clip_dir / "teams" / "player_teams.csv").exists():
            clips.append(clip_dir.name)
    return clips


def require_columns(df: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing column(s): {', '.join(sorted(missing))}")


def read_inputs(
    clip_id: str,
    outputs_root: Path,
    derived_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    tracks_path = outputs_root / clip_id / "tracks" / "tracks.csv"
    teams_path = outputs_root / clip_id / "teams" / "player_teams.csv"
    temporal_frames_path = derived_root / clip_id / "temporal_frames.csv"
    if not temporal_frames_path.exists():
        raise FileNotFoundError(f"Missing temporal frame table: {temporal_frames_path}")

    tracks = pd.read_csv(tracks_path)
    teams = pd.read_csv(teams_path)
    temporal_frames = pd.read_csv(temporal_frames_path)
    require_columns(
        tracks,
        tracks_path,
        {"frame", "timestamp", "track_id", "class_name", "confidence", "center_x", "center_y"},
    )
    require_columns(teams, teams_path, {"track_id", "team", "role"})
    require_columns(temporal_frames, temporal_frames_path, {"clip_id", "frame", "timestamp"})

    metadata = {
        "tracks_path": str(tracks_path),
        "player_teams_path": str(teams_path),
        "temporal_frames_path": str(temporal_frames_path),
        "tracks_input_rows": int(len(tracks)),
        "player_team_input_rows": int(len(teams)),
        "temporal_frame_rows": int(len(temporal_frames)),
    }
    return tracks, teams, temporal_frames, metadata


def prepare_player_tracks(tracks: pd.DataFrame, teams: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    class_values = tracks["class_name"].astype(str).str.strip()
    normalized_classes = class_values.str.lower()
    included_class_names = sorted(class_values[normalized_classes.isin(INCLUDED_CLASSES)].dropna().unique().tolist())
    excluded_class_names = sorted(class_values[~normalized_classes.isin(INCLUDED_CLASSES)].dropna().unique().tolist())

    players = tracks[normalized_classes.isin(INCLUDED_CLASSES)].copy()
    for column in ["frame", "timestamp", "track_id", "confidence", "center_x", "center_y"]:
        players[column] = pd.to_numeric(players[column], errors="coerce")
    players = players.dropna(subset=["frame", "timestamp", "track_id", "center_x", "center_y"]).copy()
    players["frame"] = players["frame"].astype(int)
    players["track_id"] = players["track_id"].astype(int)

    team_columns = [column for column in ["track_id", "team", "role"] if column in teams.columns]
    team_info = teams[team_columns].copy()
    team_info["track_id"] = pd.to_numeric(team_info["track_id"], errors="coerce")
    team_info = team_info.dropna(subset=["track_id"]).copy()
    team_info["track_id"] = team_info["track_id"].astype(int)
    team_info = team_info.drop_duplicates("track_id", keep="first")

    players = players.merge(team_info, on="track_id", how="left")
    players["team"] = players["team"].fillna("Unknown").astype(str)
    players["role"] = players["role"].fillna("unknown").astype(str)
    return players.sort_values(["track_id", "frame", "timestamp"]).reset_index(drop=True), included_class_names, excluded_class_names


def build_trajectory_points(clip_id: str, players: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for track_id, group in players.groupby("track_id", sort=True):
        group = group.sort_values(["frame", "timestamp"]).reset_index(drop=True)
        segment_id = 1
        previous_valid_speed: float | None = None
        for index, row in group.iterrows():
            is_consecutive = 0
            gap_from_previous = np.nan
            step_distance = np.nan
            speed = np.nan
            acceleration = np.nan
            motion_valid = 0

            if index > 0:
                previous = group.iloc[index - 1]
                frame_gap = int(row["frame"]) - int(previous["frame"])
                time_gap = float(row["timestamp"]) - float(previous["timestamp"])
                gap_from_previous = frame_gap
                if frame_gap != 1 or time_gap <= 0:
                    segment_id += 1
                    previous_valid_speed = None
                else:
                    dx = float(row["center_x"]) - float(previous["center_x"])
                    dy = float(row["center_y"]) - float(previous["center_y"])
                    step_distance = math.hypot(dx, dy)
                    speed = step_distance / time_gap
                    if previous_valid_speed is not None:
                        acceleration = (speed - previous_valid_speed) / time_gap
                    previous_valid_speed = speed
                    is_consecutive = 1
                    motion_valid = 1

            rows.append(
                {
                    "clip_id": clip_id,
                    "track_id": int(track_id),
                    "team": str(row["team"]),
                    "role": str(row["role"]),
                    "frame": int(row["frame"]),
                    "timestamp": float(row["timestamp"]),
                    "center_x": float(row["center_x"]),
                    "center_y": float(row["center_y"]),
                    "confidence": float(row["confidence"]) if not pd.isna(row["confidence"]) else np.nan,
                    "is_consecutive_from_previous": is_consecutive,
                    "gap_from_previous_frame": gap_from_previous,
                    "step_distance_px": step_distance,
                    "speed_px_per_second": speed,
                    "acceleration_px_per_second2": acceleration,
                    "motion_valid": motion_valid,
                    "trajectory_segment_id": segment_id,
                }
            )
    return pd.DataFrame(rows, columns=TRAJECTORY_COLUMNS)


def fragmentation_risk(max_frame_gap: float, segment_count: int) -> str:
    if max_frame_gap > 10 or segment_count >= 4:
        return "high"
    if max_frame_gap > 3 or segment_count >= 2:
        return "medium"
    return "low"


def build_movement_summary(clip_id: str, trajectory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for track_id, group in trajectory.groupby("track_id", sort=True):
        valid_speed = group.loc[group["motion_valid"] == 1, "speed_px_per_second"].dropna()
        valid_accel = group.loc[group["motion_valid"] == 1, "acceleration_px_per_second2"].dropna().abs()
        gap_values = pd.to_numeric(group["gap_from_previous_frame"], errors="coerce").dropna()
        gap_break_count = int((gap_values > 1).sum())
        max_frame_gap = int(gap_values.max()) if not gap_values.empty else 0
        segment_count = int(pd.to_numeric(group["trajectory_segment_id"], errors="coerce").max()) if not group.empty else 0
        risk = fragmentation_risk(float(max_frame_gap), segment_count)
        consecutive_motion_pairs = int((group["motion_valid"] == 1).sum())
        quality = "ok" if consecutive_motion_pairs >= 2 and risk in {"low", "medium"} else "sparse_track"

        rows.append(
            {
                "clip_id": clip_id,
                "track_id": int(track_id),
                "team": str(group["team"].iloc[0]),
                "role": str(group["role"].iloc[0]),
                "first_frame": int(group["frame"].min()),
                "last_frame": int(group["frame"].max()),
                "first_timestamp": float(group["timestamp"].min()),
                "last_timestamp": float(group["timestamp"].max()),
                "observed_frames": int(len(group)),
                "time_on_screen_seconds": float(group["timestamp"].max() - group["timestamp"].min()),
                "consecutive_motion_pairs": consecutive_motion_pairs,
                "gap_break_count": gap_break_count,
                "max_frame_gap": max_frame_gap,
                "consecutive_segment_count": segment_count,
                "total_distance_px": float(group["step_distance_px"].dropna().sum()),
                "mean_speed_px_per_second": _mean_or_nan(valid_speed),
                "median_speed_px_per_second": _median_or_nan(valid_speed),
                "max_speed_px_per_second": _max_or_nan(valid_speed),
                "mean_acceleration_px_per_second2": _mean_or_nan(valid_accel),
                "max_acceleration_px_per_second2": _max_or_nan(valid_accel),
                "mean_center_x": float(group["center_x"].mean()),
                "mean_center_y": float(group["center_y"].mean()),
                "min_center_x": float(group["center_x"].min()),
                "max_center_x": float(group["center_x"].max()),
                "min_center_y": float(group["center_y"].min()),
                "max_center_y": float(group["center_y"].max()),
                "mean_detection_confidence": _mean_or_nan(group["confidence"].dropna()),
                "low_confidence_fraction": float((group["confidence"] < LOW_CONFIDENCE_THRESHOLD).sum() / len(group)),
                "track_fragmentation_risk": risk,
                "movement_quality_flag": quality,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def infer_dimensions(players: pd.DataFrame) -> tuple[int, int]:
    valid = players.dropna(subset=["center_x", "center_y"])
    if valid.empty:
        return 0, 0
    width = int(math.ceil(float(valid["center_x"].max()))) + 1
    height = int(math.ceil(float(valid["center_y"].max()))) + 1
    return max(width, 1), max(height, 1)


def build_heatmap_cells(
    clip_id: str,
    players: pd.DataFrame,
    image_width: int,
    image_height: int,
    grid_cols: int,
    grid_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if image_width <= 0 or image_height <= 0:
        return pd.DataFrame(rows, columns=HEATMAP_COLUMNS)

    prepared = players.dropna(subset=["center_x", "center_y"]).copy()
    prepared["grid_col"] = np.floor(prepared["center_x"] / image_width * grid_cols).astype(int).clip(0, grid_cols - 1)
    prepared["grid_row"] = np.floor(prepared["center_y"] / image_height * grid_rows).astype(int).clip(0, grid_rows - 1)
    track_counts = prepared.groupby("track_id").size().to_dict()

    for keys, group in prepared.groupby(["track_id", "team", "role", "grid_col", "grid_row"], sort=True):
        track_id, team, role, grid_col, grid_row = keys
        frame_count = int(len(group))
        total_track_frames = int(track_counts.get(track_id, frame_count))
        rows.append(
            {
                "clip_id": clip_id,
                "track_id": int(track_id),
                "team": str(team),
                "role": str(role),
                "grid_col": int(grid_col),
                "grid_row": int(grid_row),
                "frame_count": frame_count,
                "fraction_of_track_frames": float(frame_count / total_track_frames) if total_track_frames else 0.0,
                "image_width_inferred": int(image_width),
                "image_height_inferred": int(image_height),
                "coordinate_space": "image_pixel_space",
            }
        )
    return pd.DataFrame(rows, columns=HEATMAP_COLUMNS)


def build_temporal_features(clip_id: str, temporal_frames: pd.DataFrame, trajectory: pd.DataFrame) -> pd.DataFrame:
    frames = temporal_frames[["frame", "timestamp"]].copy()
    frames["frame"] = pd.to_numeric(frames["frame"], errors="coerce").astype("Int64")
    frames = frames.dropna(subset=["frame"]).copy()
    frames["frame"] = frames["frame"].astype(int)
    frames["timestamp"] = pd.to_numeric(frames["timestamp"], errors="coerce")
    frames["clip_id"] = clip_id

    valid_motion = trajectory[trajectory["motion_valid"] == 1].copy()
    speed_threshold = np.nan
    if not valid_motion.empty and valid_motion["speed_px_per_second"].notna().any():
        speed_threshold = float(np.nanpercentile(valid_motion["speed_px_per_second"], HIGH_MOTION_PERCENTILE))

    aggregate_rows: list[dict[str, Any]] = []
    for frame, group in valid_motion.groupby("frame", sort=True):
        speed = pd.to_numeric(group["speed_px_per_second"], errors="coerce").dropna()
        accel = pd.to_numeric(group["acceleration_px_per_second2"], errors="coerce").dropna().abs()
        team_a = group[group["team"] == "Team A"]
        team_b = group[group["team"] == "Team B"]
        aggregate_rows.append(
            {
                "frame": int(frame),
                "valid_player_count": int(len(group)),
                "mean_player_speed_px_per_second": _mean_or_nan(speed),
                "max_player_speed_px_per_second": _max_or_nan(speed),
                "mean_player_acceleration_px_per_second2": _mean_or_nan(accel),
                "max_player_acceleration_px_per_second2": _max_or_nan(accel),
                "high_motion_player_count": int((speed >= speed_threshold).sum()) if not pd.isna(speed_threshold) else 0,
                "team_a_mean_speed_px_per_second": _mean_or_nan(team_a["speed_px_per_second"].dropna()),
                "team_b_mean_speed_px_per_second": _mean_or_nan(team_b["speed_px_per_second"].dropna()),
                "team_a_valid_player_count": int(len(team_a)),
                "team_b_valid_player_count": int(len(team_b)),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)

    gap_rows = trajectory[pd.to_numeric(trajectory["gap_from_previous_frame"], errors="coerce") > 1]
    gap_counts = gap_rows.groupby("frame").size().rename("trajectory_gap_player_count").reset_index()

    result = frames.merge(aggregate, on="frame", how="left").merge(gap_counts, on="frame", how="left")
    count_columns = [
        "valid_player_count",
        "high_motion_player_count",
        "trajectory_gap_player_count",
        "team_a_valid_player_count",
        "team_b_valid_player_count",
    ]
    for column in count_columns:
        result[column] = result[column].fillna(0).astype(int)
    result["movement_feature_valid"] = (result["valid_player_count"] > 0).astype(int)
    result = result.reindex(columns=TEMPORAL_FEATURE_COLUMNS)
    return result


def write_schema(
    path: Path,
    clip_id: str,
    input_metadata: dict[str, Any],
    included_class_names: list[str],
    excluded_class_names: list[str],
    image_width: int,
    image_height: int,
    grid_cols: int,
    grid_rows: int,
) -> None:
    payload = {
        "clip_id": clip_id,
        "build_timestamp": utc_now_iso(),
        "input_paths": {
            "tracks": input_metadata["tracks_path"],
            "player_teams": input_metadata["player_teams_path"],
            "temporal_frames": input_metadata["temporal_frames_path"],
        },
        "input_row_counts": {
            "tracks": input_metadata["tracks_input_rows"],
            "player_teams": input_metadata["player_team_input_rows"],
            "temporal_frames": input_metadata["temporal_frame_rows"],
        },
        "included_class_names": included_class_names,
        "excluded_class_names": excluded_class_names,
        "inferred_dimensions": {
            "image_width_inferred": int(image_width),
            "image_height_inferred": int(image_height),
            "coordinate_space": "image_pixel_space",
        },
        "grid_settings": {"grid_cols": int(grid_cols), "grid_rows": int(grid_rows)},
        "calculation_definitions": {
            "step_displacement": "Euclidean pixel distance between adjacent observations only when frame_gap == 1 and timestamp delta > 0.",
            "speed": "step_distance_px divided by positive timestamp delta, in pixels per second.",
            "acceleration": "difference between consecutive valid speed values divided by positive timestamp delta, in pixels per second squared.",
            "low_confidence_fraction": f"Fraction of player rows with confidence below {LOW_CONFIDENCE_THRESHOLD}.",
            "high_motion_player_count": f"Per-frame count of valid motion observations at or above the clip-level {HIGH_MOTION_PERCENTILE}th percentile speed.",
        },
        "gap_policy": "Rows are never interpolated or filled. Any frame gap other than 1 or any non-positive timestamp delta breaks the trajectory segment.",
        "quality_flag_definitions": {
            "track_fragmentation_risk": {
                "high": "max_frame_gap > 10 or consecutive_segment_count >= 4",
                "medium": "max_frame_gap > 3 or consecutive_segment_count >= 2",
                "low": "otherwise",
            },
            "movement_quality_flag": {
                "ok": "consecutive_motion_pairs >= 2 and track_fragmentation_risk is low or medium",
                "sparse_track": "otherwise",
            },
        },
        "warnings": [
            "All movement distances and speeds are in image pixel space, not metres or km/h.",
            "Camera motion, zoom, broadcast cuts, and track fragmentation can distort movement analytics.",
        ],
        "output_columns": {
            "player_movement_summary": SUMMARY_COLUMNS,
            "player_trajectory_points": TRAJECTORY_COLUMNS,
            "player_heatmap_cells": HEATMAP_COLUMNS,
            "player_movement_temporal_features": TEMPORAL_FEATURE_COLUMNS,
        },
    }
    output = ensure_output_parent(path)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def process_clip(
    clip_id: str,
    outputs_root: Path,
    derived_root: Path,
    grid_cols: int,
    grid_rows: int,
) -> dict[str, Any]:
    tracks, teams, temporal_frames, input_metadata = read_inputs(clip_id, outputs_root, derived_root)
    players, included_classes, excluded_classes = prepare_player_tracks(tracks, teams)
    trajectory = build_trajectory_points(clip_id, players)
    summary = build_movement_summary(clip_id, trajectory)
    image_width, image_height = infer_dimensions(players)
    heatmap = build_heatmap_cells(clip_id, players, image_width, image_height, grid_cols, grid_rows)
    temporal_features = build_temporal_features(clip_id, temporal_frames, trajectory)

    analytics_dir = derived_root / clip_id / "analytics"
    model_features_dir = derived_root / clip_id / "model_features"
    summary_path = ensure_output_parent(analytics_dir / "player_movement_summary.csv")
    trajectory_path = ensure_output_parent(analytics_dir / "player_trajectory_points.csv")
    heatmap_path = ensure_output_parent(analytics_dir / "player_heatmap_cells.csv")
    schema_path = analytics_dir / "player_movement_schema.json"
    temporal_features_path = ensure_output_parent(model_features_dir / "player_movement_temporal_features.csv")

    summary.to_csv(summary_path, index=False)
    trajectory.to_csv(trajectory_path, index=False)
    heatmap.to_csv(heatmap_path, index=False)
    temporal_features.to_csv(temporal_features_path, index=False)
    write_schema(
        path=schema_path,
        clip_id=clip_id,
        input_metadata=input_metadata,
        included_class_names=included_classes,
        excluded_class_names=excluded_classes,
        image_width=image_width,
        image_height=image_height,
        grid_cols=grid_cols,
        grid_rows=grid_rows,
    )

    return {
        "clip_id": clip_id,
        "status": "success",
        "tracks_input_rows": input_metadata["tracks_input_rows"],
        "player_team_input_rows": input_metadata["player_team_input_rows"],
        "player_tracks_written": int(len(summary)),
        "trajectory_rows_written": int(len(trajectory)),
        "heatmap_rows_written": int(len(heatmap)),
        "temporal_feature_rows_written": int(len(temporal_features)),
        "output_directory": str((derived_root / clip_id).resolve()),
        "error_message": "",
    }


def _mean_or_nan(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else np.nan


def _median_or_nan(values: pd.Series) -> float:
    return float(values.median()) if len(values) else np.nan


def _max_or_nan(values: pd.Series) -> float:
    return float(values.max()) if len(values) else np.nan


def main() -> int:
    args = parse_args()
    try:
        if args.grid_cols <= 0 or args.grid_rows <= 0:
            raise ValueError("--grid-cols and --grid-rows must be positive")
        outputs_root = Path(args.outputs_root)
        derived_root = enforce_derived_root(Path(args.derived_root))
        rows: list[dict[str, Any]] = []
        for clip_id in candidate_clips(outputs_root):
            try:
                rows.append(process_clip(clip_id, outputs_root, derived_root, args.grid_cols, args.grid_rows))
            except Exception as error:
                tracks_path = outputs_root / clip_id / "tracks" / "tracks.csv"
                teams_path = outputs_root / clip_id / "teams" / "player_teams.csv"
                tracks_rows = len(pd.read_csv(tracks_path)) if tracks_path.exists() else 0
                team_rows = len(pd.read_csv(teams_path)) if teams_path.exists() else 0
                rows.append(
                    {
                        "clip_id": clip_id,
                        "status": "failed",
                        "tracks_input_rows": tracks_rows,
                        "player_team_input_rows": team_rows,
                        "player_tracks_written": 0,
                        "trajectory_rows_written": 0,
                        "heatmap_rows_written": 0,
                        "temporal_feature_rows_written": 0,
                        "output_directory": "",
                        "error_message": str(error),
                    }
                )

        summary_path = ensure_output_parent(derived_root / "player_movement_build_summary.csv")
        with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_BUILD_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        print("Player movement build summary:")
        for row in rows:
            print(
                f"{row['clip_id']}: {row['status']} tracks={row['player_tracks_written']} "
                f"trajectory_rows={row['trajectory_rows_written']} error={row['error_message']}"
            )
        print(f"Candidate clips: {len(rows)}")
        print(f"Successful clips: {sum(row['status'] == 'success' for row in rows)}")
        print(f"Failed clips: {sum(row['status'] == 'failed' for row in rows)}")
        print(f"Summary CSV: {summary_path}")
        return 0 if rows and all(row["status"] == "success" for row in rows) else 1
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
