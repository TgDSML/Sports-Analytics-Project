"""Build conservative weak shot candidates from image-space ball motion."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import PROJECT_ROOT, ensure_output_parent, reject_outputs_path  # noqa: E402


COMMON_COLUMNS = [
    "clip_id",
    "event_id",
    "event_type",
    "start_frame",
    "end_frame",
    "center_frame",
    "start_timestamp",
    "end_timestamp",
    "team",
    "player_id",
    "secondary_player_id",
    "confidence",
    "confidence_tier",
    "quality_flag",
    "label_source",
    "rule_reasons",
    "ball_speed_peak",
    "ball_acceleration_peak",
    "ball_distance_px",
    "possession_before",
    "possession_after",
    "frame_gap_count",
    "source_event_id",
    "source_file",
]
SHOT_COLUMNS = COMMON_COLUMNS + [
    "goal_side_candidate",
    "attack_direction_assumed",
    "image_width_inferred",
    "image_height_inferred",
    "near_goal_region",
    "start_ball_x",
    "start_ball_y",
    "end_ball_x",
    "end_ball_y",
    "valid_ball_motion_frame_count",
    "ball_interpolated_fraction",
    "carry_overlap_fraction",
    "geometry_quality_flag",
]
SUMMARY_COLUMNS = [
    "clip_id",
    "status",
    "temporal_frame_rows",
    "candidate_rows",
    "medium_confidence_count",
    "low_confidence_count",
    "inferred_image_width",
    "inferred_image_height",
    "output_path",
    "missing_inputs",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conservative weak shot candidate rows.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--derived-root", default=str(Path("temporal_module") / "data" / "derived"))
    parser.add_argument("--motion-lookaround-frames", type=int, default=12)
    parser.add_argument("--speed-quantile", type=float, default=0.95)
    parser.add_argument("--acceleration-quantile", type=float, default=0.95)
    parser.add_argument("--goal-region-width-fraction", type=float, default=0.20)
    parser.add_argument("--merge-gap-frames", type=int, default=12)
    return parser.parse_args()


def enforce_derived_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = (PROJECT_ROOT / "temporal_module" / "data" / "derived").resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"Candidate outputs must be under {allowed}: {resolved}") from error
    reject_outputs_path(resolved)
    return resolved


def eligible_clips(derived_root: Path) -> list[str]:
    if not derived_root.exists():
        raise FileNotFoundError(f"Derived root not found: {derived_root}")
    return sorted(path.name for path in derived_root.iterdir() if path.is_dir() and (path / "temporal_frames.csv").exists())


def normalize_temporal_frames(frames: pd.DataFrame) -> pd.DataFrame:
    required = {"frame", "timestamp", "ball_x", "ball_y", "ball_speed", "ball_acceleration"}
    missing = required - set(frames.columns)
    if missing:
        raise ValueError(f"temporal_frames.csv missing column(s): {', '.join(sorted(missing))}")
    result = frames.copy()
    for column in [
        "frame",
        "timestamp",
        "ball_x",
        "ball_y",
        "ball_speed",
        "ball_acceleration",
        "ball_velocity_valid",
        "ball_acceleration_valid",
        "ball_missing",
        "ball_is_interpolated",
        "distance_to_ball",
    ]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["frame", "timestamp"]).copy()
    result["frame"] = result["frame"].astype(int)
    return result.sort_values(["frame", "timestamp"]).reset_index(drop=True)


def infer_dimensions(frames: pd.DataFrame) -> tuple[int, int]:
    valid = frames.dropna(subset=["ball_x", "ball_y"])
    if "ball_missing" in valid.columns:
        valid = valid[pd.to_numeric(valid["ball_missing"], errors="coerce").fillna(0) == 0]
    if valid.empty:
        return 0, 0
    width = int(np.ceil(float(valid["ball_x"].max()))) + 1
    height = int(np.ceil(float(valid["ball_y"].max()))) + 1
    return max(width, 1), max(height, 1)


def carry_frames(labels_path: Path) -> set[int]:
    if not labels_path.exists():
        return set()
    labels = pd.read_csv(labels_path)
    if not {"frame", "label", "label_source"}.issubset(labels.columns):
        return set()
    accepted = labels[(labels["label"].astype(str) == "carry") & (labels["label_source"].astype(str) == "carry_csv_ok")]
    return set(pd.to_numeric(accepted["frame"], errors="coerce").dropna().astype(int).tolist())


def peak_frames(frames: pd.DataFrame, speed_quantile: float, acceleration_quantile: float) -> pd.DataFrame:
    speed = pd.to_numeric(frames["ball_speed"], errors="coerce")
    accel = pd.to_numeric(frames["ball_acceleration"], errors="coerce")
    speed_threshold = float(speed.dropna().quantile(speed_quantile)) if speed.notna().any() else float("inf")
    accel_threshold = float(accel.dropna().quantile(acceleration_quantile)) if accel.notna().any() else float("inf")
    valid_motion = (
        (pd.to_numeric(frames.get("ball_velocity_valid", 0), errors="coerce").fillna(0) == 1)
        | (pd.to_numeric(frames.get("ball_acceleration_valid", 0), errors="coerce").fillna(0) == 1)
    )
    peaks = frames[valid_motion & ((speed >= speed_threshold) | (accel >= accel_threshold))].copy()
    peaks["_speed_threshold"] = speed_threshold
    peaks["_acceleration_threshold"] = accel_threshold
    return peaks


def merge_peak_frames(peaks: pd.DataFrame, merge_gap_frames: int) -> list[tuple[int, int]]:
    if peaks.empty:
        return []
    peak_values = sorted(peaks["frame"].astype(int).tolist())
    ranges: list[tuple[int, int]] = []
    start = peak_values[0]
    end = peak_values[0]
    for frame in peak_values[1:]:
        if frame - end <= merge_gap_frames:
            end = frame
        else:
            ranges.append((start, end))
            start = frame
            end = frame
    ranges.append((start, end))
    return ranges


def side_for_x(x: float, width: int, fraction: float) -> str:
    if not np.isfinite(x) or width <= 0:
        return ""
    if x <= width * fraction:
        return "left_image_end"
    if x >= width * (1.0 - fraction):
        return "right_image_end"
    return ""


def build_candidates_for_clip(
    clip_id: str,
    frames: pd.DataFrame,
    carry_frame_set: set[int],
    image_width: int,
    image_height: int,
    args: argparse.Namespace,
) -> pd.DataFrame:
    peaks = peak_frames(frames, args.speed_quantile, args.acceleration_quantile)
    ranges = merge_peak_frames(peaks, args.merge_gap_frames)
    rows: list[dict[str, Any]] = []
    for peak_start, peak_end in ranges:
        start_frame = int(peak_start - args.motion_lookaround_frames)
        end_frame = int(peak_end + args.motion_lookaround_frames)
        span = frames[(frames["frame"] >= start_frame) & (frames["frame"] <= end_frame)].copy()
        if span.empty:
            continue
        start_row = span.iloc[0]
        end_row = span.iloc[-1]
        frame_values = span["frame"].to_numpy(dtype=int)
        ball_speed_peak = max_or_nan(span["ball_speed"])
        ball_acceleration_peak = max_or_nan(span["ball_acceleration"])
        valid_motion = (
            (pd.to_numeric(span.get("ball_velocity_valid", 0), errors="coerce").fillna(0) == 1)
            | (pd.to_numeric(span.get("ball_acceleration_valid", 0), errors="coerce").fillna(0) == 1)
        )
        interpolated = pd.to_numeric(span.get("ball_is_interpolated", 0), errors="coerce").fillna(0)
        missing = pd.to_numeric(span.get("ball_missing", 0), errors="coerce").fillna(0)
        carry_overlap = float(span["frame"].isin(carry_frame_set).mean()) if len(span) else 0.0
        side_values = [
            side_for_x(float(x), image_width, args.goal_region_width_fraction)
            for x in span["ball_x"].dropna().tolist()
        ]
        side_values = [value for value in side_values if value]
        goal_side = side_values[0] if side_values else ""
        near_goal_region = int(bool(side_values))
        non_interpolated_evidence = bool(((missing == 0) & (interpolated == 0) & valid_motion).any())
        strong_speed = np.isfinite(ball_speed_peak) and not peaks.empty and ball_speed_peak >= float(peaks["_speed_threshold"].iloc[0])
        no_strong_carry_overlap = carry_overlap < 0.25
        if strong_speed and non_interpolated_evidence and near_goal_region and no_strong_carry_overlap:
            tier = "medium"
            confidence = 0.55
        else:
            tier = "low"
            confidence = 0.25 if carry_overlap >= 0.25 else 0.35
        reasons = [
            "ball_motion_spike",
            f"near_goal_region={near_goal_region}",
            f"non_interpolated_evidence={int(non_interpolated_evidence)}",
            f"carry_overlap_fraction={carry_overlap:.3f}",
        ]
        rows.append(
            {
                "clip_id": clip_id,
                "event_id": len(rows) + 1,
                "event_type": "shot_candidate",
                "start_frame": int(span["frame"].min()),
                "end_frame": int(span["frame"].max()),
                "center_frame": int(round((int(peak_start) + int(peak_end)) / 2)),
                "start_timestamp": float(start_row["timestamp"]),
                "end_timestamp": float(end_row["timestamp"]),
                "team": "",
                "player_id": "",
                "secondary_player_id": "",
                "confidence": confidence,
                "confidence_tier": tier,
                "quality_flag": "weak_candidate_not_confirmed_shot",
                "label_source": "heuristic_ball_motion_candidate",
                "rule_reasons": ";".join(reasons),
                "ball_speed_peak": ball_speed_peak,
                "ball_acceleration_peak": ball_acceleration_peak,
                "ball_distance_px": min_or_nan(span.get("distance_to_ball", pd.Series(dtype=float))),
                "possession_before": "",
                "possession_after": "",
                "frame_gap_count": int((np.diff(frame_values) != 1).sum()) if len(frame_values) > 1 else 0,
                "source_event_id": "",
                "source_file": "",
                "goal_side_candidate": goal_side,
                "attack_direction_assumed": "unknown",
                "image_width_inferred": int(image_width),
                "image_height_inferred": int(image_height),
                "near_goal_region": near_goal_region,
                "start_ball_x": float(start_row["ball_x"]) if not pd.isna(start_row["ball_x"]) else np.nan,
                "start_ball_y": float(start_row["ball_y"]) if not pd.isna(start_row["ball_y"]) else np.nan,
                "end_ball_x": float(end_row["ball_x"]) if not pd.isna(end_row["ball_x"]) else np.nan,
                "end_ball_y": float(end_row["ball_y"]) if not pd.isna(end_row["ball_y"]) else np.nan,
                "valid_ball_motion_frame_count": int(valid_motion.sum()),
                "ball_interpolated_fraction": float(interpolated.mean()) if len(span) else np.nan,
                "carry_overlap_fraction": carry_overlap,
                "geometry_quality_flag": "weak_image_space_only",
            }
        )
    return pd.DataFrame(rows, columns=SHOT_COLUMNS)


def write_schema(path: Path, payload: dict[str, Any]) -> None:
    output = ensure_output_parent(path)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def process_clip(clip_id: str, outputs_root: Path, derived_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    frames_path = derived_root / clip_id / "temporal_frames.csv"
    ball_tracks_path = outputs_root / clip_id / "tracks" / "ball_tracks.csv"
    labels_path = derived_root / clip_id / "temporal_labels.csv"
    frames = normalize_temporal_frames(pd.read_csv(frames_path))
    _ = pd.read_csv(ball_tracks_path) if ball_tracks_path.exists() else pd.DataFrame()
    missing_inputs = ";".join(
        name
        for name, path in [
            ("ball_tracks", ball_tracks_path),
            ("temporal_labels", labels_path),
        ]
        if not path.exists()
    )
    image_width, image_height = infer_dimensions(frames)
    candidates = build_candidates_for_clip(
        clip_id=clip_id,
        frames=frames,
        carry_frame_set=carry_frames(labels_path),
        image_width=image_width,
        image_height=image_height,
        args=args,
    )
    output_path = ensure_output_parent(derived_root / clip_id / "events" / "shot_candidates.csv")
    candidates.to_csv(output_path, index=False)
    write_schema(
        output_path.with_name("shot_candidates_schema.json"),
        {
            "clip_id": clip_id,
            "input_paths": {
                "temporal_frames": str(frames_path),
                "ball_tracks": str(ball_tracks_path) if ball_tracks_path.exists() else "",
                "temporal_labels": str(labels_path) if labels_path.exists() else "",
            },
            "output_columns": SHOT_COLUMNS,
            "candidate_count": int(len(candidates)),
            "settings": {
                "motion_lookaround_frames": args.motion_lookaround_frames,
                "speed_quantile": args.speed_quantile,
                "acceleration_quantile": args.acceleration_quantile,
                "goal_region_width_fraction": args.goal_region_width_fraction,
                "merge_gap_frames": args.merge_gap_frames,
            },
            "known_limitations": [
                "No homography or pitch calibration.",
                "No reliable attack direction.",
                "No detected goal posts.",
                "No official shot labels.",
                "Measurements are image-space only.",
                "Camera motion may affect apparent ball motion.",
            ],
        },
    )
    return {
        "clip_id": clip_id,
        "status": "success",
        "temporal_frame_rows": int(len(frames)),
        "candidate_rows": int(len(candidates)),
        "medium_confidence_count": int((candidates["confidence_tier"] == "medium").sum()) if not candidates.empty else 0,
        "low_confidence_count": int((candidates["confidence_tier"] == "low").sum()) if not candidates.empty else 0,
        "inferred_image_width": int(image_width),
        "inferred_image_height": int(image_height),
        "output_path": str(output_path),
        "missing_inputs": missing_inputs,
        "error_message": "",
    }


def max_or_nan(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.max()) if len(clean) else np.nan


def min_or_nan(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.min()) if len(clean) else np.nan


def main() -> int:
    args = parse_args()
    try:
        if not 0.0 < args.speed_quantile < 1.0:
            raise ValueError("--speed-quantile must be between 0 and 1")
        if not 0.0 < args.acceleration_quantile < 1.0:
            raise ValueError("--acceleration-quantile must be between 0 and 1")
        if not 0.0 < args.goal_region_width_fraction < 0.5:
            raise ValueError("--goal-region-width-fraction must be between 0 and 0.5")
        derived_root = enforce_derived_root(Path(args.derived_root))
        outputs_root = Path(args.outputs_root)
        rows: list[dict[str, Any]] = []
        for clip_id in eligible_clips(derived_root):
            try:
                rows.append(process_clip(clip_id, outputs_root, derived_root, args))
            except Exception as error:
                frames_path = derived_root / clip_id / "temporal_frames.csv"
                rows.append(
                    {
                        "clip_id": clip_id,
                        "status": "failed",
                        "temporal_frame_rows": len(pd.read_csv(frames_path)) if frames_path.exists() else 0,
                        "candidate_rows": 0,
                        "medium_confidence_count": 0,
                        "low_confidence_count": 0,
                        "inferred_image_width": 0,
                        "inferred_image_height": 0,
                        "output_path": "",
                        "missing_inputs": "",
                        "error_message": str(error),
                    }
                )
        summary_path = ensure_output_parent(derived_root / "shot_candidates_build_summary.csv")
        with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print("Shot candidate build summary:")
        for row in rows:
            print(f"{row['clip_id']}: {row['status']} candidates={row['candidate_rows']} error={row['error_message']}")
        print(f"Summary CSV: {summary_path}")
        return 0 if rows and all(row["status"] == "success" for row in rows) else 1
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
