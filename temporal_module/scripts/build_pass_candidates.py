"""Build confidence-tiered weak pass candidates without changing legacy pass exports."""

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

from src.io_utils import PROJECT_ROOT, ensure_output_parent, reject_outputs_path  # noqa: E402


TEAM_LABELS = {"Team A", "Team B"}
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
PASS_COLUMNS = COMMON_COLUMNS + [
    "from_player_id",
    "to_player_id",
    "same_team_transition",
    "previous_owner_stable_frames",
    "new_owner_stable_frames",
    "transition_duration_seconds",
    "valid_ball_motion_frame_count",
    "ball_missing_or_interpolated_fraction",
    "matched_weak_pass",
    "temporal_consistency_ok",
]
SUMMARY_COLUMNS = [
    "clip_id",
    "status",
    "temporal_frame_rows",
    "weak_pass_rows",
    "candidate_rows",
    "high_confidence_count",
    "medium_confidence_count",
    "low_confidence_count",
    "output_path",
    "missing_inputs",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build improved weak pass candidate rows.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--derived-root", default=str(Path("temporal_module") / "data" / "derived"))
    parser.add_argument("--max-transition-seconds", type=float, default=2.0)
    parser.add_argument("--min-previous-owner-frames", type=int, default=3)
    parser.add_argument("--min-new-owner-frames", type=int, default=2)
    parser.add_argument("--motion-lookaround-frames", type=int, default=12)
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


def read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def require_columns(df: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing column(s): {', '.join(sorted(missing))}")


def normalize_temporal_frames(frames: pd.DataFrame) -> pd.DataFrame:
    required = {"frame", "timestamp", "possessor_track_id", "possession_team"}
    missing = required - set(frames.columns)
    if missing:
        raise ValueError(f"temporal_frames.csv missing column(s): {', '.join(sorted(missing))}")
    result = frames.copy()
    numeric_columns = [
        "frame",
        "timestamp",
        "possessor_track_id",
        "distance_to_ball",
        "ball_speed",
        "ball_acceleration",
        "ball_velocity_valid",
        "ball_acceleration_valid",
        "ball_missing",
        "ball_is_interpolated",
        "frames_since_possession_change",
    ]
    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result = result.dropna(subset=["frame", "timestamp"]).copy()
    result["frame"] = result["frame"].astype(int)
    result["possession_team"] = result["possession_team"].astype(str)
    return result.sort_values(["frame", "timestamp"]).reset_index(drop=True)


def build_owner_segments(frames: pd.DataFrame) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for row in frames.itertuples(index=False):
        team = str(getattr(row, "possession_team"))
        player_value = getattr(row, "possessor_track_id")
        valid_owner = team in TEAM_LABELS and not pd.isna(player_value) and int(player_value) >= 0
        owner = (team, int(player_value)) if valid_owner else None
        if owner is None:
            if current is not None:
                segments.append(current)
                current = None
            continue
        frame = int(getattr(row, "frame"))
        timestamp = float(getattr(row, "timestamp"))
        if (
            current is None
            or current["team"] != owner[0]
            or current["player_id"] != owner[1]
            or frame != current["end_frame"] + 1
        ):
            if current is not None:
                segments.append(current)
            current = {
                "team": owner[0],
                "player_id": owner[1],
                "start_frame": frame,
                "end_frame": frame,
                "start_timestamp": timestamp,
                "end_timestamp": timestamp,
                "frames": 1,
            }
        else:
            current["end_frame"] = frame
            current["end_timestamp"] = timestamp
            current["frames"] += 1
    if current is not None:
        segments.append(current)
    return segments


def motion_context(frames: pd.DataFrame, start_frame: int, end_frame: int, lookaround: int) -> dict[str, Any]:
    span = frames[(frames["frame"] >= start_frame - lookaround) & (frames["frame"] <= end_frame + lookaround)].copy()
    event_span = frames[(frames["frame"] >= start_frame) & (frames["frame"] <= end_frame)].copy()
    speed = pd.to_numeric(span.get("ball_speed", pd.Series(dtype=float)), errors="coerce")
    accel = pd.to_numeric(span.get("ball_acceleration", pd.Series(dtype=float)), errors="coerce")
    valid_motion = (
        (pd.to_numeric(span.get("ball_velocity_valid", 0), errors="coerce").fillna(0) == 1)
        | (pd.to_numeric(span.get("ball_acceleration_valid", 0), errors="coerce").fillna(0) == 1)
    )
    missing = pd.to_numeric(event_span.get("ball_missing", 0), errors="coerce").fillna(0)
    interpolated = pd.to_numeric(event_span.get("ball_is_interpolated", 0), errors="coerce").fillna(0)
    frame_values = event_span["frame"].to_numpy(dtype=int) if not event_span.empty else np.asarray([], dtype=int)
    frame_gap_count = int((np.diff(frame_values) != 1).sum()) if len(frame_values) > 1 else 0
    return {
        "ball_speed_peak": max_or_nan(speed),
        "ball_acceleration_peak": max_or_nan(accel),
        "valid_ball_motion_frame_count": int(valid_motion.sum()),
        "ball_missing_or_interpolated_fraction": float(((missing == 1) | (interpolated == 1)).mean()) if len(event_span) else np.nan,
        "frame_gap_count": frame_gap_count,
        "ball_distance_px": min_or_nan(pd.to_numeric(event_span.get("distance_to_ball", pd.Series(dtype=float)), errors="coerce")),
    }


def match_weak_pass(weak_passes: pd.DataFrame, previous: dict[str, Any], current: dict[str, Any]) -> tuple[bool, str]:
    if weak_passes.empty:
        return False, ""
    required = {"pass_id", "start_frame", "end_frame", "from_player_id", "to_player_id", "team"}
    if not required.issubset(weak_passes.columns):
        return False, ""
    candidates = weak_passes.copy()
    for column in ["start_frame", "end_frame", "from_player_id", "to_player_id", "pass_id"]:
        candidates[column] = pd.to_numeric(candidates[column], errors="coerce")
    matched = candidates[
        (candidates["team"].astype(str) == current["team"])
        & (candidates["from_player_id"] == previous["player_id"])
        & (candidates["to_player_id"] == current["player_id"])
        & (candidates["start_frame"] <= current["start_frame"])
        & (candidates["end_frame"] >= previous["end_frame"])
    ]
    if matched.empty:
        return False, ""
    return True, str(int(matched.iloc[0]["pass_id"]))


def build_candidates_for_clip(
    clip_id: str,
    frames: pd.DataFrame,
    weak_passes: pd.DataFrame,
    weak_source_file: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    segments = build_owner_segments(frames)
    speed_threshold = quantile_or_inf(frames.get("ball_speed", pd.Series(dtype=float)), 0.75)
    accel_threshold = quantile_or_inf(frames.get("ball_acceleration", pd.Series(dtype=float)), 0.75)
    rows: list[dict[str, Any]] = []
    for previous, current in zip(segments, segments[1:]):
        if previous["team"] != current["team"]:
            continue
        if previous["player_id"] == current["player_id"]:
            continue
        transition_duration = max(0.0, float(current["start_timestamp"]) - float(previous["end_timestamp"]))
        if transition_duration > args.max_transition_seconds:
            continue
        start_frame = int(previous["end_frame"])
        end_frame = int(current["start_frame"])
        context = motion_context(frames, start_frame, end_frame, args.motion_lookaround_frames)
        matched_weak, source_event_id = match_weak_pass(weak_passes, previous, current)
        previous_stable = int(previous["frames"])
        new_stable = int(current["frames"])
        stable = previous_stable >= args.min_previous_owner_frames and new_stable >= args.min_new_owner_frames
        valid_motion = context["valid_ball_motion_frame_count"] > 0
        strong_motion = (
            finite_ge(context["ball_speed_peak"], speed_threshold)
            or finite_ge(context["ball_acceleration_peak"], accel_threshold)
        )
        temporal_ok = stable and context["frame_gap_count"] == 0 and transition_duration <= args.max_transition_seconds
        if temporal_ok and valid_motion and strong_motion:
            tier = "high"
            confidence = 0.85
        elif temporal_ok and (valid_motion or matched_weak):
            tier = "medium"
            confidence = 0.65
        else:
            tier = "low"
            confidence = 0.35
        reasons = [
            "same_team_player_change",
            f"previous_stable={previous_stable}",
            f"new_stable={new_stable}",
            f"valid_ball_motion_frames={context['valid_ball_motion_frame_count']}",
        ]
        if matched_weak:
            reasons.append("matched_weak_pass")
        row = {
            "clip_id": clip_id,
            "event_id": len(rows) + 1,
            "event_type": "pass_candidate",
            "start_frame": start_frame,
            "end_frame": end_frame,
            "center_frame": int(round((start_frame + end_frame) / 2)),
            "start_timestamp": float(previous["end_timestamp"]),
            "end_timestamp": float(current["start_timestamp"]),
            "team": current["team"],
            "player_id": int(previous["player_id"]),
            "secondary_player_id": int(current["player_id"]),
            "confidence": confidence,
            "confidence_tier": tier,
            "quality_flag": "weak_candidate_not_ground_truth",
            "label_source": "possession_transition_with_ball_motion_heuristic",
            "rule_reasons": ";".join(reasons),
            "ball_speed_peak": context["ball_speed_peak"],
            "ball_acceleration_peak": context["ball_acceleration_peak"],
            "ball_distance_px": context["ball_distance_px"],
            "possession_before": f"{previous['team']}:{previous['player_id']}",
            "possession_after": f"{current['team']}:{current['player_id']}",
            "frame_gap_count": context["frame_gap_count"],
            "source_event_id": source_event_id,
            "source_file": weak_source_file if matched_weak else "",
            "from_player_id": int(previous["player_id"]),
            "to_player_id": int(current["player_id"]),
            "same_team_transition": 1,
            "previous_owner_stable_frames": previous_stable,
            "new_owner_stable_frames": new_stable,
            "transition_duration_seconds": transition_duration,
            "valid_ball_motion_frame_count": context["valid_ball_motion_frame_count"],
            "ball_missing_or_interpolated_fraction": context["ball_missing_or_interpolated_fraction"],
            "matched_weak_pass": int(matched_weak),
            "temporal_consistency_ok": int(temporal_ok),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=PASS_COLUMNS)


def write_schema(path: Path, payload: dict[str, Any]) -> None:
    output = ensure_output_parent(path)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def process_clip(clip_id: str, outputs_root: Path, derived_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    frames_path = derived_root / clip_id / "temporal_frames.csv"
    weak_path = derived_root / clip_id / "passes_weak.csv"
    possession_path = outputs_root / clip_id / "possession" / "possession.csv"
    ball_tracks_path = outputs_root / clip_id / "tracks" / "ball_tracks.csv"
    frames = normalize_temporal_frames(pd.read_csv(frames_path))
    weak_passes = read_optional_csv(weak_path)
    _ = read_optional_csv(possession_path)
    _ = read_optional_csv(ball_tracks_path)
    missing_inputs = ";".join(
        name
        for name, path in [
            ("passes_weak", weak_path),
            ("possession", possession_path),
            ("ball_tracks", ball_tracks_path),
        ]
        if not path.exists()
    )
    candidates = build_candidates_for_clip(
        clip_id=clip_id,
        frames=frames,
        weak_passes=weak_passes,
        weak_source_file=str(weak_path) if weak_path.exists() else "",
        args=args,
    )
    output_path = ensure_output_parent(derived_root / clip_id / "events" / "pass_candidates.csv")
    candidates.to_csv(output_path, index=False)
    write_schema(
        output_path.with_name("pass_candidates_schema.json"),
        {
            "clip_id": clip_id,
            "input_paths": {
                "temporal_frames": str(frames_path),
                "passes_weak": str(weak_path) if weak_path.exists() else "",
                "possession": str(possession_path) if possession_path.exists() else "",
                "ball_tracks": str(ball_tracks_path) if ball_tracks_path.exists() else "",
            },
            "output_columns": PASS_COLUMNS,
            "candidate_count": int(len(candidates)),
            "settings": {
                "max_transition_seconds": args.max_transition_seconds,
                "min_previous_owner_frames": args.min_previous_owner_frames,
                "min_new_owner_frames": args.min_new_owner_frames,
                "motion_lookaround_frames": args.motion_lookaround_frames,
            },
            "warnings": ["Weak pass candidates are heuristic candidates, not ground truth."],
        },
    )
    return {
        "clip_id": clip_id,
        "status": "success",
        "temporal_frame_rows": int(len(frames)),
        "weak_pass_rows": int(len(weak_passes)),
        "candidate_rows": int(len(candidates)),
        "high_confidence_count": int((candidates["confidence_tier"] == "high").sum()) if not candidates.empty else 0,
        "medium_confidence_count": int((candidates["confidence_tier"] == "medium").sum()) if not candidates.empty else 0,
        "low_confidence_count": int((candidates["confidence_tier"] == "low").sum()) if not candidates.empty else 0,
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


def quantile_or_inf(values: pd.Series, quantile: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.quantile(quantile)) if len(clean) else float("inf")


def finite_ge(value: float, threshold: float) -> bool:
    return np.isfinite(value) and np.isfinite(threshold) and value >= threshold


def main() -> int:
    args = parse_args()
    try:
        derived_root = enforce_derived_root(Path(args.derived_root))
        outputs_root = Path(args.outputs_root)
        rows: list[dict[str, Any]] = []
        for clip_id in eligible_clips(derived_root):
            try:
                rows.append(process_clip(clip_id, outputs_root, derived_root, args))
            except Exception as error:
                frames_path = derived_root / clip_id / "temporal_frames.csv"
                weak_path = derived_root / clip_id / "passes_weak.csv"
                rows.append(
                    {
                        "clip_id": clip_id,
                        "status": "failed",
                        "temporal_frame_rows": len(pd.read_csv(frames_path)) if frames_path.exists() else 0,
                        "weak_pass_rows": len(pd.read_csv(weak_path)) if weak_path.exists() else 0,
                        "candidate_rows": 0,
                        "high_confidence_count": 0,
                        "medium_confidence_count": 0,
                        "low_confidence_count": 0,
                        "output_path": "",
                        "missing_inputs": "",
                        "error_message": str(error),
                    }
                )
        summary_path = ensure_output_parent(derived_root / "pass_candidates_build_summary.csv")
        with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print("Pass candidate build summary:")
        for row in rows:
            print(f"{row['clip_id']}: {row['status']} candidates={row['candidate_rows']} error={row['error_message']}")
        print(f"Summary CSV: {summary_path}")
        return 0 if rows and all(row["status"] == "success" for row in rows) else 1
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
