"""Score refined pass candidates with transparent heuristic evidence."""

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

from src.io_utils import PROJECT_ROOT, ensure_output_parent, reject_outputs_path, utc_now_iso  # noqa: E402


IDENTITY_COLUMNS = [
    "clip_id",
    "raw_event_id",
    "refined_event_id",
    "start_frame",
    "end_frame",
    "center_frame",
    "team",
    "confidence",
    "confidence_tier",
    "refinement_status",
    "refinement_reasons",
    "is_recommended_for_review",
    "is_eligible_for_weak_label",
]

FEATURE_COLUMNS = [
    "candidate_duration_frames",
    "candidate_duration_seconds",
    "sender_stable_frames_before",
    "receiver_stable_frames_after",
    "same_team_sender_receiver",
    "sender_ball_distance_before",
    "receiver_ball_distance_after",
    "ball_distance_during_candidate",
    "ball_speed_mean",
    "ball_speed_peak",
    "ball_speed_above_baseline",
    "ball_acceleration_peak",
    "ball_missing_fraction",
    "ball_interpolated_fraction",
    "free_ball_frames",
    "free_ball_fraction",
    "possession_change_count_in_candidate",
    "possession_flicker_warning",
    "carry_overlap_fraction",
    "turnover_overlap_fraction",
    "shot_overlap_fraction",
    "nearby_opponent_count_sender",
    "nearby_opponent_count_receiver",
    "receiver_approach_speed_when_available",
    "sender_departure_speed_when_available",
    "feature_completeness_fraction",
    "evidence_reasons",
]

SCORE_COLUMNS = [
    "pass_plausibility_score",
    "pass_score_tier",
    "pass_score_status",
    "pass_score_reasons",
    "is_eligible_for_pass_score_weak_label",
]

CALIBRATED_COLUMNS = [
    "pass_score_version",
    "post_turnover_pass_context",
    "nearest_turnover_end_frame",
    "nearest_turnover_frame_distance",
    "recovery_to_pass_context",
    "recovery_turnover_unified_event_id",
    "recovery_turnover_end_frame",
    "recovery_turnover_team_match",
    "recovery_pass_frames_after_turnover",
    "recovery_flicker_penalty_suppressed",
    "recovery_turnover_penalty_suppressed",
    "required_receiver_stable_frames_for_calibrated_eligibility",
    "turnover_penalty_applied",
    "pass_plausibility_score_calibrated",
    "pass_score_tier_calibrated",
    "pass_score_status_calibrated",
    "is_eligible_for_calibrated_pass_weak_label",
    "calibrated_pass_score_reasons",
]

SUMMARY_COLUMNS = [
    "clip_id",
    "status",
    "refined_pass_candidates",
    "high_score_candidates",
    "medium_score_candidates",
    "low_score_candidates",
    "score_weak_label_candidates",
    "average_pass_plausibility_score",
    "same_team_confirmed_count",
    "sender_receiver_confirmed_count",
    "flicker_warning_count",
    "high_interpolation_count",
    "carry_overlap_penalty_count",
    "turnover_overlap_penalty_count",
    "strict_score_weak_label_candidates",
    "calibrated_score_weak_label_candidates",
    "post_turnover_pass_context_count",
    "turnover_penalty_suppressed_count",
    "calibrated_high_score_candidates",
    "calibrated_medium_score_candidates",
    "calibrated_low_score_candidates",
    "recovery_to_pass_context_count",
    "recovery_turnover_penalty_suppressed_count",
    "recovery_flicker_penalty_suppressed_count",
    "recovery_to_pass_calibrated_weak_label_candidates",
    "output_path",
    "error_message",
]

TEAM_LABELS = {"Team A", "Team B"}
PASS_SCORE_VERSION = "calibrated_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score refined pass candidates with interpretable heuristic evidence.")
    parser.add_argument("--derived-root", default=str(Path("temporal_module") / "data" / "derived"))
    parser.add_argument("--sender-context-frames", type=int, default=10)
    parser.add_argument("--receiver-context-frames", type=int, default=10)
    parser.add_argument("--minimum-stable-frames", type=int, default=4)
    parser.add_argument("--free-ball-distance-threshold-px", type=float, default=35.0)
    parser.add_argument("--maximum-ball-missing-fraction", type=float, default=0.25)
    parser.add_argument("--maximum-ball-interpolated-fraction", type=float, default=0.25)
    parser.add_argument("--max-plausible-pass-duration-frames", type=int, default=24)
    parser.add_argument("--post-turnover-context-frames", type=int, default=5)
    parser.add_argument("--recovery-to-pass-overlap-tolerance-frames", type=int, default=3)
    parser.add_argument("--recovery-to-pass-min-receiver-stable-frames", type=int, default=2)
    parser.add_argument("--recovery-to-pass-min-frames-after-turnover", type=int, default=2)
    return parser.parse_args()


def enforce_derived_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = (PROJECT_ROOT / "temporal_module" / "data" / "derived").resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"Scoring outputs must be under {allowed}: {resolved}") from error
    reject_outputs_path(resolved)
    return resolved


def eligible_clip_ids(derived_root: Path) -> list[str]:
    if not derived_root.exists():
        raise FileNotFoundError(f"Derived root not found: {derived_root}")
    return sorted(path.name for path in derived_root.iterdir() if path.is_dir())


def read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def require_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    return pd.read_csv(path)


def normalize_frames(frames: pd.DataFrame) -> pd.DataFrame:
    if "frame" not in frames.columns:
        raise ValueError("temporal_frames.csv missing frame column")
    result = frames.copy()
    for column in result.columns:
        if column != "clip_id" and column not in team_columns(result):
            result[column] = robust_numeric_conversion(result[column])
    result["frame"] = pd.to_numeric(result["frame"], errors="coerce")
    result = result.dropna(subset=["frame"]).copy()
    result["frame"] = result["frame"].astype(int)
    return result.sort_values("frame").reset_index(drop=True)


def team_columns(df: pd.DataFrame) -> set[str]:
    return {column for column in df.columns if column.endswith("_team") or column in {"possession_team", "team"}}


def robust_numeric_conversion(series: pd.Series) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    non_empty = series.notna() & (series.astype(str).str.strip() != "")
    if not non_empty.any():
        return converted
    if converted[non_empty].notna().any():
        return converted
    return series


def get_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    by_lower = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def get_value(row: pd.Series, column: str, default: Any = "") -> Any:
    if column not in row.index:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return value


def to_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value) or str(value).strip() == "":
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def to_int(value: Any, default: int | None = 0) -> int | None:
    result = to_float(value, np.nan)
    if not np.isfinite(result):
        return default
    return int(round(result))


def owner_tuple(row: pd.Series, possessor_col: str | None, team_col: str | None) -> tuple[str, int] | None:
    if possessor_col is None or team_col is None:
        return None
    team = str(get_value(row, team_col, "")).strip()
    player_id = to_int(get_value(row, possessor_col, ""), None)
    if team not in TEAM_LABELS or player_id is None or player_id < 0:
        return None
    return team, player_id


def infer_sender_receiver(
    frames: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    possessor_col = get_col(frames, ["possessor_track_id", "nearest_player_id", "player_id"])
    team_col = get_col(frames, ["possession_team", "team"])
    distance_col = get_col(frames, ["distance_to_ball", "nearest_distance"])
    before = frames[(frames["frame"] >= start_frame - args.sender_context_frames) & (frames["frame"] < start_frame)].copy()
    after = frames[(frames["frame"] > end_frame) & (frames["frame"] <= end_frame + args.receiver_context_frames)].copy()
    sender_owner, sender_stable, sender_row = trailing_stable_owner(before, possessor_col, team_col)
    receiver_owner, receiver_stable, receiver_row = leading_stable_owner(after, possessor_col, team_col)
    same_team = ""
    if sender_owner is not None and receiver_owner is not None:
        same_team = 1 if sender_owner[0] == receiver_owner[0] else 0
    return {
        "sender_owner": sender_owner,
        "receiver_owner": receiver_owner,
        "sender_stable_frames_before": sender_stable,
        "receiver_stable_frames_after": receiver_stable,
        "same_team_sender_receiver": same_team,
        "sender_ball_distance_before": row_numeric(sender_row, distance_col),
        "receiver_ball_distance_after": row_numeric(receiver_row, distance_col),
    }


def trailing_stable_owner(df: pd.DataFrame, possessor_col: str | None, team_col: str | None) -> tuple[tuple[str, int] | None, int, pd.Series | None]:
    if df.empty:
        return None, 0, None
    ordered = df.sort_values("frame", ascending=False)
    first_row = ordered.iloc[0]
    owner = owner_tuple(first_row, possessor_col, team_col)
    if owner is None:
        return None, 0, first_row
    count = 0
    for _, row in ordered.iterrows():
        if owner_tuple(row, possessor_col, team_col) == owner:
            count += 1
        else:
            break
    return owner, count, first_row


def leading_stable_owner(df: pd.DataFrame, possessor_col: str | None, team_col: str | None) -> tuple[tuple[str, int] | None, int, pd.Series | None]:
    if df.empty:
        return None, 0, None
    ordered = df.sort_values("frame")
    first_row = ordered.iloc[0]
    owner = owner_tuple(first_row, possessor_col, team_col)
    if owner is None:
        return None, 0, first_row
    count = 0
    for _, row in ordered.iterrows():
        if owner_tuple(row, possessor_col, team_col) == owner:
            count += 1
        else:
            break
    return owner, count, first_row


def row_numeric(row: pd.Series | None, column: str | None) -> float:
    if row is None or column is None:
        return np.nan
    return to_float(get_value(row, column, np.nan))


def candidate_motion_features(frames: pd.DataFrame, start_frame: int, end_frame: int, args: argparse.Namespace) -> dict[str, Any]:
    span = frames[(frames["frame"] >= start_frame) & (frames["frame"] <= end_frame)].copy()
    before = frames[(frames["frame"] >= start_frame - args.sender_context_frames) & (frames["frame"] < start_frame)].copy()
    speed_col = get_col(frames, ["ball_speed"])
    accel_col = get_col(frames, ["ball_acceleration"])
    ball_x_col = get_col(frames, ["ball_x"])
    ball_y_col = get_col(frames, ["ball_y"])
    missing_col = get_col(frames, ["ball_missing"])
    interpolated_col = get_col(frames, ["ball_is_interpolated", "is_interpolated"])
    distance_col = get_col(frames, ["distance_to_ball", "nearest_distance"])
    possession_changed_col = get_col(frames, ["possession_changed"])

    speed = numeric_series(span, speed_col)
    accel = numeric_series(span, accel_col)
    before_speed = numeric_series(before, speed_col)
    missing = numeric_series(span, missing_col)
    interpolated = numeric_series(span, interpolated_col)
    distance = numeric_series(span, distance_col)

    free_ball = pd.Series(dtype=float)
    if len(distance):
        free_ball = distance[distance > args.free_ball_distance_threshold_px]
    possession_change_count = int((numeric_series(span, possession_changed_col) == 1).sum()) if possession_changed_col else 0
    ball_distance = np.nan
    if ball_x_col and ball_y_col and len(span) >= 2:
        clean = span.dropna(subset=[ball_x_col, ball_y_col])
        if len(clean) >= 2:
            dx = to_float(clean.iloc[-1][ball_x_col]) - to_float(clean.iloc[0][ball_x_col])
            dy = to_float(clean.iloc[-1][ball_y_col]) - to_float(clean.iloc[0][ball_y_col])
            ball_distance = float(np.hypot(dx, dy))

    baseline = float(before_speed.median()) if len(before_speed.dropna()) else np.nan
    speed_peak = max_or_nan(speed)
    if np.isfinite(speed_peak) and np.isfinite(baseline):
        speed_above_baseline: int | None = int(speed_peak > baseline)
    else:
        speed_above_baseline = None
    return {
        "ball_distance_during_candidate": ball_distance,
        "ball_speed_mean": mean_or_nan(speed),
        "ball_speed_peak": speed_peak,
        "ball_acceleration_peak": max_or_nan(accel),
        "ball_speed_local_baseline": baseline,
        "ball_speed_above_baseline": speed_above_baseline,
        "ball_missing_fraction": fraction_true(missing == 1, len(span), default=np.nan if missing_col is None else 0.0),
        "ball_interpolated_fraction": fraction_true(interpolated == 1, len(span), default=np.nan if interpolated_col is None else 0.0),
        "free_ball_frames": int(len(free_ball)),
        "free_ball_fraction": float(len(free_ball) / len(span)) if len(span) else np.nan,
        "possession_change_count_in_candidate": possession_change_count,
        "possession_flicker_warning": int(possession_change_count > 1),
    }


def numeric_series(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None or column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def overlap_fraction(unified: pd.DataFrame, event_type: str, start_frame: int, end_frame: int) -> float:
    if unified.empty or not {"event_type", "start_frame", "end_frame"}.issubset(unified.columns):
        return np.nan
    table = unified[unified["event_type"].astype(str) == event_type].copy()
    if table.empty:
        return 0.0
    denom = max(1, end_frame - start_frame + 1)
    max_overlap = 0
    for _, row in table.iterrows():
        other_start = to_int(get_value(row, "start_frame", ""), None)
        other_end = to_int(get_value(row, "end_frame", ""), None)
        if other_start is None or other_end is None:
            continue
        overlap = max(0, min(end_frame, other_end) - max(start_frame, other_start) + 1)
        max_overlap = max(max_overlap, overlap)
    return float(max_overlap / denom)


def nearest_turnover_context(
    unified: pd.DataFrame,
    start_frame: int,
    features: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    turnovers = preferred_turnover_rows(unified)
    if turnovers.empty:
        return {
            "post_turnover_pass_context": 0,
            "nearest_turnover_end_frame": "",
            "nearest_turnover_frame_distance": "",
        }

    nearest_end_frame: int | None = None
    nearest_distance: int | None = None
    for _, row in turnovers.iterrows():
        turnover_end = to_int(get_value(row, "end_frame", ""), None)
        if turnover_end is None:
            continue
        distance = abs(start_frame - turnover_end)
        if nearest_distance is None or distance < nearest_distance:
            nearest_end_frame = turnover_end
            nearest_distance = distance

    if nearest_end_frame is None or nearest_distance is None:
        return {
            "post_turnover_pass_context": 0,
            "nearest_turnover_end_frame": "",
            "nearest_turnover_frame_distance": "",
        }

    same_team = features.get("same_team_sender_receiver") == 1
    sender_stable = to_float(features.get("sender_stable_frames_before"), 0.0) >= args.minimum_stable_frames
    receiver_stable = to_float(features.get("receiver_stable_frames_after"), 0.0) >= args.minimum_stable_frames
    carry_overlap = to_float(features.get("carry_overlap_fraction"), np.nan)
    missing_fraction = to_float(features.get("ball_missing_fraction"), np.nan)
    interpolated_fraction = to_float(features.get("ball_interpolated_fraction"), np.nan)
    quality_ok = (
        np.isfinite(carry_overlap)
        and carry_overlap <= 0.20
        and np.isfinite(missing_fraction)
        and missing_fraction <= args.maximum_ball_missing_fraction
        and np.isfinite(interpolated_fraction)
        and interpolated_fraction <= args.maximum_ball_interpolated_fraction
    )
    context = (
        nearest_distance <= args.post_turnover_context_frames
        and same_team
        and sender_stable
        and receiver_stable
        and quality_ok
    )
    return {
        "post_turnover_pass_context": int(context),
        "nearest_turnover_end_frame": nearest_end_frame,
        "nearest_turnover_frame_distance": nearest_distance,
    }


def preferred_turnover_rows(unified: pd.DataFrame) -> pd.DataFrame:
    if unified.empty or not {"event_type", "end_frame"}.issubset(unified.columns):
        return pd.DataFrame()
    turnovers = unified[unified["event_type"].astype(str) == "turnover"].copy()
    if turnovers.empty:
        return turnovers
    if "source_event_type" in turnovers.columns:
        refined = turnovers[turnovers["source_event_type"].astype(str) == "turnover_candidate_refined"].copy()
        if not refined.empty:
            if "canonical_event_id" in refined.columns and "unified_event_id" in refined.columns:
                canonical = refined[
                    refined["canonical_event_id"].astype(str) == refined["unified_event_id"].astype(str)
                ].copy()
                if not canonical.empty:
                    return canonical
            return refined
    return turnovers


def canonical_turnover_rows(unified: pd.DataFrame) -> pd.DataFrame:
    required = {"source_event_type", "canonical_event_id", "unified_event_id", "end_frame"}
    if unified.empty or not required.issubset(unified.columns):
        return pd.DataFrame()
    return unified[
        (unified["source_event_type"].astype(str) == "turnover_candidate_refined")
        & (unified["canonical_event_id"].astype(str) == unified["unified_event_id"].astype(str))
    ].copy()


def recovery_to_pass_context(
    unified: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    candidate_team: str,
    features: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    turnovers = canonical_turnover_rows(unified)
    if turnovers.empty:
        return empty_recovery_context()

    best_row: pd.Series | None = None
    best_frames_after: int | None = None
    window_start = start_frame - args.recovery_to_pass_overlap_tolerance_frames
    for _, row in turnovers.iterrows():
        turnover_end = to_int(get_value(row, "end_frame", ""), None)
        if turnover_end is None or turnover_end < window_start or turnover_end > end_frame:
            continue
        frames_after = end_frame - turnover_end
        if best_frames_after is None or frames_after < best_frames_after:
            best_row = row
            best_frames_after = frames_after

    if best_row is None or best_frames_after is None:
        return empty_recovery_context()

    turnover_team = str(get_value(best_row, "team", "")).strip()
    pass_team = str(candidate_team).strip()
    team_match: int | str = ""
    if pass_team and turnover_team:
        team_match = 1 if pass_team == turnover_team else 0

    same_team = features.get("same_team_sender_receiver") == 1
    sender_stable = to_float(features.get("sender_stable_frames_before"), 0.0) >= args.minimum_stable_frames
    receiver_stable = (
        to_float(features.get("receiver_stable_frames_after"), 0.0)
        >= args.recovery_to_pass_min_receiver_stable_frames
    )
    free_ball_ok = to_float(features.get("free_ball_frames"), 0.0) >= 2
    carry_overlap = to_float(features.get("carry_overlap_fraction"), np.nan)
    missing_fraction = to_float(features.get("ball_missing_fraction"), np.nan)
    interpolated_fraction = to_float(features.get("ball_interpolated_fraction"), np.nan)
    quality_ok = (
        np.isfinite(carry_overlap)
        and carry_overlap <= 0.20
        and np.isfinite(missing_fraction)
        and missing_fraction <= args.maximum_ball_missing_fraction
        and np.isfinite(interpolated_fraction)
        and interpolated_fraction <= args.maximum_ball_interpolated_fraction
    )
    context = (
        best_frames_after >= args.recovery_to_pass_min_frames_after_turnover
        and same_team
        and team_match == 1
        and sender_stable
        and receiver_stable
        and free_ball_ok
        and quality_ok
    )
    return {
        "recovery_to_pass_context": int(context),
        "recovery_turnover_unified_event_id": get_value(best_row, "unified_event_id", ""),
        "recovery_turnover_end_frame": to_int(get_value(best_row, "end_frame", ""), None),
        "recovery_turnover_team_match": team_match,
        "recovery_pass_frames_after_turnover": best_frames_after,
    }


def empty_recovery_context() -> dict[str, Any]:
    return {
        "recovery_to_pass_context": 0,
        "recovery_turnover_unified_event_id": "",
        "recovery_turnover_end_frame": "",
        "recovery_turnover_team_match": "",
        "recovery_pass_frames_after_turnover": "",
    }


def player_speed_from_temporal_frames(frames: pd.DataFrame, frame: int, player_id: int | None) -> float:
    if player_id is None:
        return np.nan
    rows = frames[frames["frame"] == frame]
    if rows.empty:
        return np.nan
    row = rows.iloc[0]
    for slot in range(1, 5):
        track_col = f"p{slot}_track_id"
        speed_col = f"p{slot}_speed"
        if track_col in row.index and speed_col in row.index and to_int(row[track_col], None) == player_id:
            return to_float(row[speed_col])
    return np.nan


def opponent_count_from_frame(frames: pd.DataFrame, frame: int) -> float:
    rows = frames[frames["frame"] == frame]
    if rows.empty:
        return np.nan
    row = rows.iloc[0]
    if "defenders_near_ball" in row.index:
        return to_float(row["defenders_near_ball"])
    return np.nan


def score_candidate(features: dict[str, Any], refined_row: pd.Series, args: argparse.Namespace) -> dict[str, Any]:
    features = with_feature_defaults(features)
    score = 0.0
    reasons: list[str] = []
    same_team = features.get("same_team_sender_receiver")
    if same_team == 1:
        score += 0.20
        reasons.append("same_team_sender_receiver_confirmed")
    elif same_team == 0:
        score -= 0.25
        reasons.append("same_team_sender_receiver_false")

    if to_float(features.get("sender_stable_frames_before"), 0.0) >= args.minimum_stable_frames:
        score += 0.15
        reasons.append("sender_stable_before_release")
    if to_float(features.get("receiver_stable_frames_after"), 0.0) >= args.minimum_stable_frames:
        score += 0.15
        reasons.append("receiver_stable_after_receipt")
    if to_float(features.get("free_ball_frames"), 0.0) > 0:
        score += 0.10
        reasons.append("free_ball_phase_detected")
    duration_frames = to_float(features.get("candidate_duration_frames"), np.nan)
    duration_plausible = 2 <= duration_frames <= args.max_plausible_pass_duration_frames
    if duration_plausible:
        score += 0.10
        reasons.append("duration_plausible")
    else:
        if duration_frames > args.max_plausible_pass_duration_frames:
            score -= 0.10
            reasons.append("duration_too_long")
    ball_distance = to_float(features.get("ball_distance_during_candidate"), np.nan)
    if np.isfinite(ball_distance) and ball_distance > 0:
        score += 0.10
        reasons.append("ball_distance_plausible")
    ball_speed_peak = to_float(features.get("ball_speed_peak"), np.nan)
    speed_above_baseline = features.get("ball_speed_above_baseline")
    if speed_above_baseline == 1:
        score += 0.10
        reasons.append("ball_motion_confirmed")
    elif speed_above_baseline is None or speed_above_baseline == "":
        reasons.append("ball_speed_baseline_unavailable")
    elif np.isfinite(ball_speed_peak):
        reasons.append("ball_speed_not_above_baseline")
    missing_fraction = to_float(features.get("ball_missing_fraction"), np.nan)
    interpolated_fraction = to_float(features.get("ball_interpolated_fraction"), np.nan)
    if np.isfinite(missing_fraction) and missing_fraction <= args.maximum_ball_missing_fraction:
        score += 0.05
        reasons.append("ball_missing_fraction_ok")
    elif np.isfinite(missing_fraction):
        score -= 0.10
        reasons.append("high_ball_missing")
    if np.isfinite(interpolated_fraction) and interpolated_fraction <= args.maximum_ball_interpolated_fraction:
        score += 0.05
        reasons.append("ball_interpolation_fraction_ok")
    elif np.isfinite(interpolated_fraction):
        score -= 0.10
        reasons.append("high_ball_interpolation")

    possession_flicker = bool(to_int(features.get("possession_flicker_warning"), 0))
    if possession_flicker:
        score -= 0.20
        reasons.append("possession_flicker_penalty")
    carry_overlap_fraction = to_float(features.get("carry_overlap_fraction"), np.nan)
    turnover_overlap_fraction = to_float(features.get("turnover_overlap_fraction"), np.nan)
    if np.isfinite(carry_overlap_fraction) and carry_overlap_fraction > 0.20:
        score -= 0.15
        reasons.append("carry_overlap_penalty")
    if np.isfinite(turnover_overlap_fraction) and turnover_overlap_fraction > 0.20:
        score -= 0.15
        reasons.append("turnover_overlap_penalty")
    if to_float(features.get("feature_completeness_fraction"), 0.0) < 0.60:
        score -= 0.10
        reasons.append("insufficient_feature_coverage")

    score = float(min(max(score, 0.0), 1.0))
    tier = "high" if score >= 0.70 else "medium" if score >= 0.45 else "low"
    refined_eligible = to_int(get_value(refined_row, "is_eligible_for_weak_label", 0), 0) == 1
    weak_status = (
        score >= 0.70
        and refined_eligible
        and not possession_flicker
        and same_team == 1
        and np.isfinite(missing_fraction)
        and missing_fraction <= args.maximum_ball_missing_fraction
        and np.isfinite(interpolated_fraction)
        and interpolated_fraction <= args.maximum_ball_interpolated_fraction
    )
    if weak_status:
        status = "weak_label_candidate"
    elif score >= 0.45:
        status = "accepted_for_review"
    else:
        status = "rejected_or_insufficient_evidence"
    return {
        "pass_plausibility_score": score,
        "pass_score_tier": tier,
        "pass_score_status": status,
        "pass_score_reasons": ";".join(reasons),
        "is_eligible_for_pass_score_weak_label": int(weak_status),
    }


def calibrate_score_candidate(
    features: dict[str, Any],
    strict_score: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    features = with_feature_defaults(features)
    score = to_float(strict_score.get("pass_plausibility_score"), 0.0)
    reasons = split_reasons(str(strict_score.get("pass_score_reasons", "")))
    turnover_overlap_fraction = to_float(features.get("turnover_overlap_fraction"), np.nan)
    post_turnover_context = to_int(features.get("post_turnover_pass_context"), 0) == 1
    recovery_context = to_int(features.get("recovery_to_pass_context"), 0) == 1
    possession_flicker = bool(to_int(features.get("possession_flicker_warning"), 0))
    turnover_penalty_would_apply = np.isfinite(turnover_overlap_fraction) and turnover_overlap_fraction > 0.20
    turnover_penalty_applied = int(turnover_penalty_would_apply and not post_turnover_context and not recovery_context)
    recovery_turnover_suppressed = int(recovery_context)
    recovery_flicker_suppressed = int(recovery_context)

    if recovery_context:
        if turnover_penalty_would_apply:
            score += 0.15
            reasons = [reason for reason in reasons if reason != "turnover_overlap_penalty"]
        if possession_flicker:
            score += 0.20
            reasons = [reason for reason in reasons if reason != "possession_flicker_penalty"]
        score += 0.05
        reasons.append("recovery_to_pass_context_confirmed")
        reasons.append("recovery_turnover_overlap_penalty_suppressed")
        reasons.append("recovery_possession_flicker_penalty_suppressed")
    elif post_turnover_context:
        if "post_turnover_pass_context_confirmed" not in reasons:
            reasons.append("post_turnover_pass_context_confirmed")
        if turnover_penalty_would_apply:
            score += 0.15
            reasons = [reason for reason in reasons if reason != "turnover_overlap_penalty"]
            reasons.append("turnover_overlap_penalty_suppressed")
        score += 0.05
    elif turnover_penalty_applied and "turnover_overlap_penalty" not in reasons:
        reasons.append("turnover_overlap_penalty")

    score = float(min(max(score, 0.0), 1.0))
    tier = "high" if score >= 0.70 else "medium" if score >= 0.45 else "low"
    if tier == "high":
        reasons.append("calibrated_high_score")
    elif tier == "medium":
        reasons.append("calibrated_medium_score")
    else:
        reasons.append("calibrated_low_score")

    same_team = features.get("same_team_sender_receiver") == 1
    sender_stable = to_float(features.get("sender_stable_frames_before"), 0.0) >= args.minimum_stable_frames
    required_receiver_stable_frames = (
        args.recovery_to_pass_min_receiver_stable_frames if recovery_context else args.minimum_stable_frames
    )
    receiver_stable = (
        to_float(features.get("receiver_stable_frames_after"), 0.0) >= required_receiver_stable_frames
    )
    missing_fraction = to_float(features.get("ball_missing_fraction"), np.nan)
    interpolated_fraction = to_float(features.get("ball_interpolated_fraction"), np.nan)
    carry_overlap_fraction = to_float(features.get("carry_overlap_fraction"), np.nan)
    weak_status = (
        score >= 0.70
        and same_team
        and sender_stable
        and receiver_stable
        and np.isfinite(missing_fraction)
        and missing_fraction <= args.maximum_ball_missing_fraction
        and np.isfinite(interpolated_fraction)
        and interpolated_fraction <= args.maximum_ball_interpolated_fraction
        and np.isfinite(carry_overlap_fraction)
        and carry_overlap_fraction <= 0.20
        and (not possession_flicker or recovery_context)
    )

    if same_team:
        reasons.append("same_team_sender_receiver_confirmed")
    if recovery_context:
        reasons.append("recovery_receiver_stability_threshold_applied")
    if sender_stable and receiver_stable:
        reasons.append("stable_sender_receiver_confirmed")
    if np.isfinite(interpolated_fraction) and interpolated_fraction > args.maximum_ball_interpolated_fraction:
        reasons.append("high_ball_interpolation")
    if np.isfinite(carry_overlap_fraction) and carry_overlap_fraction > 0.20:
        reasons.append("carry_overlap_penalty")
    if possession_flicker and not recovery_context:
        reasons.append("possession_flicker_penalty")
    if to_float(features.get("feature_completeness_fraction"), 0.0) < 0.60:
        reasons.append("insufficient_feature_coverage")

    if weak_status:
        status = "weak_label_candidate_calibrated"
        reasons.append("calibrated_weak_label_eligible")
    elif score >= 0.45:
        status = "accepted_for_review_calibrated"
    else:
        status = "rejected_or_insufficient_evidence_calibrated"

    return {
        "pass_score_version": PASS_SCORE_VERSION,
        "recovery_flicker_penalty_suppressed": recovery_flicker_suppressed,
        "recovery_turnover_penalty_suppressed": recovery_turnover_suppressed,
        "required_receiver_stable_frames_for_calibrated_eligibility": required_receiver_stable_frames,
        "turnover_penalty_applied": turnover_penalty_applied,
        "pass_plausibility_score_calibrated": score,
        "pass_score_tier_calibrated": tier,
        "pass_score_status_calibrated": status,
        "is_eligible_for_calibrated_pass_weak_label": int(weak_status),
        "calibrated_pass_score_reasons": ";".join(unique_preserve_order(reasons)),
    }


def split_reasons(value: str) -> list[str]:
    return [part for part in value.split(";") if part]


def with_feature_defaults(features: dict[str, Any]) -> dict[str, Any]:
    result = dict(features)
    defaults = {
        "candidate_duration_frames": np.nan,
        "candidate_duration_seconds": np.nan,
        "sender_stable_frames_before": 0,
        "receiver_stable_frames_after": 0,
        "same_team_sender_receiver": "",
        "sender_ball_distance_before": np.nan,
        "receiver_ball_distance_after": np.nan,
        "ball_distance_during_candidate": np.nan,
        "ball_speed_mean": np.nan,
        "ball_speed_peak": np.nan,
        "ball_speed_above_baseline": None,
        "ball_acceleration_peak": np.nan,
        "ball_missing_fraction": np.nan,
        "ball_interpolated_fraction": np.nan,
        "free_ball_frames": 0,
        "free_ball_fraction": np.nan,
        "possession_change_count_in_candidate": 0,
        "possession_flicker_warning": 0,
        "carry_overlap_fraction": np.nan,
        "turnover_overlap_fraction": np.nan,
        "shot_overlap_fraction": np.nan,
        "feature_completeness_fraction": 0.0,
        "post_turnover_pass_context": 0,
        "nearest_turnover_end_frame": "",
        "nearest_turnover_frame_distance": "",
        "recovery_to_pass_context": 0,
        "recovery_turnover_unified_event_id": "",
        "recovery_turnover_end_frame": "",
        "recovery_turnover_team_match": "",
        "recovery_pass_frames_after_turnover": "",
        "recovery_flicker_penalty_suppressed": 0,
        "recovery_turnover_penalty_suppressed": 0,
        "required_receiver_stable_frames_for_calibrated_eligibility": 0,
    }
    for key, value in defaults.items():
        if key not in result:
            result[key] = value
    return result


def build_features_for_candidate(
    clip_id: str,
    row: pd.Series,
    frames: pd.DataFrame,
    unified: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    start_frame = to_int(get_value(row, "start_frame", ""), None)
    end_frame = to_int(get_value(row, "end_frame", ""), None)
    center_frame = to_int(get_value(row, "center_frame", ""), None)
    if start_frame is None or end_frame is None:
        raise ValueError("Pass candidate missing start_frame or end_frame")
    if center_frame is None:
        center_frame = int(round((start_frame + end_frame) / 2))
    duration_frames = end_frame - start_frame + 1
    start_ts = to_float(get_value(row, "start_timestamp", np.nan))
    end_ts = to_float(get_value(row, "end_timestamp", np.nan))
    if not np.isfinite(start_ts):
        start_ts = timestamp_for_frame(frames, start_frame)
    if not np.isfinite(end_ts):
        end_ts = timestamp_for_frame(frames, end_frame)
    duration_seconds = end_ts - start_ts if np.isfinite(start_ts) and np.isfinite(end_ts) else np.nan

    owner = infer_sender_receiver(frames, start_frame, end_frame, args)
    motion = candidate_motion_features(frames, start_frame, end_frame, args)
    reasons: list[str] = []
    if unified.empty:
        reasons.append("unified_catalog_unavailable")
    carry_overlap = overlap_fraction(unified, "carry", start_frame, end_frame)
    turnover_overlap = overlap_fraction(unified, "turnover", start_frame, end_frame)
    shot_overlap = overlap_fraction(unified, "shot", start_frame, end_frame)
    sender_id = owner["sender_owner"][1] if owner["sender_owner"] else None
    receiver_id = owner["receiver_owner"][1] if owner["receiver_owner"] else None

    feature_values = {
        "candidate_duration_frames": duration_frames,
        "candidate_duration_seconds": duration_seconds,
        "sender_stable_frames_before": owner["sender_stable_frames_before"],
        "receiver_stable_frames_after": owner["receiver_stable_frames_after"],
        "same_team_sender_receiver": owner["same_team_sender_receiver"],
        "sender_ball_distance_before": owner["sender_ball_distance_before"],
        "receiver_ball_distance_after": owner["receiver_ball_distance_after"],
        "ball_distance_during_candidate": motion["ball_distance_during_candidate"],
        "ball_speed_mean": motion["ball_speed_mean"],
        "ball_speed_peak": motion["ball_speed_peak"],
        "ball_speed_above_baseline": motion.get("ball_speed_above_baseline"),
        "ball_acceleration_peak": motion["ball_acceleration_peak"],
        "ball_missing_fraction": motion["ball_missing_fraction"],
        "ball_interpolated_fraction": motion["ball_interpolated_fraction"],
        "free_ball_frames": motion["free_ball_frames"],
        "free_ball_fraction": motion["free_ball_fraction"],
        "possession_change_count_in_candidate": motion["possession_change_count_in_candidate"],
        "possession_flicker_warning": motion["possession_flicker_warning"],
        "carry_overlap_fraction": carry_overlap,
        "turnover_overlap_fraction": turnover_overlap,
        "shot_overlap_fraction": shot_overlap,
        "nearby_opponent_count_sender": opponent_count_from_frame(frames, start_frame - 1),
        "nearby_opponent_count_receiver": opponent_count_from_frame(frames, end_frame + 1),
        "receiver_approach_speed_when_available": player_speed_from_temporal_frames(frames, end_frame + 1, receiver_id),
        "sender_departure_speed_when_available": player_speed_from_temporal_frames(frames, start_frame - 1, sender_id),
    }
    feature_values.update(nearest_turnover_context(unified, start_frame, feature_values, args))
    feature_values.update(
        recovery_to_pass_context(unified, start_frame, end_frame, get_value(row, "team", ""), feature_values, args)
    )
    feature_values["feature_completeness_fraction"] = completeness_fraction(feature_values)
    evidence_reasons = evidence_reason_list(feature_values)
    reasons.extend(evidence_reasons)
    feature_values["evidence_reasons"] = ";".join(unique_preserve_order(reasons))
    return feature_values


def evidence_reason_list(features: dict[str, Any]) -> list[str]:
    reasons = []
    if features["same_team_sender_receiver"] == 1:
        reasons.append("same_team_sender_receiver_confirmed")
    if features["sender_stable_frames_before"] > 0:
        reasons.append("sender_context_available")
    if features["receiver_stable_frames_after"] > 0:
        reasons.append("receiver_context_available")
    if features["free_ball_frames"] > 0:
        reasons.append("free_ball_phase_detected")
    if np.isfinite(features["ball_speed_peak"]):
        reasons.append("ball_motion_confirmed")
    if features.get("ball_speed_above_baseline") is None or features.get("ball_speed_above_baseline") == "":
        reasons.append("ball_speed_baseline_unavailable")
    if features["possession_flicker_warning"]:
        reasons.append("possession_flicker_warning")
    if np.isfinite(features["carry_overlap_fraction"]) and features["carry_overlap_fraction"] > 0.20:
        reasons.append("carry_overlap_penalty")
    if np.isfinite(features["turnover_overlap_fraction"]) and features["turnover_overlap_fraction"] > 0.20:
        reasons.append("turnover_overlap_penalty")
    if np.isfinite(features["ball_interpolated_fraction"]) and features["ball_interpolated_fraction"] > 0.25:
        reasons.append("high_ball_interpolation")
    return reasons


def completeness_fraction(features: dict[str, Any]) -> float:
    considered = [key for key in FEATURE_COLUMNS if key not in {"feature_completeness_fraction", "evidence_reasons"}]
    available = 0
    for key in considered:
        value = features.get(key, "")
        if value != "" and not (isinstance(value, float) and not np.isfinite(value)):
            available += 1
    return float(available / len(considered)) if considered else 0.0


def timestamp_for_frame(frames: pd.DataFrame, frame: int) -> float:
    if "timestamp" not in frames.columns or frames.empty:
        return np.nan
    idx = (frames["frame"] - frame).abs().idxmin()
    return to_float(frames.loc[idx, "timestamp"])


def process_clip(clip_id: str, derived_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    clip_dir = derived_root / clip_id
    events_dir = clip_dir / "events"
    refined_path = events_dir / "pass_candidates_refined.csv"
    frames_path = clip_dir / "temporal_frames.csv"
    movement_path = clip_dir / "model_features" / "player_movement_temporal_features.csv"
    shape_path = clip_dir / "model_features" / "shape_temporal_features.csv"
    unified_path = events_dir / "event_candidates_unified.csv"

    refined = require_csv(refined_path, "pass candidates refined")
    frames = normalize_frames(require_csv(frames_path, "temporal frames"))
    _movement = read_optional_csv(movement_path)
    _shape = read_optional_csv(shape_path)
    unified = read_optional_csv(unified_path)

    feature_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    for _, row in refined.iterrows():
        identity = {column: get_value(row, column, "") for column in IDENTITY_COLUMNS}
        identity["clip_id"] = clip_id
        features = build_features_for_candidate(clip_id, row, frames, unified, args)
        score = score_candidate(features, row, args)
        calibrated = calibrate_score_candidate(features, score, args)
        feature_rows.append({**identity, **features, **calibrated})
        refined_payload = row.to_dict()
        scored_rows.append({**refined_payload, **features, **score, **calibrated})

    quality_features = pd.DataFrame(feature_rows, columns=IDENTITY_COLUMNS + FEATURE_COLUMNS + CALIBRATED_COLUMNS)
    scored = pd.DataFrame(scored_rows)
    scored_columns = list(refined.columns) + [
        column for column in FEATURE_COLUMNS + SCORE_COLUMNS + CALIBRATED_COLUMNS if column not in refined.columns
    ]
    for column in scored_columns:
        if column not in scored.columns:
            scored[column] = ""
    scored = scored[scored_columns]

    quality_path = ensure_output_parent(events_dir / "pass_candidate_quality_features.csv")
    scored_path = ensure_output_parent(events_dir / "pass_candidates_scored.csv")
    schema_path = ensure_output_parent(events_dir / "pass_candidate_scoring_schema.json")
    quality_features.to_csv(quality_path, index=False)
    scored.to_csv(scored_path, index=False)
    write_schema(schema_path, args, {
        "pass_candidates_refined": str(refined_path),
        "temporal_frames": str(frames_path),
        "player_movement_temporal_features": str(movement_path) if movement_path.exists() else "",
        "shape_temporal_features": str(shape_path) if shape_path.exists() else "",
        "event_candidates_unified": str(unified_path) if unified_path.exists() else "",
    })
    return build_summary_row(clip_id, scored, scored_path)


def build_summary_row(clip_id: str, scored: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    score = pd.to_numeric(scored.get("pass_plausibility_score", pd.Series(dtype=float)), errors="coerce")
    strict_weak = (
        int(pd.to_numeric(scored.get("is_eligible_for_pass_score_weak_label", 0), errors="coerce").fillna(0).sum())
        if not scored.empty
        else 0
    )
    return {
        "clip_id": clip_id,
        "status": "success",
        "refined_pass_candidates": int(len(scored)),
        "high_score_candidates": int((scored.get("pass_score_tier", "") == "high").sum()) if not scored.empty else 0,
        "medium_score_candidates": int((scored.get("pass_score_tier", "") == "medium").sum()) if not scored.empty else 0,
        "low_score_candidates": int((scored.get("pass_score_tier", "") == "low").sum()) if not scored.empty else 0,
        "score_weak_label_candidates": strict_weak,
        "average_pass_plausibility_score": float(score.mean()) if len(score.dropna()) else np.nan,
        "same_team_confirmed_count": int((scored.get("same_team_sender_receiver", "") == 1).sum()) if not scored.empty else 0,
        "sender_receiver_confirmed_count": int(((pd.to_numeric(scored.get("sender_stable_frames_before", 0), errors="coerce").fillna(0) > 0) & (pd.to_numeric(scored.get("receiver_stable_frames_after", 0), errors="coerce").fillna(0) > 0)).sum()) if not scored.empty else 0,
        "flicker_warning_count": int(pd.to_numeric(scored.get("possession_flicker_warning", 0), errors="coerce").fillna(0).sum()) if not scored.empty else 0,
        "high_interpolation_count": int((pd.to_numeric(scored.get("ball_interpolated_fraction", 0), errors="coerce").fillna(0) > 0.25).sum()) if not scored.empty else 0,
        "carry_overlap_penalty_count": int((pd.to_numeric(scored.get("carry_overlap_fraction", 0), errors="coerce").fillna(0) > 0.20).sum()) if not scored.empty else 0,
        "turnover_overlap_penalty_count": int((pd.to_numeric(scored.get("turnover_overlap_fraction", 0), errors="coerce").fillna(0) > 0.20).sum()) if not scored.empty else 0,
        "strict_score_weak_label_candidates": strict_weak,
        "calibrated_score_weak_label_candidates": int(pd.to_numeric(scored.get("is_eligible_for_calibrated_pass_weak_label", 0), errors="coerce").fillna(0).sum()) if not scored.empty else 0,
        "post_turnover_pass_context_count": int(pd.to_numeric(scored.get("post_turnover_pass_context", 0), errors="coerce").fillna(0).sum()) if not scored.empty else 0,
        "turnover_penalty_suppressed_count": int(scored.get("calibrated_pass_score_reasons", pd.Series(dtype=str)).astype(str).str.contains("turnover_overlap_penalty_suppressed", regex=False).sum()) if not scored.empty else 0,
        "calibrated_high_score_candidates": int((scored.get("pass_score_tier_calibrated", "") == "high").sum()) if not scored.empty else 0,
        "calibrated_medium_score_candidates": int((scored.get("pass_score_tier_calibrated", "") == "medium").sum()) if not scored.empty else 0,
        "calibrated_low_score_candidates": int((scored.get("pass_score_tier_calibrated", "") == "low").sum()) if not scored.empty else 0,
        "recovery_to_pass_context_count": int(pd.to_numeric(scored.get("recovery_to_pass_context", 0), errors="coerce").fillna(0).sum()) if not scored.empty else 0,
        "recovery_turnover_penalty_suppressed_count": int(pd.to_numeric(scored.get("recovery_turnover_penalty_suppressed", 0), errors="coerce").fillna(0).sum()) if not scored.empty else 0,
        "recovery_flicker_penalty_suppressed_count": int(pd.to_numeric(scored.get("recovery_flicker_penalty_suppressed", 0), errors="coerce").fillna(0).sum()) if not scored.empty else 0,
        "recovery_to_pass_calibrated_weak_label_candidates": int(((pd.to_numeric(scored.get("recovery_to_pass_context", 0), errors="coerce").fillna(0) == 1) & (pd.to_numeric(scored.get("is_eligible_for_calibrated_pass_weak_label", 0), errors="coerce").fillna(0) == 1)).sum()) if not scored.empty else 0,
        "output_path": str(output_path),
        "error_message": "",
    }


def failed_row(clip_id: str, error: Exception) -> dict[str, Any]:
    row = {column: 0 for column in SUMMARY_COLUMNS}
    row["clip_id"] = clip_id
    row["status"] = "failed"
    row["output_path"] = ""
    row["error_message"] = str(error)
    return row


def write_schema(path: Path, args: argparse.Namespace, input_paths: dict[str, str]) -> None:
    payload = {
        "build_timestamp": utc_now_iso(),
        "input_paths": input_paths,
        "output_files": [
            "pass_candidate_quality_features.csv",
            "pass_candidates_scored.csv",
            "pass_candidate_scoring_schema.json",
        ],
        "identity_columns": IDENTITY_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "score_columns": SCORE_COLUMNS,
        "calibrated_columns": CALIBRATED_COLUMNS,
        "cli_settings": vars(args),
        "scoring_rule": {
            "baseline_score_version": "strict_v1",
            "positive_terms": [
                "+0.20 same-team sender and receiver confirmed",
                "+0.15 sender stable before start",
                "+0.15 receiver stable after end",
                "+0.10 free-ball phase",
                "+0.10 plausible duration",
                "+0.10 positive ball distance",
                "+0.10 ball speed above local baseline",
                "+0.05 acceptable missing fraction",
                "+0.05 acceptable interpolated fraction",
            ],
            "negative_terms": [
                "-0.25 same-team sender/receiver false",
                "-0.20 possession flicker",
                "-0.15 carry overlap > 0.20",
                "-0.15 turnover overlap > 0.20",
                "-0.10 duration too long",
                "-0.10 high missing fraction",
                "-0.10 high interpolated fraction",
                "-0.10 feature completeness < 0.60",
            ],
            "tier_thresholds": {"high": ">= 0.70", "medium": "0.45 <= score < 0.70", "low": "< 0.45"},
        },
        "calibrated_scoring_rule": {
            "pass_score_version": PASS_SCORE_VERSION,
            "post_turnover_context_frames": args.post_turnover_context_frames,
            "recovery_to_pass_overlap_tolerance_frames": args.recovery_to_pass_overlap_tolerance_frames,
            "recovery_to_pass_min_receiver_stable_frames": args.recovery_to_pass_min_receiver_stable_frames,
            "recovery_to_pass_min_frames_after_turnover": args.recovery_to_pass_min_frames_after_turnover,
            "post_turnover_context_conditions": [
                "same clip turnover row from event_candidates_unified.csv",
                "prefer canonical turnover_candidate_refined rows when available",
                "abs(candidate start_frame - turnover end_frame) <= post_turnover_context_frames",
                "same_team_sender_receiver == 1",
                "sender_stable_frames_before >= minimum_stable_frames",
                "receiver_stable_frames_after >= minimum_stable_frames",
                "carry_overlap_fraction <= 0.20",
                "ball_missing_fraction <= maximum_ball_missing_fraction",
                "ball_interpolated_fraction <= maximum_ball_interpolated_fraction",
            ],
            "recovery_to_pass_context_conditions": [
                "canonical turnover_candidate_refined row exists in event_candidates_unified.csv",
                "turnover canonical_event_id equals its unified_event_id",
                "turnover end_frame is within candidate_start_frame - recovery_to_pass_overlap_tolerance_frames through candidate_end_frame",
                "candidate_end_frame - turnover_end_frame >= recovery_to_pass_min_frames_after_turnover",
                "same_team_sender_receiver == 1",
                "pass candidate team matches canonical turnover team when both fields are available",
                "sender_stable_frames_before >= minimum_stable_frames",
                "receiver_stable_frames_after >= recovery_to_pass_min_receiver_stable_frames",
                "free_ball_frames >= 2",
                "carry_overlap_fraction <= 0.20",
                "ball_missing_fraction <= maximum_ball_missing_fraction",
                "ball_interpolated_fraction <= maximum_ball_interpolated_fraction",
            ],
            "adjustments": [
                "post-turnover context suppresses the normal turnover_overlap_fraction > 0.20 penalty",
                "+0.05 calibrated score bonus when post_turnover_pass_context == 1",
                "recovery-to-pass context suppresses turnover overlap and possession flicker penalties in calibrated scoring only",
                "+0.05 calibrated score bonus when recovery_to_pass_context == 1",
                "pass_plausibility_score remains the strict baseline score",
                "pass_plausibility_score_calibrated is clamped to [0.0, 1.0]",
            ],
            "eligibility": [
                "is_eligible_for_calibrated_pass_weak_label is separate from is_eligible_for_pass_score_weak_label",
                "calibrated eligibility does not require refined is_eligible_for_weak_label == 1",
                "calibrated eligibility requires calibrated score >= 0.70, same-team sender/receiver, stable sender/receiver, acceptable ball missing/interpolation fractions, carry_overlap_fraction <= 0.20, and no possession flicker warning unless recovery_to_pass_context explicitly suppresses it",
                "required_receiver_stable_frames_for_calibrated_eligibility equals recovery_to_pass_min_receiver_stable_frames only when recovery_to_pass_context == 1; otherwise it equals minimum_stable_frames",
            ],
        },
        "warnings": [
            "This is an interpretable heuristic scoring layer, not a trained model.",
            "Existing refined-candidate weak-label eligibility is preserved and not modified.",
            "is_eligible_for_pass_score_weak_label is separate from refined-candidate eligibility.",
            "is_eligible_for_calibrated_pass_weak_label is a provisional v2 rule based on general evidence patterns, not ground truth.",
            "Scores must be validated later against manually annotated pass / not-pass examples.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def mean_or_nan(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else np.nan


def max_or_nan(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.max()) if len(clean) else np.nan


def fraction_true(mask: pd.Series, denominator: int, default: float = np.nan) -> float:
    if denominator <= 0:
        return default
    return float(mask.sum() / denominator)


def unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def main() -> int:
    args = parse_args()
    try:
        derived_root = enforce_derived_root(Path(args.derived_root))
        rows: list[dict[str, Any]] = []
        for clip_id in eligible_clip_ids(derived_root):
            try:
                row = process_clip(clip_id, derived_root, args)
            except Exception as error:
                row = failed_row(clip_id, error)
            rows.append(row)
            print(
                f"{clip_id}: {row['status']} refined={row['refined_pass_candidates']} "
                f"high={row['high_score_candidates']} medium={row['medium_score_candidates']} "
                f"low={row['low_score_candidates']} score_weak={row['score_weak_label_candidates']} "
                f"error={row['error_message']}"
            )
        summary_path = ensure_output_parent(derived_root / "pass_candidate_scoring_build_summary.csv")
        with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Summary CSV: {summary_path}")
        return 0 if rows and all(row["status"] == "success" for row in rows) else 1
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
