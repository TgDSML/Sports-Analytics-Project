"""Refine raw event candidates for inspection, annotation, and weak-label selection."""

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


REFINEMENT_COLUMNS = [
    "raw_event_id",
    "refined_event_id",
    "refinement_status",
    "refinement_reasons",
    "is_recommended_for_review",
    "is_eligible_for_weak_label",
    "suppressed_by_event_id",
    "merged_raw_event_ids",
]

RAW_FALLBACK_COLUMNS = {
    "pass": [
        "clip_id", "event_id", "event_type", "start_frame", "end_frame", "center_frame",
        "start_timestamp", "end_timestamp", "team", "player_id", "secondary_player_id",
        "confidence", "confidence_tier", "quality_flag", "label_source", "rule_reasons",
        "ball_speed_peak", "ball_acceleration_peak", "ball_distance_px", "possession_before",
        "possession_after", "frame_gap_count", "source_event_id", "source_file",
        "from_player_id", "to_player_id", "same_team_transition", "previous_owner_stable_frames",
        "new_owner_stable_frames", "transition_duration_seconds", "valid_ball_motion_frame_count",
        "ball_missing_or_interpolated_fraction", "matched_weak_pass", "temporal_consistency_ok",
    ],
    "turnover": [
        "clip_id", "event_id", "event_type", "start_frame", "end_frame", "center_frame",
        "start_timestamp", "end_timestamp", "team", "player_id", "secondary_player_id",
        "confidence", "confidence_tier", "quality_flag", "label_source", "rule_reasons",
        "ball_speed_peak", "ball_acceleration_peak", "ball_distance_px", "possession_before",
        "possession_after", "frame_gap_count", "source_event_id", "source_file",
        "turnover_subtype", "switch_frame", "previous_player_id", "winner_player_id",
        "previous_team", "winner_team", "previous_owner_stable_frames", "winner_stable_frames",
        "winner_distance_to_ball", "sustained_team_switch", "matched_interception",
        "interception_frame_delta",
    ],
    "shot": [
        "clip_id", "event_id", "event_type", "start_frame", "end_frame", "center_frame",
        "start_timestamp", "end_timestamp", "team", "player_id", "secondary_player_id",
        "confidence", "confidence_tier", "quality_flag", "label_source", "rule_reasons",
        "ball_speed_peak", "ball_acceleration_peak", "ball_distance_px", "possession_before",
        "possession_after", "frame_gap_count", "source_event_id", "source_file",
        "goal_side_candidate", "attack_direction_assumed", "image_width_inferred",
        "image_height_inferred", "near_goal_region", "start_ball_x", "start_ball_y",
        "end_ball_x", "end_ball_y", "valid_ball_motion_frame_count",
        "ball_interpolated_fraction", "carry_overlap_fraction", "geometry_quality_flag",
    ],
}

SUMMARY_COLUMNS = [
    "clip_id",
    "pass_raw_count",
    "pass_refined_count",
    "pass_merged_count",
    "pass_suppressed_count",
    "pass_reviewable_count",
    "pass_weak_label_eligible_count",
    "turnover_raw_count",
    "turnover_refined_count",
    "turnover_suppressed_count",
    "turnover_reviewable_count",
    "turnover_weak_label_eligible_count",
    "shot_raw_count",
    "shot_refined_count",
    "shot_suppressed_count",
    "shot_reviewable_count",
    "shot_weak_label_eligible_count",
    "status",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine raw pass, turnover, and shot candidate CSVs.")
    parser.add_argument("--derived-root", default=str(Path("temporal_module") / "data" / "derived"))
    parser.add_argument("--pass-merge-gap-frames", type=int, default=12)
    parser.add_argument("--max-merged-pass-span-frames", type=int, default=24)
    parser.add_argument("--min-pass-duration-seconds", type=float, default=0.08)
    parser.add_argument("--min-pass-frame-span", type=int, default=2)
    parser.add_argument("--turnover-context-before-frames", type=int, default=8)
    parser.add_argument("--turnover-context-after-frames", type=int, default=8)
    parser.add_argument("--min-turnover-previous-stable-frames", type=int, default=3)
    parser.add_argument("--min-turnover-winner-stable-frames", type=int, default=2)
    parser.add_argument("--shot-local-peak-window-frames", type=int, default=6)
    parser.add_argument("--min-end-region-fraction", type=float, default=0.20)
    parser.add_argument("--max-ball-interpolated-fraction-for-review", type=float, default=0.25)
    parser.add_argument("--max-carry-overlap-fraction-for-review", type=float, default=0.20)
    return parser.parse_args()


def enforce_derived_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = (PROJECT_ROOT / "temporal_module" / "data" / "derived").resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"Refined outputs must be under {allowed}: {resolved}") from error
    reject_outputs_path(resolved)
    return resolved


def candidate_clip_ids(derived_root: Path) -> list[str]:
    if not derived_root.exists():
        raise FileNotFoundError(f"Derived root not found: {derived_root}")
    clip_ids = []
    for clip_dir in sorted(path for path in derived_root.iterdir() if path.is_dir()):
        events_dir = clip_dir / "events"
        if not events_dir.exists():
            continue
        if any((events_dir / name).exists() for name in ["pass_candidates.csv", "turnover_candidates.csv", "shot_candidates.csv"]):
            clip_ids.append(clip_dir.name)
    return clip_ids


def read_candidate_csv(path: Path, event_kind: str) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=RAW_FALLBACK_COLUMNS[event_kind])


def ordered_output_columns(raw_columns: list[str], extra_columns: list[str] | None = None) -> list[str]:
    columns = list(raw_columns)
    for column in extra_columns or []:
        if column not in columns:
            columns.append(column)
    for column in REFINEMENT_COLUMNS:
        if column not in columns:
            columns.append(column)
    return columns


def numeric(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def scalar(row: pd.Series, column: str, default: Any = "") -> Any:
    return row[column] if column in row.index and not pd.isna(row[column]) else default


def read_temporal_bounds(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing temporal frame table: {path}")
    frames = pd.read_csv(path)
    if "frame" not in frames.columns:
        raise ValueError(f"{path} missing frame column")
    frame_values = pd.to_numeric(frames["frame"], errors="coerce").dropna().astype(int)
    if frame_values.empty:
        raise ValueError(f"{path} has no valid frame values")
    return {
        "min_frame": int(frame_values.min()),
        "max_frame": int(frame_values.max()),
        "frames": frames,
    }


def pass_group_key(row: pd.Series) -> tuple[str, str]:
    return (str(scalar(row, "clip_id")), str(scalar(row, "team")))


def pass_merge_decision(
    current_group: pd.DataFrame,
    representative: pd.Series,
    current: pd.Series,
    merge_gap: int,
    max_merged_span: int,
) -> tuple[bool, bool]:
    if str(scalar(representative, "team")) != str(scalar(current, "team")):
        return False, False
    previous_end = to_int(scalar(representative, "end_frame", -10**9), -10**9)
    current_start = to_int(scalar(current, "start_frame", 10**9), 10**9)
    if current_start - previous_end > merge_gap:
        return False, False

    speed_prev = to_float(scalar(representative, "ball_speed_peak", np.nan))
    speed_cur = to_float(scalar(current, "ball_speed_peak", np.nan))
    accel_prev = to_float(scalar(representative, "ball_acceleration_peak", np.nan))
    accel_cur = to_float(scalar(current, "ball_acceleration_peak", np.nan))
    similar_speed = both_nan(speed_prev, speed_cur) or nearly_equal(speed_prev, speed_cur)
    similar_accel = both_nan(accel_prev, accel_cur) or nearly_equal(accel_prev, accel_cur)
    reverse_owner = (
        str(scalar(representative, "from_player_id")) == str(scalar(current, "to_player_id"))
        and str(scalar(representative, "to_player_id")) == str(scalar(current, "from_player_id"))
    )
    overlapping = current_start <= previous_end
    existing_merge_conditions = bool(overlapping or similar_speed or similar_accel or reverse_owner)
    if not existing_merge_conditions:
        return False, False

    combined_start = min(
        to_int(current_group["start_frame"].min(), to_int(scalar(current, "start_frame", current_start), current_start)),
        to_int(scalar(current, "start_frame", current_start), current_start),
    )
    combined_end = max(
        to_int(current_group["end_frame"].max(), previous_end),
        to_int(scalar(current, "end_frame", current_start), current_start),
    )
    combined_span = combined_end - combined_start + 1
    if combined_span > max_merged_span:
        return False, True
    return True, False


def refine_pass_candidates(raw: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    output_columns = ordered_output_columns(list(raw.columns))
    if raw.empty:
        return pd.DataFrame(columns=output_columns)

    working = raw.copy()
    for column in ["start_frame", "end_frame", "center_frame", "confidence", "ball_speed_peak", "ball_acceleration_peak", "transition_duration_seconds", "frame_gap_count"]:
        working[column] = numeric(working, column)
    working = working.sort_values(["clip_id", "team", "start_frame", "end_frame"], na_position="last").reset_index(drop=True)

    groups: list[list[int]] = []
    merge_blocked_indices: set[int] = set()
    for _, group in working.groupby(["clip_id", "team"], sort=False, dropna=False):
        current_group: list[int] = []
        representative: pd.Series | None = None
        for index, row in group.iterrows():
            should_merge = False
            merge_blocked = False
            if representative is not None:
                should_merge, merge_blocked = pass_merge_decision(
                    current_group=working.loc[current_group],
                    representative=representative,
                    current=row,
                    merge_gap=args.pass_merge_gap_frames,
                    max_merged_span=args.max_merged_pass_span_frames,
                )
            if should_merge:
                current_group.append(index)
                representative = strongest_row(working.loc[current_group])
            else:
                if merge_blocked:
                    merge_blocked_indices.update(current_group)
                    merge_blocked_indices.add(index)
                if current_group:
                    groups.append(current_group)
                current_group = [index]
                representative = row
        if current_group:
            groups.append(current_group)

    refined_rows: list[pd.Series] = []
    next_refined_id = 1
    for indices in groups:
        members = working.loc[indices].copy()
        representative = strongest_row(members).copy()
        raw_ids = [str(int(value)) if not pd.isna(value) else "" for value in numeric(members, "event_id").tolist()]
        start_frame = int(numeric(members, "start_frame").min())
        end_frame = int(numeric(members, "end_frame").max())
        duration = to_float(scalar(representative, "transition_duration_seconds", np.nan))
        frame_span = end_frame - start_frame
        reasons: list[str] = []
        status = "kept"
        suppressed_by = ""
        if len(members) > 1:
            status = "merged"
            reasons.append("merged_nearby_same_team_candidates")
            representative["start_frame"] = start_frame
            representative["end_frame"] = end_frame
            representative["merged_raw_event_ids"] = ";".join(raw_ids)
        else:
            representative["merged_raw_event_ids"] = ""
        if not np.isfinite(duration):
            duration = duration_from_timestamps(representative)
        if np.isfinite(duration) and duration < args.min_pass_duration_seconds:
            status = "suppressed"
            reasons.append(f"duration_below_{args.min_pass_duration_seconds:.3f}s")
        if frame_span < args.min_pass_frame_span:
            status = "suppressed"
            reasons.append(f"frame_span_below_{args.min_pass_frame_span}")
        if any(index in merge_blocked_indices for index in indices):
            reasons.append("merge_blocked_span_too_long")
        temporal_ok = to_int(scalar(representative, "temporal_consistency_ok", 0), 0) == 1
        no_frame_gaps = to_int(scalar(representative, "frame_gap_count", 0), 0) == 0
        high_tier = str(scalar(representative, "confidence_tier")).lower() == "high"
        duration_ok = not reasons or not any(reason.startswith(("duration_below", "frame_span_below")) for reason in reasons)
        weak_eligible = int(status != "suppressed" and high_tier and temporal_ok and no_frame_gaps and duration_ok)
        reviewable = int(status != "suppressed")
        if status == "kept" and "merge_blocked_span_too_long" in reasons and not weak_eligible:
            status = "review_only"
        if not reasons:
            reasons.append("passes_refinement_checks")
        representative["raw_event_id"] = raw_ids[0] if raw_ids else ""
        representative["refined_event_id"] = next_refined_id
        representative["refinement_status"] = status
        representative["refinement_reasons"] = ";".join(reasons)
        representative["is_recommended_for_review"] = reviewable
        representative["is_eligible_for_weak_label"] = weak_eligible
        representative["suppressed_by_event_id"] = suppressed_by
        refined_rows.append(representative)
        next_refined_id += 1

    result = pd.DataFrame(refined_rows)
    for column in output_columns:
        if column not in result.columns:
            result[column] = ""
    return result[output_columns]


def refine_turnover_candidates(raw: pd.DataFrame, bounds: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    extra_columns = ["raw_start_frame", "raw_end_frame"]
    output_columns = ordered_output_columns(list(raw.columns), extra_columns)
    if raw.empty:
        return pd.DataFrame(columns=output_columns)
    working = raw.copy()
    for column in ["start_frame", "end_frame", "center_frame", "switch_frame", "previous_owner_stable_frames", "winner_stable_frames"]:
        working[column] = numeric(working, column)
    rows = []
    for _, row in working.iterrows():
        result = row.copy()
        switch_frame = to_int(scalar(result, "switch_frame", scalar(result, "center_frame", bounds["min_frame"])), bounds["min_frame"])
        result["raw_start_frame"] = scalar(result, "start_frame", "")
        result["raw_end_frame"] = scalar(result, "end_frame", "")
        result["start_frame"] = max(bounds["min_frame"], switch_frame - args.turnover_context_before_frames)
        result["end_frame"] = min(bounds["max_frame"], switch_frame + args.turnover_context_after_frames)
        result["center_frame"] = switch_frame
        previous_stable = to_int(scalar(result, "previous_owner_stable_frames", 0), 0)
        winner_stable = to_int(scalar(result, "winner_stable_frames", 0), 0)
        matched = to_int(scalar(result, "matched_interception", 0), 0) == 1
        sustained = to_int(scalar(result, "sustained_team_switch", 0), 0) == 1
        tier = str(scalar(result, "confidence_tier")).lower()
        reasons = ["local_window_around_switch_frame"]
        if matched:
            reasons.append("matched_interception_anchor")
        if previous_stable < args.min_turnover_previous_stable_frames:
            reasons.append("previous_owner_not_stable")
        if winner_stable < args.min_turnover_winner_stable_frames:
            reasons.append("winner_not_stable")
        if not sustained:
            reasons.append("team_switch_not_sustained")

        stable_enough = (
            previous_stable >= args.min_turnover_previous_stable_frames
            and winner_stable >= args.min_turnover_winner_stable_frames
            and sustained
        )
        if tier == "high" and matched and stable_enough:
            status = "kept"
            reviewable = 1
            weak_eligible = 1
        elif tier == "medium" and stable_enough:
            status = "review_only"
            reviewable = 1
            weak_eligible = 0
        elif stable_enough:
            status = "review_only" if tier == "low" else "kept"
            reviewable = int(tier != "low")
            weak_eligible = 0
        else:
            status = "suppressed" if tier == "low" else "review_only"
            reviewable = int(status == "review_only")
            weak_eligible = 0
        result["raw_event_id"] = scalar(result, "event_id", "")
        result["refined_event_id"] = len(rows) + 1
        result["refinement_status"] = status
        result["refinement_reasons"] = ";".join(reasons)
        result["is_recommended_for_review"] = reviewable
        result["is_eligible_for_weak_label"] = weak_eligible
        result["suppressed_by_event_id"] = ""
        result["merged_raw_event_ids"] = ""
        rows.append(result)
    result_df = pd.DataFrame(rows)
    for column in output_columns:
        if column not in result_df.columns:
            result_df[column] = ""
    return result_df[output_columns]


def refine_shot_candidates(raw: pd.DataFrame, bounds: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    output_columns = ordered_output_columns(list(raw.columns))
    if raw.empty:
        return pd.DataFrame(columns=output_columns)
    frames = bounds["frames"].copy()
    if "frame" in frames.columns:
        frames["frame"] = pd.to_numeric(frames["frame"], errors="coerce")
    if "ball_x" in frames.columns:
        frames["ball_x"] = pd.to_numeric(frames["ball_x"], errors="coerce")
    working = raw.copy()
    for column in ["center_frame", "start_frame", "end_frame", "valid_ball_motion_frame_count", "ball_interpolated_fraction", "carry_overlap_fraction"]:
        working[column] = numeric(working, column)
    rows = []
    for _, row in working.iterrows():
        result = row.copy()
        center_frame = to_int(scalar(result, "center_frame", bounds["min_frame"]), bounds["min_frame"])
        local = frames[
            (frames["frame"] >= center_frame - args.shot_local_peak_window_frames)
            & (frames["frame"] <= center_frame + args.shot_local_peak_window_frames)
        ].copy()
        image_width = to_float(scalar(result, "image_width_inferred", np.nan))
        if not np.isfinite(image_width) or image_width <= 0:
            image_width = float(local["ball_x"].max() + 1) if "ball_x" in local and local["ball_x"].notna().any() else np.nan
        local_near_goal = False
        if np.isfinite(image_width) and "ball_x" in local.columns:
            local_near_goal = bool(
                ((local["ball_x"] <= image_width * args.min_end_region_fraction)
                | (local["ball_x"] >= image_width * (1.0 - args.min_end_region_fraction))).any()
            )
        interpolated_fraction = to_float(scalar(result, "ball_interpolated_fraction", 1.0), 1.0)
        carry_overlap = to_float(scalar(result, "carry_overlap_fraction", 1.0), 1.0)
        valid_motion = to_int(scalar(result, "valid_ball_motion_frame_count", 0), 0) > 0
        span = to_int(scalar(result, "end_frame", center_frame), center_frame) - to_int(scalar(result, "start_frame", center_frame), center_frame)
        reasons = ["annotation_review_candidate_only"]
        if local_near_goal:
            reasons.append("local_near_goal_region")
        else:
            reasons.append("local_near_goal_region_absent")
        if interpolated_fraction > args.max_ball_interpolated_fraction_for_review:
            reasons.append("interpolation_fraction_too_high")
        if carry_overlap > args.max_carry_overlap_fraction_for_review:
            reasons.append("carry_overlap_too_high")
        if not valid_motion:
            reasons.append("valid_ball_motion_absent")
        if span > args.shot_local_peak_window_frames * 6 and not local_near_goal:
            reasons.append("long_candidate_span_without_local_peak_support")
        reviewable = int(
            local_near_goal
            and interpolated_fraction <= args.max_ball_interpolated_fraction_for_review
            and carry_overlap <= args.max_carry_overlap_fraction_for_review
            and valid_motion
        )
        result["raw_event_id"] = scalar(result, "event_id", "")
        result["refined_event_id"] = len(rows) + 1
        result["refinement_status"] = "review_only" if reviewable else "suppressed"
        result["refinement_reasons"] = ";".join(reasons)
        result["is_recommended_for_review"] = reviewable
        result["is_eligible_for_weak_label"] = 0
        result["suppressed_by_event_id"] = ""
        result["merged_raw_event_ids"] = ""
        rows.append(result)
    result_df = pd.DataFrame(rows)
    for column in output_columns:
        if column not in result_df.columns:
            result_df[column] = ""
    return result_df[output_columns]


def strongest_row(group: pd.DataFrame) -> pd.Series:
    ranked = group.copy()
    ranked["_confidence_rank"] = pd.to_numeric(ranked.get("confidence", 0), errors="coerce").fillna(0)
    ranked["_speed_rank"] = pd.to_numeric(ranked.get("ball_speed_peak", 0), errors="coerce").fillna(0)
    ranked = ranked.sort_values(["_confidence_rank", "_speed_rank"], ascending=[False, False])
    return ranked.iloc[0].drop(labels=["_confidence_rank", "_speed_rank"], errors="ignore")


def nearly_equal(a: float, b: float) -> bool:
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= scale * 0.05


def both_nan(a: float, b: float) -> bool:
    return not np.isfinite(a) and not np.isfinite(b)


def to_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        if str(value).strip() == "":
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def to_int(value: Any, default: int = 0) -> int:
    result = to_float(value, float(default))
    if not np.isfinite(result):
        return int(default)
    return int(result)


def duration_from_timestamps(row: pd.Series) -> float:
    start = scalar(row, "start_timestamp", np.nan)
    end = scalar(row, "end_timestamp", np.nan)
    try:
        return float(end) - float(start)
    except (TypeError, ValueError):
        return np.nan


def counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {"merged": 0, "suppressed": 0, "reviewable": 0, "weak": 0}
    return {
        "merged": int((df["refinement_status"] == "merged").sum()) if "refinement_status" in df else 0,
        "suppressed": int((df["refinement_status"] == "suppressed").sum()) if "refinement_status" in df else 0,
        "reviewable": int(pd.to_numeric(df.get("is_recommended_for_review", 0), errors="coerce").fillna(0).sum()),
        "weak": int(pd.to_numeric(df.get("is_eligible_for_weak_label", 0), errors="coerce").fillna(0).sum()),
    }


def write_csv(path: Path, df: pd.DataFrame) -> None:
    output = ensure_output_parent(path)
    df.to_csv(output, index=False)


def write_schema(path: Path, args: argparse.Namespace, input_paths: dict[str, str]) -> None:
    payload = {
        "build_timestamp": utc_now_iso(),
        "input_paths": input_paths,
        "added_columns": REFINEMENT_COLUMNS,
        "cli_settings": vars(args),
        "refinement_rules": {
            "pass": [
                "Suppress pass candidates below minimum duration or frame-span thresholds.",
                "Merge nearby same-team pass candidates that overlap, occur within merge gap, share similar ball-motion peaks, or look like reverse owner flicker.",
                "Block a proposed pass merge when the combined inclusive frame span would exceed max_merged_pass_span_frames; blocked rows stay separate and receive merge_blocked_span_too_long.",
                "Only high pass candidates may be weak-label eligible, and only when temporal_consistency_ok == 1, frame_gap_count == 0, and duration/span thresholds pass.",
            ],
            "turnover": [
                "Replace raw turnover ranges with local switch-frame windows.",
                "High matched interceptions can be weak-label eligible when stability and sustained-switch checks pass.",
                "Medium turnover candidates are review-only by default.",
            ],
            "shot": [
                "Shots remain annotation-review candidates only and are never weak-label eligible.",
                "Reviewability requires local near-goal evidence, acceptable interpolation and carry-overlap fractions, and valid ball-motion evidence.",
            ],
        },
        "warnings": [
            "Refined candidates are not ground truth.",
            "Shot candidates remain weak image-space heuristics without homography, attack direction, or goal-post detections.",
        ],
    }
    output = ensure_output_parent(path)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def process_clip(clip_id: str, derived_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    clip_dir = derived_root / clip_id
    events_dir = clip_dir / "events"
    temporal_path = clip_dir / "temporal_frames.csv"
    bounds = read_temporal_bounds(temporal_path)

    pass_path = events_dir / "pass_candidates.csv"
    turnover_path = events_dir / "turnover_candidates.csv"
    shot_path = events_dir / "shot_candidates.csv"

    pass_raw = read_candidate_csv(pass_path, "pass")
    turnover_raw = read_candidate_csv(turnover_path, "turnover")
    shot_raw = read_candidate_csv(shot_path, "shot")

    pass_refined = refine_pass_candidates(pass_raw, args)
    turnover_refined = refine_turnover_candidates(turnover_raw, bounds, args)
    shot_refined = refine_shot_candidates(shot_raw, bounds, args)

    write_csv(events_dir / "pass_candidates_refined.csv", pass_refined)
    write_csv(events_dir / "turnover_candidates_refined.csv", turnover_refined)
    write_csv(events_dir / "shot_candidates_refined.csv", shot_refined)
    write_schema(
        events_dir / "refined_event_candidates_schema.json",
        args,
        {
            "temporal_frames": str(temporal_path),
            "pass_candidates": str(pass_path) if pass_path.exists() else "",
            "turnover_candidates": str(turnover_path) if turnover_path.exists() else "",
            "shot_candidates": str(shot_path) if shot_path.exists() else "",
        },
    )

    pass_counts = counts(pass_refined)
    turnover_counts = counts(turnover_refined)
    shot_counts = counts(shot_refined)
    return {
        "clip_id": clip_id,
        "pass_raw_count": int(len(pass_raw)),
        "pass_refined_count": int(len(pass_refined)),
        "pass_merged_count": pass_counts["merged"],
        "pass_suppressed_count": pass_counts["suppressed"],
        "pass_reviewable_count": pass_counts["reviewable"],
        "pass_weak_label_eligible_count": pass_counts["weak"],
        "turnover_raw_count": int(len(turnover_raw)),
        "turnover_refined_count": int(len(turnover_refined)),
        "turnover_suppressed_count": turnover_counts["suppressed"],
        "turnover_reviewable_count": turnover_counts["reviewable"],
        "turnover_weak_label_eligible_count": turnover_counts["weak"],
        "shot_raw_count": int(len(shot_raw)),
        "shot_refined_count": int(len(shot_refined)),
        "shot_suppressed_count": shot_counts["suppressed"],
        "shot_reviewable_count": shot_counts["reviewable"],
        "shot_weak_label_eligible_count": shot_counts["weak"],
        "status": "success",
        "error_message": "",
    }


def failed_row(clip_id: str, error: Exception) -> dict[str, Any]:
    row = {column: 0 for column in SUMMARY_COLUMNS}
    row["clip_id"] = clip_id
    row["status"] = "failed"
    row["error_message"] = str(error)
    return row


def main() -> int:
    args = parse_args()
    try:
        derived_root = enforce_derived_root(Path(args.derived_root))
        rows: list[dict[str, Any]] = []
        for clip_id in candidate_clip_ids(derived_root):
            try:
                row = process_clip(clip_id, derived_root, args)
            except Exception as error:
                row = failed_row(clip_id, error)
            rows.append(row)
            print(
                f"{clip_id}: "
                f"pass raw/refined/merged/suppressed/review/weak="
                f"{row['pass_raw_count']}/{row['pass_refined_count']}/{row['pass_merged_count']}/"
                f"{row['pass_suppressed_count']}/{row['pass_reviewable_count']}/{row['pass_weak_label_eligible_count']} "
                f"turnover raw/refined/suppressed/review/weak="
                f"{row['turnover_raw_count']}/{row['turnover_refined_count']}/{row['turnover_suppressed_count']}/"
                f"{row['turnover_reviewable_count']}/{row['turnover_weak_label_eligible_count']} "
                f"shot raw/refined/suppressed/review/weak="
                f"{row['shot_raw_count']}/{row['shot_refined_count']}/{row['shot_suppressed_count']}/"
                f"{row['shot_reviewable_count']}/{row['shot_weak_label_eligible_count']} "
                f"status={row['status']}"
            )

        summary_path = ensure_output_parent(derived_root / "refined_event_candidates_build_summary.csv")
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
