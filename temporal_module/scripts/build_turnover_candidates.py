"""Build confidence-tiered weak turnover and interception candidates."""

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


TEAM_LABELS = {"Team A", "Team B"}
PLAUSIBLE_WINNER_DISTANCE_PX = 100.0
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
TURNOVER_COLUMNS = COMMON_COLUMNS + [
    "turnover_subtype",
    "switch_frame",
    "previous_player_id",
    "winner_player_id",
    "previous_team",
    "winner_team",
    "previous_owner_stable_frames",
    "winner_stable_frames",
    "winner_distance_to_ball",
    "sustained_team_switch",
    "matched_interception",
    "interception_frame_delta",
]
SUMMARY_COLUMNS = [
    "clip_id",
    "status",
    "temporal_frame_rows",
    "existing_interception_rows",
    "candidate_rows",
    "interception_candidate_count",
    "possession_turnover_candidate_count",
    "high_confidence_count",
    "medium_confidence_count",
    "low_confidence_count",
    "output_path",
    "missing_inputs",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weak turnover/interception candidate rows.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--derived-root", default=str(Path("temporal_module") / "data" / "derived"))
    parser.add_argument("--min-previous-owner-frames", type=int, default=3)
    parser.add_argument("--min-winner-frames", type=int, default=2)
    parser.add_argument("--interception-match-frame-tolerance", type=int, default=2)
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


def normalize_temporal_frames(frames: pd.DataFrame) -> pd.DataFrame:
    required = {"frame", "timestamp", "possessor_track_id", "possession_team"}
    missing = required - set(frames.columns)
    if missing:
        raise ValueError(f"temporal_frames.csv missing column(s): {', '.join(sorted(missing))}")
    result = frames.copy()
    for column in [
        "frame",
        "timestamp",
        "possessor_track_id",
        "distance_to_ball",
        "ball_speed",
        "ball_acceleration",
        "ball_velocity_valid",
        "ball_acceleration_valid",
    ]:
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


def motion_context(frames: pd.DataFrame, center_frame: int, lookaround: int) -> dict[str, Any]:
    span = frames[(frames["frame"] >= center_frame - lookaround) & (frames["frame"] <= center_frame + lookaround)].copy()
    frame_values = span["frame"].to_numpy(dtype=int) if not span.empty else np.asarray([], dtype=int)
    return {
        "ball_speed_peak": max_or_nan(span.get("ball_speed", pd.Series(dtype=float))),
        "ball_acceleration_peak": max_or_nan(span.get("ball_acceleration", pd.Series(dtype=float))),
        "ball_distance_px": min_or_nan(span.get("distance_to_ball", pd.Series(dtype=float))),
        "frame_gap_count": int((np.diff(frame_values) != 1).sum()) if len(frame_values) > 1 else 0,
    }


def match_interception(interceptions: pd.DataFrame, switch_frame: int, tolerance: int) -> tuple[bool, str, float]:
    if interceptions.empty or "frame" not in interceptions.columns:
        return False, "", np.nan
    rows = interceptions.copy()
    rows["frame"] = pd.to_numeric(rows["frame"], errors="coerce")
    rows = rows.dropna(subset=["frame"]).copy()
    if rows.empty:
        return False, "", np.nan
    rows["delta"] = (rows["frame"] - switch_frame).abs()
    matched = rows[rows["delta"] <= tolerance].sort_values("delta")
    if matched.empty:
        return False, "", np.nan
    row = matched.iloc[0]
    event_id = row.get("interception_id", "")
    return True, str(int(float(event_id))) if str(event_id) != "" and not pd.isna(event_id) else "", float(row["frame"] - switch_frame)


def build_candidates_for_clip(
    clip_id: str,
    frames: pd.DataFrame,
    interceptions: pd.DataFrame,
    interception_source_file: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    segments = build_owner_segments(frames)
    rows: list[dict[str, Any]] = []
    for previous, current in zip(segments, segments[1:]):
        if previous["team"] == current["team"]:
            continue
        if previous["team"] not in TEAM_LABELS or current["team"] not in TEAM_LABELS:
            continue
        switch_frame = int(current["start_frame"])
        context = motion_context(frames, switch_frame, args.motion_lookaround_frames)
        switch_rows = frames[frames["frame"] == switch_frame]
        winner_distance = min_or_nan(switch_rows.get("distance_to_ball", pd.Series(dtype=float)))
        matched, source_event_id, frame_delta = match_interception(
            interceptions,
            switch_frame,
            args.interception_match_frame_tolerance,
        )
        previous_stable = int(previous["frames"])
        winner_stable = int(current["frames"])
        stable = previous_stable >= args.min_previous_owner_frames and winner_stable >= args.min_winner_frames
        plausible_distance = np.isfinite(winner_distance) and winner_distance <= PLAUSIBLE_WINNER_DISTANCE_PX
        sustained = int(winner_stable >= args.min_winner_frames)
        if stable and plausible_distance and matched:
            tier = "high"
            confidence = 0.85
        elif stable and plausible_distance:
            tier = "medium"
            confidence = 0.60
        else:
            tier = "low"
            confidence = 0.35
        subtype = "interception" if matched else "possession_turnover"
        start_frame = int(previous["end_frame"])
        end_frame = int(current["start_frame"])
        reasons = [
            "team_switch",
            f"previous_stable={previous_stable}",
            f"winner_stable={winner_stable}",
            f"plausible_distance={int(plausible_distance)}",
        ]
        if matched:
            reasons.append("matched_existing_interception")
        rows.append(
            {
                "clip_id": clip_id,
                "event_id": len(rows) + 1,
                "event_type": "turnover_candidate",
                "start_frame": start_frame,
                "end_frame": end_frame,
                "center_frame": switch_frame,
                "start_timestamp": float(previous["end_timestamp"]),
                "end_timestamp": float(current["start_timestamp"]),
                "team": current["team"],
                "player_id": int(current["player_id"]),
                "secondary_player_id": int(previous["player_id"]),
                "confidence": confidence,
                "confidence_tier": tier,
                "quality_flag": "weak_candidate_not_ground_truth",
                "label_source": "possession_team_switch_with_interception_anchor",
                "rule_reasons": ";".join(reasons),
                "ball_speed_peak": context["ball_speed_peak"],
                "ball_acceleration_peak": context["ball_acceleration_peak"],
                "ball_distance_px": context["ball_distance_px"],
                "possession_before": f"{previous['team']}:{previous['player_id']}",
                "possession_after": f"{current['team']}:{current['player_id']}",
                "frame_gap_count": context["frame_gap_count"],
                "source_event_id": source_event_id,
                "source_file": interception_source_file if matched else "",
                "turnover_subtype": subtype,
                "switch_frame": switch_frame,
                "previous_player_id": int(previous["player_id"]),
                "winner_player_id": int(current["player_id"]),
                "previous_team": previous["team"],
                "winner_team": current["team"],
                "previous_owner_stable_frames": previous_stable,
                "winner_stable_frames": winner_stable,
                "winner_distance_to_ball": winner_distance,
                "sustained_team_switch": sustained,
                "matched_interception": int(matched),
                "interception_frame_delta": frame_delta,
            }
        )
    return pd.DataFrame(rows, columns=TURNOVER_COLUMNS)


def write_schema(path: Path, payload: dict[str, Any]) -> None:
    output = ensure_output_parent(path)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def process_clip(clip_id: str, outputs_root: Path, derived_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    frames_path = derived_root / clip_id / "temporal_frames.csv"
    interceptions_path = outputs_root / clip_id / "interceptions" / "interceptions.csv"
    possession_path = outputs_root / clip_id / "possession" / "possession.csv"
    ball_tracks_path = outputs_root / clip_id / "tracks" / "ball_tracks.csv"
    frames = normalize_temporal_frames(pd.read_csv(frames_path))
    interceptions = read_optional_csv(interceptions_path)
    _ = read_optional_csv(possession_path)
    _ = read_optional_csv(ball_tracks_path)
    missing_inputs = ";".join(
        name
        for name, path in [
            ("interceptions", interceptions_path),
            ("possession", possession_path),
            ("ball_tracks", ball_tracks_path),
        ]
        if not path.exists()
    )
    candidates = build_candidates_for_clip(
        clip_id=clip_id,
        frames=frames,
        interceptions=interceptions,
        interception_source_file=str(interceptions_path) if interceptions_path.exists() else "",
        args=args,
    )
    output_path = ensure_output_parent(derived_root / clip_id / "events" / "turnover_candidates.csv")
    candidates.to_csv(output_path, index=False)
    write_schema(
        output_path.with_name("turnover_candidates_schema.json"),
        {
            "clip_id": clip_id,
            "input_paths": {
                "temporal_frames": str(frames_path),
                "interceptions": str(interceptions_path) if interceptions_path.exists() else "",
                "possession": str(possession_path) if possession_path.exists() else "",
                "ball_tracks": str(ball_tracks_path) if ball_tracks_path.exists() else "",
            },
            "output_columns": TURNOVER_COLUMNS,
            "candidate_count": int(len(candidates)),
            "settings": {
                "min_previous_owner_frames": args.min_previous_owner_frames,
                "min_winner_frames": args.min_winner_frames,
                "interception_match_frame_tolerance": args.interception_match_frame_tolerance,
                "motion_lookaround_frames": args.motion_lookaround_frames,
                "plausible_winner_distance_px": PLAUSIBLE_WINNER_DISTANCE_PX,
            },
            "warnings": ["Weak turnover candidates are heuristic candidates, not ground truth."],
        },
    )
    return {
        "clip_id": clip_id,
        "status": "success",
        "temporal_frame_rows": int(len(frames)),
        "existing_interception_rows": int(len(interceptions)),
        "candidate_rows": int(len(candidates)),
        "interception_candidate_count": int((candidates["turnover_subtype"] == "interception").sum()) if not candidates.empty else 0,
        "possession_turnover_candidate_count": int((candidates["turnover_subtype"] == "possession_turnover").sum()) if not candidates.empty else 0,
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
                interceptions_path = outputs_root / clip_id / "interceptions" / "interceptions.csv"
                rows.append(
                    {
                        "clip_id": clip_id,
                        "status": "failed",
                        "temporal_frame_rows": len(pd.read_csv(frames_path)) if frames_path.exists() else 0,
                        "existing_interception_rows": len(pd.read_csv(interceptions_path)) if interceptions_path.exists() else 0,
                        "candidate_rows": 0,
                        "interception_candidate_count": 0,
                        "possession_turnover_candidate_count": 0,
                        "high_confidence_count": 0,
                        "medium_confidence_count": 0,
                        "low_confidence_count": 0,
                        "output_path": "",
                        "missing_inputs": "",
                        "error_message": str(error),
                    }
                )
        summary_path = ensure_output_parent(derived_root / "turnover_candidates_build_summary.csv")
        with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print("Turnover candidate build summary:")
        for row in rows:
            print(f"{row['clip_id']}: {row['status']} candidates={row['candidate_rows']} error={row['error_message']}")
        print(f"Summary CSV: {summary_path}")
        return 0 if rows and all(row["status"] == "success" for row in rows) else 1
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
