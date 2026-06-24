"""Build unified per-clip event-candidate catalogs from existing sources."""

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


UNIFIED_COLUMNS = [
    "clip_id",
    "unified_event_id",
    "canonical_event_id",
    "duplicate_of_unified_event_id",
    "event_type",
    "event_family",
    "source_event_type",
    "start_frame",
    "end_frame",
    "center_frame",
    "start_timestamp",
    "end_timestamp",
    "center_timestamp",
    "team",
    "player_id",
    "secondary_player_id",
    "confidence",
    "confidence_tier",
    "quality_flag",
    "label_source",
    "source_file",
    "source_event_id",
    "rule_reasons",
    "refinement_status",
    "refinement_reasons",
    "is_recommended_for_review",
    "is_eligible_for_weak_label",
    "is_gold_label",
    "event_priority",
    "overlap_group_id",
    "overlap_policy",
    "temporal_overlap_warning",
    "ball_speed_peak",
    "ball_acceleration_peak",
    "ball_distance_px",
    "possession_before",
    "possession_after",
    "notes",
]

SUMMARY_COLUMNS = [
    "clip_id",
    "status",
    "carry_rows",
    "passes_weak_rows",
    "pass_refined_rows",
    "interception_source_rows",
    "turnover_refined_rows",
    "shot_refined_rows",
    "unified_rows_written",
    "weak_label_eligible_rows",
    "review_recommended_rows",
    "overlap_group_count",
    "overlapping_rows",
    "duplicate_interception_source_rows",
    "output_path",
    "error_message",
]

EVENT_PRIORITY = {
    "goal": 100,
    "shot": 90,
    "turnover": 80,
    "pass": 70,
    "carry": 60,
    "duel": 50,
    "background": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified event candidate catalogs.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--derived-root", default=str(Path("temporal_module") / "data" / "derived"))
    parser.add_argument("--interception-context-frames", type=int, default=8)
    parser.add_argument("--interception-duplicate-frame-tolerance", type=int, default=2)
    return parser.parse_args()


def enforce_derived_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = (PROJECT_ROOT / "temporal_module" / "data" / "derived").resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"Unified outputs must be under {allowed}: {resolved}") from error
    reject_outputs_path(resolved)
    return resolved


def eligible_clips(derived_root: Path) -> list[str]:
    if not derived_root.exists():
        raise FileNotFoundError(f"Derived root not found: {derived_root}")
    return sorted(path.name for path in derived_root.iterdir() if path.is_dir() and (path / "temporal_frames.csv").exists())


def read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_temporal_frames(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing temporal_frames.csv: {path}")
    frames = pd.read_csv(path)
    if not {"frame", "timestamp"}.issubset(frames.columns):
        raise ValueError(f"{path} must contain frame and timestamp columns")
    frames = frames.copy()
    frames["frame"] = pd.to_numeric(frames["frame"], errors="coerce")
    frames["timestamp"] = pd.to_numeric(frames["timestamp"], errors="coerce")
    frames = frames.dropna(subset=["frame"]).copy()
    frames["frame"] = frames["frame"].astype(int)
    return frames.sort_values("frame").reset_index(drop=True)


def frame_bounds(frames: pd.DataFrame) -> tuple[int, int]:
    if frames.empty:
        raise ValueError("temporal_frames.csv has no valid frame rows")
    return int(frames["frame"].min()), int(frames["frame"].max())


def nearest_timestamp(frames: pd.DataFrame, frame: Any) -> float | str:
    frame_value = to_float(frame)
    if not np.isfinite(frame_value) or frames.empty:
        return ""
    idx = (frames["frame"] - int(round(frame_value))).abs().idxmin()
    timestamp = to_float(frames.loc[idx, "timestamp"])
    return float(timestamp) if np.isfinite(timestamp) else ""


def clamp_frame(value: Any, min_frame: int, max_frame: int) -> int | str:
    numeric = to_float(value)
    if not np.isfinite(numeric):
        return ""
    return int(min(max(int(round(numeric)), min_frame), max_frame))


def midpoint_frame(start_frame: Any, end_frame: Any) -> int | str:
    start = to_float(start_frame)
    end = to_float(end_frame)
    if not np.isfinite(start) or not np.isfinite(end):
        return ""
    return int(round((start + end) / 2.0))


def base_row(
    clip_id: str,
    event_type: str,
    event_family: str,
    source_event_type: str,
    source_file: Path,
    source_event_id: Any,
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "unified_event_id": "",
        "canonical_event_id": "",
        "duplicate_of_unified_event_id": "",
        "event_type": event_type,
        "event_family": event_family,
        "source_event_type": source_event_type,
        "start_frame": "",
        "end_frame": "",
        "center_frame": "",
        "start_timestamp": "",
        "end_timestamp": "",
        "center_timestamp": "",
        "team": "",
        "player_id": "",
        "secondary_player_id": "",
        "confidence": "",
        "confidence_tier": "",
        "quality_flag": "",
        "label_source": "",
        "source_file": str(source_file),
        "source_event_id": clean_value(source_event_id),
        "rule_reasons": "",
        "refinement_status": "",
        "refinement_reasons": "",
        "is_recommended_for_review": 0,
        "is_eligible_for_weak_label": 0,
        "is_gold_label": 0,
        "event_priority": EVENT_PRIORITY.get(event_type, 0),
        "overlap_group_id": "",
        "overlap_policy": "mark_only_no_label_resolution",
        "temporal_overlap_warning": 0,
        "ball_speed_peak": "",
        "ball_acceleration_peak": "",
        "ball_distance_px": "",
        "possession_before": "",
        "possession_after": "",
        "notes": "",
    }


def finalize_row(row: dict[str, Any], frames: pd.DataFrame, min_frame: int, max_frame: int) -> dict[str, Any] | None:
    notes = [str(row["notes"])] if row.get("notes") else []
    start = clamp_frame(row.get("start_frame"), min_frame, max_frame)
    end = clamp_frame(row.get("end_frame"), min_frame, max_frame)
    center = clamp_frame(row.get("center_frame"), min_frame, max_frame)
    if start == "" or end == "":
        notes.append("missing_frame_values_skipped")
        return None
    if center == "":
        center = midpoint_frame(start, end)
    if center == "":
        notes.append("missing_center_frame")
    row["start_frame"] = start
    row["end_frame"] = end
    row["center_frame"] = center
    if row.get("start_timestamp", "") == "":
        row["start_timestamp"] = nearest_timestamp(frames, start)
    if row.get("end_timestamp", "") == "":
        row["end_timestamp"] = nearest_timestamp(frames, end)
    if row.get("center_timestamp", "") == "":
        row["center_timestamp"] = nearest_timestamp(frames, center)
    row["notes"] = ";".join(note for note in notes if note)
    row["confidence"] = clamp_confidence(row.get("confidence"))
    row["confidence_tier"] = normalize_tier(row.get("confidence_tier"))
    return row


def rows_from_carries(clip_id: str, path: Path, df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, source in df.iterrows():
        quality = str(get_value(source, "carry_quality_flag", ""))
        ok = quality == "ok"
        row = base_row(clip_id, "carry", "possession_event", "carry_source", path, get_value(source, "carry_id", ""))
        row.update(
            {
                "start_frame": get_value(source, "start_frame", ""),
                "end_frame": get_value(source, "end_frame", ""),
                "center_frame": midpoint_frame(get_value(source, "start_frame", ""), get_value(source, "end_frame", "")),
                "start_timestamp": get_value(source, "start_time", ""),
                "end_timestamp": get_value(source, "end_time", ""),
                "team": get_value(source, "team", ""),
                "player_id": get_value(source, "player_id", ""),
                "confidence": 0.80 if ok else 0.40,
                "confidence_tier": "high" if ok else "low",
                "quality_flag": quality,
                "label_source": "existing_carry_detector",
                "is_recommended_for_review": 1,
                "is_eligible_for_weak_label": 1 if ok else 0,
                "ball_distance_px": get_value(source, "ball_distance_px", ""),
                "rule_reasons": "existing_carry_source",
            }
        )
        rows.append(row)
    return rows


def rows_from_passes_weak(clip_id: str, path: Path, df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, source in df.iterrows():
        row = base_row(clip_id, "pass", "possession_event", "passes_weak", path, get_value(source, "pass_id", ""))
        row.update(
            {
                "start_frame": get_value(source, "start_frame", ""),
                "end_frame": get_value(source, "end_frame", ""),
                "center_frame": midpoint_frame(get_value(source, "start_frame", ""), get_value(source, "end_frame", "")),
                "start_timestamp": get_value(source, "start_timestamp", ""),
                "end_timestamp": get_value(source, "end_timestamp", ""),
                "team": get_value(source, "team", ""),
                "player_id": get_value(source, "from_player_id", ""),
                "secondary_player_id": get_value(source, "to_player_id", ""),
                "confidence": get_value(source, "confidence", 0.50),
                "confidence_tier": get_value(source, "confidence_tier", "low"),
                "quality_flag": get_value(source, "quality_flag", ""),
                "label_source": "passes_weak_baseline",
                "is_recommended_for_review": 1,
                "is_eligible_for_weak_label": 0,
                "rule_reasons": get_value(source, "rule_reasons", "passes_weak_source"),
            }
        )
        rows.append(row)
    return rows


def rows_from_interceptions(
    clip_id: str,
    path: Path,
    df: pd.DataFrame,
    context_frames: int,
    min_frame: int,
    max_frame: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, source in df.iterrows():
        center = to_int(get_value(source, "frame", ""), None)
        if center is None:
            continue
        row = base_row(clip_id, "turnover", "possession_change", "interception_source", path, get_value(source, "interception_id", ""))
        row.update(
            {
                "start_frame": max(min_frame, center - context_frames),
                "end_frame": min(max_frame, center + context_frames),
                "center_frame": center,
                "center_timestamp": get_value(source, "timestamp", ""),
                "team": get_value(source, "winner_team", ""),
                "player_id": get_value(source, "winner_player_id", ""),
                "secondary_player_id": get_value(source, "previous_player_id", ""),
                "confidence": 0.85,
                "confidence_tier": "high",
                "quality_flag": get_value(source, "quality_flag", ""),
                "label_source": "existing_interception_detector",
                "is_recommended_for_review": 1,
                "is_eligible_for_weak_label": 1,
                "rule_reasons": "existing_interception_source",
                "ball_distance_px": get_value(source, "distance_to_ball", ""),
                "possession_before": join_team_player(get_value(source, "previous_team", ""), get_value(source, "previous_player_id", "")),
                "possession_after": join_team_player(get_value(source, "winner_team", ""), get_value(source, "winner_player_id", "")),
            }
        )
        rows.append(row)
    return rows


def rows_from_refined(clip_id: str, path: Path, df: pd.DataFrame, event_type: str, event_family: str, source_event_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, source in df.iterrows():
        row = base_row(
            clip_id,
            event_type,
            event_family,
            source_event_type,
            path,
            get_value(source, "source_event_id", get_value(source, "raw_event_id", get_value(source, "event_id", ""))),
        )
        row.update(
            {
                "start_frame": get_value(source, "start_frame", ""),
                "end_frame": get_value(source, "end_frame", ""),
                "center_frame": get_value(source, "center_frame", ""),
                "start_timestamp": get_value(source, "start_timestamp", ""),
                "end_timestamp": get_value(source, "end_timestamp", ""),
                "team": get_value(source, "team", ""),
                "player_id": get_value(source, "player_id", ""),
                "secondary_player_id": get_value(source, "secondary_player_id", ""),
                "confidence": get_value(source, "confidence", ""),
                "confidence_tier": get_value(source, "confidence_tier", ""),
                "quality_flag": get_value(source, "quality_flag", ""),
                "label_source": get_value(source, "label_source", ""),
                "rule_reasons": get_value(source, "rule_reasons", ""),
                "refinement_status": get_value(source, "refinement_status", ""),
                "refinement_reasons": get_value(source, "refinement_reasons", ""),
                "is_recommended_for_review": to_int(get_value(source, "is_recommended_for_review", 0), 0),
                "is_eligible_for_weak_label": 0 if event_type == "shot" else to_int(get_value(source, "is_eligible_for_weak_label", 0), 0),
                "ball_speed_peak": get_value(source, "ball_speed_peak", ""),
                "ball_acceleration_peak": get_value(source, "ball_acceleration_peak", ""),
                "ball_distance_px": get_value(source, "ball_distance_px", ""),
                "possession_before": get_value(source, "possession_before", ""),
                "possession_after": get_value(source, "possession_after", ""),
            }
        )
        if event_type == "pass":
            row["player_id"] = get_value(source, "from_player_id", row["player_id"])
            row["secondary_player_id"] = get_value(source, "to_player_id", row["secondary_player_id"])
        if event_type == "turnover":
            row["player_id"] = get_value(source, "winner_player_id", row["player_id"])
            row["secondary_player_id"] = get_value(source, "previous_player_id", row["secondary_player_id"])
            row["_turnover_subtype"] = get_value(source, "turnover_subtype", "")
        rows.append(row)
    return rows


def apply_interception_duplicate_policy(df: pd.DataFrame, frame_tolerance: int) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df, 0
    result = df.copy()
    result["canonical_event_id"] = result["unified_event_id"]
    result["duplicate_of_unified_event_id"] = ""
    source_mask = result["source_event_type"] == "interception_source"
    refined_mask = result["source_event_type"] == "turnover_candidate_refined"
    if "_turnover_subtype" in result.columns:
        refined_mask = refined_mask & (
            result["_turnover_subtype"].astype(str).isin(["", "interception"])
        )

    duplicate_count = 0
    refined = result[refined_mask].copy()
    for source_index, source_row in result[source_mask].iterrows():
        source_center = to_int(source_row.get("center_frame", ""), None)
        if source_center is None or refined.empty:
            continue
        candidates = refined.copy()
        candidates["_center_delta"] = candidates["center_frame"].apply(lambda value: abs(to_int(value, 10**9) - source_center))
        candidates = candidates[candidates["_center_delta"] <= frame_tolerance].sort_values(["_center_delta", "unified_event_id"])
        if candidates.empty:
            continue
        canonical_id = int(candidates.iloc[0]["unified_event_id"])
        result.at[source_index, "is_eligible_for_weak_label"] = 0
        result.at[source_index, "is_recommended_for_review"] = 0
        result.at[source_index, "canonical_event_id"] = canonical_id
        result.at[source_index, "duplicate_of_unified_event_id"] = canonical_id
        result.at[source_index, "notes"] = append_token(
            result.at[source_index, "notes"],
            "duplicate_source_available_for_traceability",
        )
        result.at[source_index, "rule_reasons"] = append_token(
            result.at[source_index, "rule_reasons"],
            "duplicate_of_refined_interception_turnover",
        )
        canonical_index = candidates.iloc[0].name
        result.at[canonical_index, "canonical_event_id"] = canonical_id
        result.at[canonical_index, "duplicate_of_unified_event_id"] = ""
        duplicate_count += 1
    return result, duplicate_count


def assign_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    result["overlap_group_id"] = ""
    result["temporal_overlap_warning"] = 0
    ordered = result.sort_values(["start_frame", "end_frame", "event_priority"], ascending=[True, True, False])
    intervals = [
        (idx, to_int(row["start_frame"], 0), to_int(row["end_frame"], 0))
        for idx, row in ordered.iterrows()
    ]
    parent = {idx: idx for idx, _, _ in intervals}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, (idx_a, start_a, end_a) in enumerate(intervals):
        for idx_b, start_b, end_b in intervals[i + 1:]:
            if start_a <= end_b and start_b <= end_a:
                union(idx_a, idx_b)

    groups: dict[int, list[int]] = {}
    for idx, _, _ in intervals:
        groups.setdefault(find(idx), []).append(idx)
    group_number = 1
    for members in groups.values():
        if len(members) <= 1:
            continue
        group_id = f"overlap_{group_number}"
        group_number += 1
        result.loc[members, "overlap_group_id"] = group_id
        result.loc[members, "temporal_overlap_warning"] = 1
    return result


def build_unified_for_clip(clip_id: str, outputs_root: Path, derived_root: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, int], dict[str, str]]:
    clip_dir = derived_root / clip_id
    events_dir = clip_dir / "events"
    frames_path = clip_dir / "temporal_frames.csv"
    frames = load_temporal_frames(frames_path)
    min_frame, max_frame = frame_bounds(frames)

    paths = {
        "carries": outputs_root / clip_id / "carries" / "carries.csv",
        "passes_weak": clip_dir / "passes_weak.csv",
        "interceptions": outputs_root / clip_id / "interceptions" / "interceptions.csv",
        "pass_refined": events_dir / "pass_candidates_refined.csv",
        "turnover_refined": events_dir / "turnover_candidates_refined.csv",
        "shot_refined": events_dir / "shot_candidates_refined.csv",
    }
    tables = {name: read_optional_csv(path) for name, path in paths.items()}
    rows: list[dict[str, Any]] = []
    rows.extend(rows_from_carries(clip_id, paths["carries"], tables["carries"]))
    rows.extend(rows_from_passes_weak(clip_id, paths["passes_weak"], tables["passes_weak"]))
    rows.extend(rows_from_interceptions(clip_id, paths["interceptions"], tables["interceptions"], args.interception_context_frames, min_frame, max_frame))
    rows.extend(rows_from_refined(clip_id, paths["pass_refined"], tables["pass_refined"], "pass", "possession_event", "pass_candidate_refined"))
    rows.extend(rows_from_refined(clip_id, paths["turnover_refined"], tables["turnover_refined"], "turnover", "possession_change", "turnover_candidate_refined"))
    rows.extend(rows_from_refined(clip_id, paths["shot_refined"], tables["shot_refined"], "shot", "attacking_event", "shot_candidate_refined"))

    finalized = []
    for row in rows:
        finalized_row = finalize_row(row, frames, min_frame, max_frame)
        if finalized_row is not None:
            finalized.append(finalized_row)
    df = pd.DataFrame(finalized)
    for column in UNIFIED_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    duplicate_interception_source_rows = 0
    if not df.empty:
        df = df.sort_values(["start_frame", "end_frame", "event_priority"], ascending=[True, True, False]).reset_index(drop=True)
        df["unified_event_id"] = np.arange(1, len(df) + 1)
        df, duplicate_interception_source_rows = apply_interception_duplicate_policy(
            df,
            args.interception_duplicate_frame_tolerance,
        )
        df = assign_overlaps(df)
        df = df[UNIFIED_COLUMNS]
    counts = {name: int(len(table)) for name, table in tables.items()}
    input_paths = {name: str(path) if path.exists() else "" for name, path in paths.items()}
    input_paths["temporal_frames"] = str(frames_path)
    counts["duplicate_interception_source_rows"] = int(duplicate_interception_source_rows)
    return df, counts, input_paths


def write_schema(path: Path, clip_id: str, input_paths: dict[str, str], settings: dict[str, Any]) -> None:
    payload = {
        "clip_id": clip_id,
        "build_timestamp": utc_now_iso(),
        "input_paths": input_paths,
        "output_columns": UNIFIED_COLUMNS,
        "settings": settings,
        "event_priority": EVENT_PRIORITY,
        "overlap_policy": "Overlapping rows are marked only; no deletion, deduplication, or label resolution is performed.",
        "deduplication_policy": [
            "Refined turnover candidates matched to existing interception sources are preferred for later use.",
            "Original interception source rows are retained for traceability; matched duplicates are no longer review recommended or weak-label eligible.",
            "Matched interception source rows receive canonical_event_id and duplicate_of_unified_event_id pointing to the refined turnover row.",
            "Refined pass candidates and passes_weak rows are both retained for provenance.",
        ],
        "warnings": [
            "Unified candidates are not ground truth.",
            "is_gold_label is 0 for every row.",
            "Final temporal labels are intentionally not created by this script.",
        ],
    }
    output = ensure_output_parent(path)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def process_clip(clip_id: str, outputs_root: Path, derived_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    df, counts, input_paths = build_unified_for_clip(clip_id, outputs_root, derived_root, args)
    output_path = ensure_output_parent(derived_root / clip_id / "events" / "event_candidates_unified.csv")
    df.to_csv(output_path, index=False)
    write_schema(
        output_path.with_name("event_candidates_unified_schema.json"),
        clip_id,
        input_paths,
        {
            "interception_context_frames": args.interception_context_frames,
            "interception_duplicate_frame_tolerance": args.interception_duplicate_frame_tolerance,
        },
    )
    return {
        "clip_id": clip_id,
        "status": "success",
        "carry_rows": counts["carries"],
        "passes_weak_rows": counts["passes_weak"],
        "pass_refined_rows": counts["pass_refined"],
        "interception_source_rows": counts["interceptions"],
        "turnover_refined_rows": counts["turnover_refined"],
        "shot_refined_rows": counts["shot_refined"],
        "unified_rows_written": int(len(df)),
        "weak_label_eligible_rows": int(pd.to_numeric(df.get("is_eligible_for_weak_label", 0), errors="coerce").fillna(0).sum()) if not df.empty else 0,
        "review_recommended_rows": int(pd.to_numeric(df.get("is_recommended_for_review", 0), errors="coerce").fillna(0).sum()) if not df.empty else 0,
        "overlap_group_count": int(df["overlap_group_id"].replace("", np.nan).dropna().nunique()) if not df.empty else 0,
        "overlapping_rows": int(pd.to_numeric(df.get("temporal_overlap_warning", 0), errors="coerce").fillna(0).sum()) if not df.empty else 0,
        "duplicate_interception_source_rows": int(counts["duplicate_interception_source_rows"]),
        "output_path": str(output_path),
        "error_message": "",
    }


def failed_row(clip_id: str, error: Exception) -> dict[str, Any]:
    row: dict[str, Any] = {column: 0 for column in SUMMARY_COLUMNS}
    row["clip_id"] = clip_id
    row["status"] = "failed"
    row["output_path"] = ""
    row["error_message"] = str(error)
    return row


def get_value(row: pd.Series, column: str, default: Any = "") -> Any:
    if column not in row.index:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return value


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return ""
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


def clamp_confidence(value: Any) -> float | str:
    confidence = to_float(value)
    if not np.isfinite(confidence):
        return ""
    return float(min(max(confidence, 0.0), 1.0))


def normalize_tier(value: Any) -> str:
    text = str(value).strip().lower()
    return text if text in {"high", "medium", "low"} else ""


def join_team_player(team: Any, player: Any) -> str:
    if str(team).strip() == "" and str(player).strip() == "":
        return ""
    return f"{team}:{player}"


def append_token(existing: Any, token: str) -> str:
    values = [part for part in str(existing).split(";") if part]
    if token not in values:
        values.append(token)
    return ";".join(values)


def main() -> int:
    args = parse_args()
    try:
        outputs_root = Path(args.outputs_root)
        derived_root = enforce_derived_root(Path(args.derived_root))
        rows: list[dict[str, Any]] = []
        for clip_id in eligible_clips(derived_root):
            try:
                row = process_clip(clip_id, outputs_root, derived_root, args)
            except Exception as error:
                row = failed_row(clip_id, error)
            rows.append(row)
            print(
                f"{clip_id}: {row['status']} unified={row['unified_rows_written']} "
                f"weak={row['weak_label_eligible_rows']} review={row['review_recommended_rows']} "
                f"overlap_groups={row['overlap_group_count']} error={row['error_message']}"
            )
        summary_path = ensure_output_parent(derived_root / "unified_event_candidates_build_summary.csv")
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
