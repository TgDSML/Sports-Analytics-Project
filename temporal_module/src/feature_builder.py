"""Build canonical frame-level temporal features from existing pipeline CSVs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import ensure_output_parent, read_csv_with_header, utc_now_iso, write_json


TEAM_A = "Team A"
TEAM_B = "Team B"
UNKNOWN = "Unknown"


@dataclass
class BuildResult:
    frame_table: pd.DataFrame
    metadata: dict[str, Any]
    output_csv: Path | None = None
    output_schema: Path | None = None


def build_temporal_frames(
    clip_id: str,
    tracks_path: str | Path,
    ball_tracks_path: str | Path,
    player_teams_path: str | Path,
    possession_path: str | Path,
    possession_debug_path: str | Path,
    output_path: str | Path | None = None,
    k_nearest: int = 4,
    defender_radius_px: float = 100.0,
) -> BuildResult:
    inputs = {
        "tracks": Path(tracks_path),
        "ball_tracks": Path(ball_tracks_path),
        "player_teams": Path(player_teams_path),
        "possession": Path(possession_path),
        "possession_debug": Path(possession_debug_path),
    }
    data: dict[str, pd.DataFrame] = {}
    headers: dict[str, list[str]] = {}
    assumptions: list[str] = []

    for name, path in inputs.items():
        df, header = read_csv_with_header(path)
        data[name] = df
        headers[name] = header

    schemas = _detect_schemas(data)
    normalized = _normalize_inputs(data, schemas, assumptions)
    frame_table = _build_frame_table(
        clip_id=clip_id,
        normalized=normalized,
        k_nearest=int(k_nearest),
        defender_radius_px=float(defender_radius_px),
        assumptions=assumptions,
    )

    metadata = _build_metadata(
        clip_id=clip_id,
        inputs=inputs,
        headers=headers,
        data=data,
        frame_table=frame_table,
        k_nearest=int(k_nearest),
        defender_radius_px=float(defender_radius_px),
        assumptions=assumptions,
    )

    output_csv = None
    output_schema = None
    if output_path is not None:
        output_csv = ensure_output_parent(output_path)
        frame_table.to_csv(output_csv, index=False)
        output_schema = output_csv.with_name("temporal_frames_schema.json")
        write_json(output_schema, metadata)

    return BuildResult(frame_table, metadata, output_csv, output_schema)


def _detect_schemas(data: dict[str, pd.DataFrame]) -> dict[str, dict[str, str]]:
    return {
        "tracks": {
            "frame": _find_column(data["tracks"], ["frame", "frame_id", "frame_number"], "tracks frame"),
            "timestamp": _find_column(data["tracks"], ["timestamp", "time", "time_sec"], "tracks timestamp"),
            "track_id": _find_column(data["tracks"], ["track_id", "player_id", "id"], "tracks track ID"),
            "x": _find_column(data["tracks"], ["center_x", "cx", "player_center_x"], "tracks player center x"),
            "y": _find_column(data["tracks"], ["center_y", "cy", "player_center_y"], "tracks player center y"),
        },
        "ball_tracks": {
            "frame": _find_column(data["ball_tracks"], ["frame", "frame_id", "frame_number"], "ball frame"),
            "timestamp": _find_column(data["ball_tracks"], ["timestamp", "time", "time_sec"], "ball timestamp"),
            "track_id": _find_column(data["ball_tracks"], ["track_id", "ball_track_id", "id"], "ball track ID"),
            "x": _find_column(data["ball_tracks"], ["center_x", "ball_center_x", "x"], "ball center x"),
            "y": _find_column(data["ball_tracks"], ["center_y", "ball_center_y", "y"], "ball center y"),
            "confidence": _optional_column(data["ball_tracks"], ["confidence", "ball_confidence"]),
            "is_interpolated": _optional_column(data["ball_tracks"], ["is_interpolated", "interpolated"]),
        },
        "player_teams": {
            "track_id": _find_column(data["player_teams"], ["track_id", "player_id", "id"], "team track ID"),
            "team": _find_column(data["player_teams"], ["team", "nearest_player_team"], "team label"),
        },
        "possession": {
            "frame": _find_column(data["possession"], ["frame", "frame_id", "frame_number"], "possession frame"),
            "timestamp": _find_column(data["possession"], ["timestamp", "time", "time_sec"], "possession timestamp"),
            "possessor": _find_column(
                data["possession"],
                ["nearest_player_id", "possessor_track_id", "player_id"],
                "possession/nearest player",
            ),
            "team": _find_column(data["possession"], ["team", "possession_team"], "possession team"),
            "distance": _optional_column(data["possession"], ["distance_to_ball", "nearest_distance"]),
        },
        "possession_debug": {
            "frame": _find_column(data["possession_debug"], ["frame", "frame_id", "frame_number"], "debug frame"),
            "reason": _optional_column(data["possession_debug"], ["possession_reason", "reason"]),
            "ball_x": _optional_column(data["possession_debug"], ["ball_center_x", "center_x"]),
            "ball_y": _optional_column(data["possession_debug"], ["ball_center_y", "center_y"]),
        },
    }


def _find_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    by_lower = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    raise ValueError(f"Could not locate required {label} column. Header: {list(df.columns)}")


def _optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    by_lower = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    return None


def _normalize_inputs(
    data: dict[str, pd.DataFrame],
    schemas: dict[str, dict[str, str]],
    assumptions: list[str],
) -> dict[str, pd.DataFrame]:
    tracks = pd.DataFrame(
        {
            "frame": _numeric(data["tracks"][schemas["tracks"]["frame"]]),
            "timestamp": _numeric(data["tracks"][schemas["tracks"]["timestamp"]]),
            "track_id": _numeric(data["tracks"][schemas["tracks"]["track_id"]]),
            "x": _numeric(data["tracks"][schemas["tracks"]["x"]]),
            "y": _numeric(data["tracks"][schemas["tracks"]["y"]]),
        }
    ).dropna(subset=["frame", "track_id", "x", "y"])
    tracks["frame"] = tracks["frame"].astype(int)
    tracks["track_id"] = tracks["track_id"].astype(int)

    teams = pd.DataFrame(
        {
            "track_id": _numeric(data["player_teams"][schemas["player_teams"]["track_id"]]),
            "team_raw": data["player_teams"][schemas["player_teams"]["team"]].astype(str),
        }
    ).dropna(subset=["track_id"])
    teams["track_id"] = teams["track_id"].astype(int)
    teams["team"] = teams["team_raw"].map(_normalize_team)
    tracks = tracks.merge(teams[["track_id", "team"]], on="track_id", how="left")
    tracks["team"] = tracks["team"].fillna(UNKNOWN).map(_normalize_team)

    ball_source = data["ball_tracks"]
    ball_schema = schemas["ball_tracks"]
    ball = pd.DataFrame(
        {
            "ball_track_id": _numeric(ball_source[ball_schema["track_id"]]),
            "frame": _numeric(ball_source[ball_schema["frame"]]),
            "timestamp": _numeric(ball_source[ball_schema["timestamp"]]),
            "ball_x": _numeric(ball_source[ball_schema["x"]]),
            "ball_y": _numeric(ball_source[ball_schema["y"]]),
            "ball_confidence": (
                _numeric(ball_source[ball_schema["confidence"]])
                if ball_schema.get("confidence")
                else np.nan
            ),
            "ball_is_interpolated": (
                _numeric(ball_source[ball_schema["is_interpolated"]])
                if ball_schema.get("is_interpolated")
                else 0
            ),
        }
    ).dropna(subset=["frame", "ball_x", "ball_y"])
    ball["frame"] = ball["frame"].astype(int)
    if ball.duplicated("frame").any():
        assumptions.append("Multiple ball rows per frame were reduced to highest confidence, then first row.")
        ball = (
            ball.sort_values(["frame", "ball_confidence"], ascending=[True, False], na_position="last")
            .drop_duplicates("frame", keep="first")
        )

    possession_source = data["possession"]
    possession_schema = schemas["possession"]
    possession = pd.DataFrame(
        {
            "frame": _numeric(possession_source[possession_schema["frame"]]),
            "timestamp": _numeric(possession_source[possession_schema["timestamp"]]),
            "possessor_track_id": _numeric(possession_source[possession_schema["possessor"]]),
            "possession_team": possession_source[possession_schema["team"]].astype(str),
            "distance_to_ball": (
                _numeric(possession_source[possession_schema["distance"]])
                if possession_schema.get("distance")
                else np.nan
            ),
        }
    ).dropna(subset=["frame"])
    possession["frame"] = possession["frame"].astype(int)
    possession["possession_team"] = possession["possession_team"].map(_normalize_possession_team)

    debug_source = data["possession_debug"]
    debug_schema = schemas["possession_debug"]
    debug = pd.DataFrame({"frame": _numeric(debug_source[debug_schema["frame"]])}).dropna(subset=["frame"])
    debug["frame"] = debug["frame"].astype(int)
    debug["possession_reason"] = (
        debug_source[debug_schema["reason"]].astype(str) if debug_schema.get("reason") else ""
    )

    return {"tracks": tracks, "ball": ball, "possession": possession, "debug": debug}


def _build_frame_table(
    clip_id: str,
    normalized: dict[str, pd.DataFrame],
    k_nearest: int,
    defender_radius_px: float,
    assumptions: list[str],
) -> pd.DataFrame:
    tracks = normalized["tracks"]
    ball = normalized["ball"]
    possession = normalized["possession"]
    debug = normalized["debug"]

    frames = sorted(
        set(tracks["frame"].dropna().astype(int))
        | set(ball["frame"].dropna().astype(int))
        | set(possession["frame"].dropna().astype(int))
    )
    rows = []
    ball_by_frame = ball.set_index("frame", drop=False)
    possession_by_frame = possession.set_index("frame", drop=False)
    debug_by_frame = debug.set_index("frame", drop=False)
    tracks_by_frame = {frame: group.copy() for frame, group in tracks.groupby("frame")}

    player_motion = _add_player_motion(tracks, assumptions)
    motion_by_track_frame = {
        (int(row.track_id), int(row.frame)): row
        for row in player_motion.itertuples(index=False)
    }

    for frame in frames:
        frame_tracks = tracks_by_frame.get(frame, tracks.iloc[0:0].copy())
        ball_row = _lookup_row(ball_by_frame, frame)
        possession_row = _lookup_row(possession_by_frame, frame)
        debug_row = _lookup_row(debug_by_frame, frame)

        ball_missing = ball_row is None or pd.isna(ball_row.get("ball_x")) or pd.isna(ball_row.get("ball_y"))
        ball_x = np.nan if ball_missing else float(ball_row["ball_x"])
        ball_y = np.nan if ball_missing else float(ball_row["ball_y"])

        possessor = _int_or_default(possession_row.get("possessor_track_id") if possession_row is not None else np.nan, -1)
        possession_team = (
            _normalize_possession_team(possession_row.get("possession_team"))
            if possession_row is not None
            else UNKNOWN
        )
        possession_missing = int(possession_row is None or possession_team in {UNKNOWN, "None"})

        row = {
            "clip_id": clip_id,
            "frame": int(frame),
            "timestamp": _best_timestamp(frame, ball_row, possession_row, frame_tracks, assumptions),
            "ball_x": ball_x,
            "ball_y": ball_y,
            "ball_confidence": _float_or_nan(ball_row.get("ball_confidence") if ball_row is not None else np.nan),
            "ball_is_interpolated": _int_or_default(ball_row.get("ball_is_interpolated") if ball_row is not None else 0, 0),
            "ball_missing": int(ball_missing),
            "possessor_track_id": possessor,
            "possession_team": possession_team,
            "distance_to_ball": _float_or_nan(possession_row.get("distance_to_ball") if possession_row is not None else np.nan),
            "possession_reason": str(debug_row.get("possession_reason", UNKNOWN)) if debug_row is not None else UNKNOWN,
            "possession_missing": possession_missing,
            "player_count": int(len(frame_tracks)),
            "team_a_player_count": int((frame_tracks["team"] == TEAM_A).sum()) if not frame_tracks.empty else 0,
            "team_b_player_count": int((frame_tracks["team"] == TEAM_B).sum()) if not frame_tracks.empty else 0,
            "unknown_team_player_count": int((~frame_tracks["team"].isin([TEAM_A, TEAM_B])).sum()) if not frame_tracks.empty else 0,
        }

        nearest = _nearest_players(frame_tracks, ball_x, ball_y, k_nearest)
        for idx in range(k_nearest):
            prefix = f"p{idx + 1}"
            if idx < len(nearest):
                player = nearest.iloc[idx]
                row.update(
                    {
                        f"{prefix}_track_id": int(player["track_id"]),
                        f"{prefix}_x": float(player["x"]),
                        f"{prefix}_y": float(player["y"]),
                        f"{prefix}_team": str(player["team"]),
                        f"{prefix}_distance_to_ball": float(player["distance_to_ball"]),
                        f"{prefix}_missing": 0,
                    }
                )
                motion = motion_by_track_frame.get((int(player["track_id"]), int(frame)))
                vx = _float_or_nan(getattr(motion, "vx", np.nan) if motion is not None else np.nan)
                vy = _float_or_nan(getattr(motion, "vy", np.nan) if motion is not None else np.nan)
                velocity_valid = int(not pd.isna(vx) and not pd.isna(vy))
                row.update(
                    {
                        f"{prefix}_vx": vx,
                        f"{prefix}_vy": vy,
                        f"{prefix}_speed": math.hypot(vx, vy) if velocity_valid else np.nan,
                        f"{prefix}_velocity_valid": velocity_valid,
                    }
                )
            else:
                row.update(
                    {
                        f"{prefix}_track_id": -1,
                        f"{prefix}_x": np.nan,
                        f"{prefix}_y": np.nan,
                        f"{prefix}_team": UNKNOWN,
                        f"{prefix}_distance_to_ball": np.nan,
                        f"{prefix}_missing": 1,
                        f"{prefix}_vx": np.nan,
                        f"{prefix}_vy": np.nan,
                        f"{prefix}_speed": np.nan,
                        f"{prefix}_velocity_valid": 0,
                    }
                )

        row.update(_team_shape_features(frame_tracks, TEAM_A, "team_a"))
        row.update(_team_shape_features(frame_tracks, TEAM_B, "team_b"))
        row.update(_possessor_context(frame_tracks, possessor, possession_team, ball_x, ball_y, defender_radius_px))
        rows.append(row)

    result = pd.DataFrame(rows)
    result = _add_ball_motion(result, assumptions)
    result = _add_possession_change_features(result)
    return _ordered_columns(result, k_nearest)


def _nearest_players(frame_tracks: pd.DataFrame, ball_x: float, ball_y: float, k: int) -> pd.DataFrame:
    if frame_tracks.empty or pd.isna(ball_x) or pd.isna(ball_y):
        return frame_tracks.iloc[0:0].copy()
    candidates = frame_tracks.dropna(subset=["x", "y"]).copy()
    candidates["distance_to_ball"] = np.hypot(candidates["x"] - ball_x, candidates["y"] - ball_y)
    return candidates.sort_values(["distance_to_ball", "track_id"]).head(k)


def _team_shape_features(frame_tracks: pd.DataFrame, team: str, prefix: str) -> dict[str, Any]:
    group = frame_tracks[frame_tracks["team"] == team].dropna(subset=["x", "y"])
    if group.empty:
        return {
            f"{prefix}_centroid_x": np.nan,
            f"{prefix}_centroid_y": np.nan,
            f"{prefix}_width": np.nan,
            f"{prefix}_depth": np.nan,
            f"{prefix}_spread": np.nan,
            f"{prefix}_shape_valid": 0,
        }
    centroid_x = float(group["x"].mean())
    centroid_y = float(group["y"].mean())
    spread = float(np.hypot(group["x"] - centroid_x, group["y"] - centroid_y).mean())
    return {
        f"{prefix}_centroid_x": centroid_x,
        f"{prefix}_centroid_y": centroid_y,
        f"{prefix}_width": float(group["x"].max() - group["x"].min()),
        f"{prefix}_depth": float(group["y"].max() - group["y"].min()),
        f"{prefix}_spread": spread,
        f"{prefix}_shape_valid": 1,
    }


def _possessor_context(
    frame_tracks: pd.DataFrame,
    possessor: int,
    possession_team: str,
    ball_x: float,
    ball_y: float,
    defender_radius_px: float,
) -> dict[str, Any]:
    result = {
        "nearest_teammate_distance": np.nan,
        "nearest_opponent_distance": np.nan,
        "defenders_near_ball": np.nan,
    }
    if possessor < 0 or possession_team not in {TEAM_A, TEAM_B} or frame_tracks.empty:
        return result
    possessor_rows = frame_tracks[frame_tracks["track_id"] == possessor]
    if possessor_rows.empty:
        return result
    possessor_row = possessor_rows.iloc[0]
    others = frame_tracks[frame_tracks["track_id"] != possessor].dropna(subset=["x", "y"]).copy()
    if not others.empty:
        others["distance_to_possessor"] = np.hypot(others["x"] - possessor_row["x"], others["y"] - possessor_row["y"])
        teammates = others[others["team"] == possession_team]
        opponents = others[others["team"].isin([TEAM_A, TEAM_B]) & (others["team"] != possession_team)]
        if not teammates.empty:
            result["nearest_teammate_distance"] = float(teammates["distance_to_possessor"].min())
        if not opponents.empty:
            result["nearest_opponent_distance"] = float(opponents["distance_to_possessor"].min())
    if not pd.isna(ball_x) and not pd.isna(ball_y):
        defenders = frame_tracks[
            frame_tracks["team"].isin([TEAM_A, TEAM_B]) & (frame_tracks["team"] != possession_team)
        ].copy()
        if not defenders.empty:
            distance = np.hypot(defenders["x"] - ball_x, defenders["y"] - ball_y)
            result["defenders_near_ball"] = int((distance <= defender_radius_px).sum())
        else:
            result["defenders_near_ball"] = 0
    return result


def _add_player_motion(tracks: pd.DataFrame, assumptions: list[str]) -> pd.DataFrame:
    result = tracks.sort_values(["track_id", "frame"]).copy()
    result["dt"] = result.groupby("track_id")["timestamp"].diff()
    fallback = result["dt"].isna() | (result["dt"] <= 0)
    if fallback.any():
        frame_diff = result.groupby("track_id")["frame"].diff()
        result.loc[fallback, "dt"] = frame_diff.loc[fallback]
        if frame_diff.loc[fallback].notna().any():
            assumptions.append("Player velocity used frame-difference fallback where timestamp deltas were missing/invalid.")
    result["vx"] = result.groupby("track_id")["x"].diff() / result["dt"]
    result["vy"] = result.groupby("track_id")["y"].diff() / result["dt"]
    result.loc[result["dt"].isna() | (result["dt"] <= 0), ["vx", "vy"]] = np.nan
    return result[["frame", "track_id", "vx", "vy"]]


def _add_ball_motion(df: pd.DataFrame, assumptions: list[str]) -> pd.DataFrame:
    result = df.sort_values("frame").copy()
    dt = result["timestamp"].diff()
    fallback = dt.isna() | (dt <= 0)
    if fallback.any():
        frame_diff = result["frame"].diff()
        dt.loc[fallback] = frame_diff.loc[fallback]
        if frame_diff.loc[fallback].notna().any():
            assumptions.append("Ball velocity used frame-difference fallback where timestamp deltas were missing/invalid.")
    result["ball_vx"] = result["ball_x"].diff() / dt
    result["ball_vy"] = result["ball_y"].diff() / dt
    valid_velocity = (~result["ball_vx"].isna()) & (~result["ball_vy"].isna()) & (result["ball_missing"] == 0)
    result["ball_velocity_valid"] = valid_velocity.astype(int)
    result["ball_speed"] = np.hypot(result["ball_vx"], result["ball_vy"])
    result.loc[result["ball_velocity_valid"] == 0, ["ball_vx", "ball_vy", "ball_speed"]] = np.nan

    result["ball_ax"] = result["ball_vx"].diff() / dt
    result["ball_ay"] = result["ball_vy"].diff() / dt
    valid_accel = (~result["ball_ax"].isna()) & (~result["ball_ay"].isna()) & (result["ball_velocity_valid"] == 1)
    result["ball_acceleration_valid"] = valid_accel.astype(int)
    result["ball_acceleration"] = np.hypot(result["ball_ax"], result["ball_ay"])
    result.loc[result["ball_acceleration_valid"] == 0, ["ball_ax", "ball_ay", "ball_acceleration"]] = np.nan
    return result


def _add_possession_change_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.sort_values("frame").copy()
    owner = list(zip(result["possession_team"], result["possessor_track_id"]))
    changed = []
    frames_since = []
    last_owner = None
    last_change_frame = None
    for frame, current_owner in zip(result["frame"], owner):
        is_missing = current_owner[0] in {UNKNOWN, "None"} or int(current_owner[1]) < 0
        if last_owner is None or is_missing:
            is_changed = 0
            if not is_missing:
                last_owner = current_owner
                last_change_frame = int(frame)
        elif current_owner != last_owner:
            is_changed = 1
            last_owner = current_owner
            last_change_frame = int(frame)
        else:
            is_changed = 0
        changed.append(is_changed)
        frames_since.append(0 if last_change_frame is None else int(frame) - last_change_frame)
    result["possession_changed"] = changed
    result["frames_since_possession_change"] = frames_since
    return result


def _ordered_columns(df: pd.DataFrame, k_nearest: int) -> pd.DataFrame:
    base = [
        "clip_id", "frame", "timestamp",
        "ball_x", "ball_y", "ball_confidence", "ball_is_interpolated", "ball_missing",
        "ball_vx", "ball_vy", "ball_speed", "ball_ax", "ball_ay", "ball_acceleration",
        "ball_velocity_valid", "ball_acceleration_valid",
        "possessor_track_id", "possession_team", "distance_to_ball", "possession_reason",
        "possession_missing", "possession_changed", "frames_since_possession_change",
        "player_count", "team_a_player_count", "team_b_player_count", "unknown_team_player_count",
    ]
    nearest_cols = []
    for idx in range(k_nearest):
        prefix = f"p{idx + 1}"
        nearest_cols.extend([
            f"{prefix}_track_id", f"{prefix}_x", f"{prefix}_y", f"{prefix}_team",
            f"{prefix}_distance_to_ball", f"{prefix}_missing",
            f"{prefix}_vx", f"{prefix}_vy", f"{prefix}_speed", f"{prefix}_velocity_valid",
        ])
    tail = [
        "nearest_teammate_distance", "nearest_opponent_distance", "defenders_near_ball",
        "team_a_centroid_x", "team_a_centroid_y", "team_a_width", "team_a_depth",
        "team_a_spread", "team_a_shape_valid",
        "team_b_centroid_x", "team_b_centroid_y", "team_b_width", "team_b_depth",
        "team_b_spread", "team_b_shape_valid",
    ]
    columns = [column for column in base + nearest_cols + tail if column in df.columns]
    return df[columns]


def _build_metadata(
    clip_id: str,
    inputs: dict[str, Path],
    headers: dict[str, list[str]],
    data: dict[str, pd.DataFrame],
    frame_table: pd.DataFrame,
    k_nearest: int,
    defender_radius_px: float,
    assumptions: list[str],
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "build_timestamp": utc_now_iso(),
        "input_paths": {name: str(Path(path)) for name, path in inputs.items()},
        "detected_source_headers": headers,
        "input_row_counts": {name: int(len(df)) for name, df in data.items()},
        "output_row_count": int(len(frame_table)),
        "output_columns": list(frame_table.columns),
        "missing_value_count": {
            column: int(frame_table[column].isna().sum())
            for column in frame_table.columns
        },
        "k_nearest_players": int(k_nearest),
        "defender_radius_px": float(defender_radius_px),
        "assumptions": sorted(set(assumptions)),
    }


def _lookup_row(indexed: pd.DataFrame, frame: int) -> pd.Series | None:
    if frame not in indexed.index:
        return None
    row = indexed.loc[frame]
    if isinstance(row, pd.DataFrame):
        return row.iloc[0]
    return row


def _best_timestamp(frame: int, ball_row, possession_row, frame_tracks: pd.DataFrame, assumptions: list[str]) -> float:
    for row, column in ((possession_row, "timestamp"), (ball_row, "timestamp")):
        if row is not None:
            value = _float_or_nan(row.get(column))
            if not pd.isna(value):
                return value
    if not frame_tracks.empty and "timestamp" in frame_tracks:
        values = pd.to_numeric(frame_tracks["timestamp"], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])
    assumptions.append("Timestamp missing for at least one frame; used frame index as timestamp fallback.")
    return float(frame)


def _normalize_team(value: Any) -> str:
    text = str(value).strip()
    if text == TEAM_A:
        return TEAM_A
    if text == TEAM_B:
        return TEAM_B
    return UNKNOWN


def _normalize_possession_team(value: Any) -> str:
    text = str(value).strip()
    if text == TEAM_A:
        return TEAM_A
    if text == TEAM_B:
        return TEAM_B
    if text == "None":
        return "None"
    return UNKNOWN


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _float_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result


def _int_or_default(value: Any, default: int) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default

