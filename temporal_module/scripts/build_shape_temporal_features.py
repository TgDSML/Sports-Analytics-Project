"""Build team shape temporal features and summaries from temporal frame tables."""

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


EPSILON = 1e-6
SHAPE_CHANGE_PERCENTILE = 90.0

REQUIRED_TEMPORAL_COLUMNS = {
    "frame",
    "timestamp",
    "team_a_centroid_x",
    "team_a_centroid_y",
    "team_a_width",
    "team_a_depth",
    "team_a_spread",
    "team_a_shape_valid",
    "team_b_centroid_x",
    "team_b_centroid_y",
    "team_b_width",
    "team_b_depth",
    "team_b_spread",
    "team_b_shape_valid",
}

TEMPORAL_FEATURE_COLUMNS = [
    "clip_id",
    "frame",
    "timestamp",
    "team",
    "centroid_x",
    "centroid_y",
    "width",
    "depth",
    "spread",
    "compactness_proxy",
    "shape_valid",
    "centroid_dx",
    "centroid_dy",
    "centroid_speed_px_per_second",
    "width_delta",
    "depth_delta",
    "spread_delta",
    "rolling_centroid_speed_mean",
    "rolling_width_mean",
    "rolling_depth_mean",
    "rolling_spread_mean",
    "rolling_width_std",
    "rolling_depth_std",
    "rolling_spread_std",
    "shape_change_flag",
    "frame_gap_from_previous",
    "valid_mask",
]

FIVE_SECOND_COLUMNS = [
    "clip_id",
    "team",
    "window_id",
    "start_frame",
    "end_frame",
    "start_timestamp",
    "end_timestamp",
    "total_frame_rows",
    "valid_shape_frames",
    "valid_shape_fraction",
    "mean_centroid_x",
    "mean_centroid_y",
    "mean_width",
    "mean_depth",
    "mean_spread",
    "mean_compactness_proxy",
    "mean_centroid_speed_px_per_second",
    "width_variation_std",
    "depth_variation_std",
    "spread_variation_std",
    "shape_change_frame_count",
    "shape_quality_flag",
]

TEAM_SUMMARY_COLUMNS = [
    "clip_id",
    "team",
    "total_frame_rows",
    "valid_shape_frames",
    "valid_shape_fraction",
    "mean_centroid_x",
    "mean_centroid_y",
    "mean_width",
    "median_width",
    "mean_depth",
    "median_depth",
    "mean_spread",
    "median_spread",
    "mean_compactness_proxy",
    "mean_centroid_speed_px_per_second",
    "width_std",
    "depth_std",
    "spread_std",
    "shape_change_frame_count",
    "shape_quality_flag",
]

BUILD_SUMMARY_COLUMNS = [
    "clip_id",
    "status",
    "input_frame_rows",
    "team_a_valid_fraction",
    "team_b_valid_fraction",
    "temporal_feature_rows_written",
    "summary_rows_written",
    "five_second_rows_written",
    "output_directory",
    "error_message",
]

TEAM_MAPPINGS = {
    "Team A": {
        "centroid_x": "team_a_centroid_x",
        "centroid_y": "team_a_centroid_y",
        "width": "team_a_width",
        "depth": "team_a_depth",
        "spread": "team_a_spread",
        "shape_valid": "team_a_shape_valid",
    },
    "Team B": {
        "centroid_x": "team_b_centroid_x",
        "centroid_y": "team_b_centroid_y",
        "width": "team_b_width",
        "depth": "team_b_depth",
        "spread": "team_b_spread",
        "shape_valid": "team_b_shape_valid",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build temporal team-shape features from temporal_frames.csv.")
    parser.add_argument("--derived-root", default=str(Path("temporal_module") / "data" / "derived"))
    parser.add_argument("--rolling-frames", type=int, default=25)
    parser.add_argument("--window-seconds", type=float, default=5.0)
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


def eligible_clips(derived_root: Path) -> list[tuple[str, Path]]:
    if not derived_root.exists():
        raise FileNotFoundError(f"Derived root not found: {derived_root}")
    return [
        (path.name, path / "temporal_frames.csv")
        for path in sorted(derived_root.iterdir())
        if path.is_dir() and (path / "temporal_frames.csv").exists()
    ]


def require_columns(df: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing column(s): {', '.join(sorted(missing))}")


def long_shape_table(clip_id: str, frames: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = frames.copy()
    base["frame"] = pd.to_numeric(base["frame"], errors="coerce")
    base["timestamp"] = pd.to_numeric(base["timestamp"], errors="coerce")
    base = base.dropna(subset=["frame", "timestamp"]).copy()
    base["frame"] = base["frame"].astype(int)

    for team, mapping in TEAM_MAPPINGS.items():
        part = pd.DataFrame(
            {
                "clip_id": clip_id,
                "frame": base["frame"],
                "timestamp": base["timestamp"],
                "team": team,
                "centroid_x": pd.to_numeric(base[mapping["centroid_x"]], errors="coerce"),
                "centroid_y": pd.to_numeric(base[mapping["centroid_y"]], errors="coerce"),
                "width": pd.to_numeric(base[mapping["width"]], errors="coerce"),
                "depth": pd.to_numeric(base[mapping["depth"]], errors="coerce"),
                "spread": pd.to_numeric(base[mapping["spread"]], errors="coerce"),
                "shape_valid": pd.to_numeric(base[mapping["shape_valid"]], errors="coerce").fillna(0).astype(int),
            }
        )
        rows.append(part)
    table = pd.concat(rows, ignore_index=True)
    table = table.sort_values(["team", "frame", "timestamp"]).reset_index(drop=True)
    valid_base = (
        (table["shape_valid"] == 1)
        & table[["centroid_x", "centroid_y", "width", "depth", "spread"]].notna().all(axis=1)
    )
    table["valid_mask"] = valid_base.astype(int)
    table["compactness_proxy"] = np.where(
        (table["valid_mask"] == 1) & (table["spread"] > EPSILON),
        1.0 / np.maximum(table["spread"], EPSILON),
        np.nan,
    )
    return table


def add_temporal_derivatives(table: pd.DataFrame, rolling_frames: int) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    result_parts = []
    thresholds: dict[str, dict[str, float]] = {}
    for team, group in table.groupby("team", sort=True):
        group = group.sort_values(["frame", "timestamp"]).reset_index(drop=True)
        previous_frame = group["frame"].shift(1)
        previous_timestamp = group["timestamp"].shift(1)
        group["frame_gap_from_previous"] = group["frame"] - previous_frame
        dt = group["timestamp"] - previous_timestamp
        valid_delta = (
            (group["valid_mask"] == 1)
            & (group["valid_mask"].shift(1) == 1)
            & (group["frame_gap_from_previous"] == 1)
            & (dt > 0)
        )

        group["centroid_dx"] = np.where(valid_delta, group["centroid_x"] - group["centroid_x"].shift(1), np.nan)
        group["centroid_dy"] = np.where(valid_delta, group["centroid_y"] - group["centroid_y"].shift(1), np.nan)
        group["centroid_speed_px_per_second"] = np.where(
            valid_delta,
            np.hypot(group["centroid_dx"], group["centroid_dy"]) / dt,
            np.nan,
        )
        group["width_delta"] = np.where(valid_delta, group["width"] - group["width"].shift(1), np.nan)
        group["depth_delta"] = np.where(valid_delta, group["depth"] - group["depth"].shift(1), np.nan)
        group["spread_delta"] = np.where(valid_delta, group["spread"] - group["spread"].shift(1), np.nan)

        threshold_values = {
            "width_delta_abs_threshold": percentile_or_inf(group["width_delta"].abs().dropna(), SHAPE_CHANGE_PERCENTILE),
            "depth_delta_abs_threshold": percentile_or_inf(group["depth_delta"].abs().dropna(), SHAPE_CHANGE_PERCENTILE),
            "spread_delta_abs_threshold": percentile_or_inf(group["spread_delta"].abs().dropna(), SHAPE_CHANGE_PERCENTILE),
        }
        thresholds[team] = threshold_values
        group["shape_change_flag"] = (
            (group["valid_mask"] == 1)
            & (
                (group["width_delta"].abs() > threshold_values["width_delta_abs_threshold"])
                | (group["depth_delta"].abs() > threshold_values["depth_delta_abs_threshold"])
                | (group["spread_delta"].abs() > threshold_values["spread_delta_abs_threshold"])
            )
        ).astype(int)

        sequence_break = ~valid_delta
        sequence_id = sequence_break.cumsum()
        group["rolling_centroid_speed_mean"] = rolling_valid(group, sequence_id, "centroid_speed_px_per_second", rolling_frames, valid_delta)
        for source, mean_name, std_name in [
            ("width", "rolling_width_mean", "rolling_width_std"),
            ("depth", "rolling_depth_mean", "rolling_depth_std"),
            ("spread", "rolling_spread_mean", "rolling_spread_std"),
        ]:
            group[mean_name] = rolling_valid(group, sequence_id, source, rolling_frames, group["valid_mask"] == 1)
            group[std_name] = rolling_valid(group, sequence_id, source, rolling_frames, group["valid_mask"] == 1, std=True)

        result_parts.append(group)
    result = pd.concat(result_parts, ignore_index=True)
    result = result.sort_values(["frame", "team"]).reset_index(drop=True)
    return result[TEMPORAL_FEATURE_COLUMNS], thresholds


def rolling_valid(
    group: pd.DataFrame,
    sequence_id: pd.Series,
    column: str,
    rolling_frames: int,
    valid_series: pd.Series,
    std: bool = False,
) -> pd.Series:
    output = pd.Series(np.nan, index=group.index, dtype=float)
    temp = group[[column]].copy()
    temp["sequence_id"] = sequence_id
    temp["_valid"] = valid_series
    for _, sequence in temp.groupby("sequence_id", sort=False):
        valid_index = sequence.index[sequence["_valid"] & sequence[column].notna()]
        if len(valid_index) == 0:
            continue
        values = sequence.loc[valid_index, column]
        rolled = values.rolling(window=rolling_frames, min_periods=1)
        output.loc[valid_index] = (rolled.std(ddof=0) if std else rolled.mean()).to_numpy()
    return output


def percentile_or_inf(values: pd.Series, percentile: float) -> float:
    if len(values) == 0:
        return float("inf")
    return float(np.nanpercentile(values, percentile))


def quality_flag(valid_fraction: float) -> str:
    if valid_fraction >= 0.70:
        return "ok"
    if valid_fraction >= 0.40:
        return "partial"
    return "low_coverage"


def build_five_second_summary(features: pd.DataFrame, window_seconds: float) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=FIVE_SECOND_COLUMNS)
    start_time = float(features["timestamp"].min())
    working = features.copy()
    working["window_id"] = np.floor((working["timestamp"] - start_time) / window_seconds).astype(int)
    rows: list[dict[str, Any]] = []
    for (team, window_id), group in working.groupby(["team", "window_id"], sort=True):
        valid = group[group["valid_mask"] == 1]
        if valid.empty:
            continue
        total_rows = int(len(group))
        valid_rows = int(len(valid))
        valid_fraction = valid_rows / total_rows if total_rows else 0.0
        rows.append(
            {
                "clip_id": str(group["clip_id"].iloc[0]),
                "team": str(team),
                "window_id": int(window_id),
                "start_frame": int(group["frame"].min()),
                "end_frame": int(group["frame"].max()),
                "start_timestamp": float(group["timestamp"].min()),
                "end_timestamp": float(group["timestamp"].max()),
                "total_frame_rows": total_rows,
                "valid_shape_frames": valid_rows,
                "valid_shape_fraction": float(valid_fraction),
                "mean_centroid_x": mean_or_nan(valid["centroid_x"]),
                "mean_centroid_y": mean_or_nan(valid["centroid_y"]),
                "mean_width": mean_or_nan(valid["width"]),
                "mean_depth": mean_or_nan(valid["depth"]),
                "mean_spread": mean_or_nan(valid["spread"]),
                "mean_compactness_proxy": mean_or_nan(valid["compactness_proxy"]),
                "mean_centroid_speed_px_per_second": mean_or_nan(valid["centroid_speed_px_per_second"].dropna()),
                "width_variation_std": std_or_nan(valid["width"]),
                "depth_variation_std": std_or_nan(valid["depth"]),
                "spread_variation_std": std_or_nan(valid["spread"]),
                "shape_change_frame_count": int(group["shape_change_flag"].sum()),
                "shape_quality_flag": quality_flag(valid_fraction),
            }
        )
    return pd.DataFrame(rows, columns=FIVE_SECOND_COLUMNS)


def build_team_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for team, group in features.groupby("team", sort=True):
        valid = group[group["valid_mask"] == 1]
        total_rows = int(len(group))
        valid_rows = int(len(valid))
        valid_fraction = valid_rows / total_rows if total_rows else 0.0
        rows.append(
            {
                "clip_id": str(group["clip_id"].iloc[0]) if total_rows else "",
                "team": str(team),
                "total_frame_rows": total_rows,
                "valid_shape_frames": valid_rows,
                "valid_shape_fraction": float(valid_fraction),
                "mean_centroid_x": mean_or_nan(valid["centroid_x"]),
                "mean_centroid_y": mean_or_nan(valid["centroid_y"]),
                "mean_width": mean_or_nan(valid["width"]),
                "median_width": median_or_nan(valid["width"]),
                "mean_depth": mean_or_nan(valid["depth"]),
                "median_depth": median_or_nan(valid["depth"]),
                "mean_spread": mean_or_nan(valid["spread"]),
                "median_spread": median_or_nan(valid["spread"]),
                "mean_compactness_proxy": mean_or_nan(valid["compactness_proxy"]),
                "mean_centroid_speed_px_per_second": mean_or_nan(valid["centroid_speed_px_per_second"].dropna()),
                "width_std": std_or_nan(valid["width"]),
                "depth_std": std_or_nan(valid["depth"]),
                "spread_std": std_or_nan(valid["spread"]),
                "shape_change_frame_count": int(group["shape_change_flag"].sum()),
                "shape_quality_flag": quality_flag(valid_fraction),
            }
        )
    return pd.DataFrame(rows, columns=TEAM_SUMMARY_COLUMNS)


def write_schema(
    path: Path,
    clip_id: str,
    input_path: Path,
    row_count: int,
    rolling_frames: int,
    window_seconds: float,
    thresholds: dict[str, dict[str, float]],
) -> None:
    payload = {
        "clip_id": clip_id,
        "build_timestamp": utc_now_iso(),
        "input_path": str(input_path),
        "row_count": int(row_count),
        "field_mappings": TEAM_MAPPINGS,
        "rolling_settings": {
            "rolling_frames": int(rolling_frames),
            "policy": "Rolling metrics are computed within consecutive valid-shape sequences only.",
        },
        "five_second_window_settings": {
            "window_seconds": float(window_seconds),
            "policy": "Non-overlapping timestamp windows begin at the first clip timestamp.",
        },
        "gap_policy": "Deltas and centroid speed are not calculated across frame gaps greater than 1 or non-positive timestamp deltas.",
        "calculations": {
            "compactness_proxy": f"1 / max(spread, {EPSILON}) for valid positive spread; otherwise null.",
            "centroid_speed_px_per_second": "Euclidean centroid delta divided by positive timestamp delta.",
            "shape_change_flag": (
                f"1 when valid width/depth/spread absolute delta exceeds the team-specific "
                f"{SHAPE_CHANGE_PERCENTILE}th percentile of valid absolute deltas."
            ),
            "valid_mask": "1 only when shape_valid == 1 and centroid, width, depth, and spread are all present.",
        },
        "thresholds": json_safe_thresholds(thresholds),
        "warnings": [
            "All team shape metrics are in image pixel space.",
            "Camera motion, zoom, broadcast cuts, and track fragmentation can affect tactical-shape metrics.",
        ],
        "output_columns": {
            "shape_temporal_features": TEMPORAL_FEATURE_COLUMNS,
            "shape_5s_summary": FIVE_SECOND_COLUMNS,
            "team_shape_summary": TEAM_SUMMARY_COLUMNS,
        },
    }
    output = ensure_output_parent(path)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def process_clip(clip_id: str, temporal_frames_path: Path, derived_root: Path, rolling_frames: int, window_seconds: float) -> dict[str, Any]:
    frames = pd.read_csv(temporal_frames_path)
    require_columns(frames, temporal_frames_path, REQUIRED_TEMPORAL_COLUMNS)
    long_table = long_shape_table(clip_id, frames)
    features, thresholds = add_temporal_derivatives(long_table, rolling_frames)
    five_second = build_five_second_summary(features, window_seconds)
    team_summary = build_team_summary(features)

    analytics_dir = derived_root / clip_id / "analytics"
    model_features_dir = derived_root / clip_id / "model_features"
    features_path = ensure_output_parent(model_features_dir / "shape_temporal_features.csv")
    five_second_path = ensure_output_parent(analytics_dir / "shape_5s_summary.csv")
    team_summary_path = ensure_output_parent(analytics_dir / "team_shape_summary.csv")
    schema_path = analytics_dir / "shape_temporal_schema.json"

    features.to_csv(features_path, index=False)
    five_second.to_csv(five_second_path, index=False)
    team_summary.to_csv(team_summary_path, index=False)
    write_schema(schema_path, clip_id, temporal_frames_path, len(frames), rolling_frames, window_seconds, thresholds)

    team_a_fraction = valid_fraction_for_team(features, "Team A")
    team_b_fraction = valid_fraction_for_team(features, "Team B")
    return {
        "clip_id": clip_id,
        "status": "success",
        "input_frame_rows": int(len(frames)),
        "team_a_valid_fraction": team_a_fraction,
        "team_b_valid_fraction": team_b_fraction,
        "temporal_feature_rows_written": int(len(features)),
        "summary_rows_written": int(len(team_summary)),
        "five_second_rows_written": int(len(five_second)),
        "output_directory": str((derived_root / clip_id).resolve()),
        "error_message": "",
    }


def valid_fraction_for_team(features: pd.DataFrame, team: str) -> float:
    group = features[features["team"] == team]
    if group.empty:
        return 0.0
    return float((group["valid_mask"] == 1).sum() / len(group))


def json_safe_thresholds(thresholds: dict[str, dict[str, float]]) -> dict[str, dict[str, float | None]]:
    safe: dict[str, dict[str, float | None]] = {}
    for team, values in thresholds.items():
        safe[team] = {}
        for key, value in values.items():
            safe[team][key] = None if not np.isfinite(value) else float(value)
    return safe


def mean_or_nan(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else np.nan


def median_or_nan(values: pd.Series) -> float:
    return float(values.median()) if len(values) else np.nan


def std_or_nan(values: pd.Series) -> float:
    return float(values.std(ddof=0)) if len(values) else np.nan


def main() -> int:
    args = parse_args()
    try:
        if args.rolling_frames <= 0:
            raise ValueError("--rolling-frames must be positive")
        if args.window_seconds <= 0:
            raise ValueError("--window-seconds must be positive")
        derived_root = enforce_derived_root(Path(args.derived_root))
        rows: list[dict[str, Any]] = []
        for clip_id, temporal_frames_path in eligible_clips(derived_root):
            try:
                rows.append(process_clip(clip_id, temporal_frames_path, derived_root, args.rolling_frames, args.window_seconds))
            except Exception as error:
                input_rows = len(pd.read_csv(temporal_frames_path)) if temporal_frames_path.exists() else 0
                rows.append(
                    {
                        "clip_id": clip_id,
                        "status": "failed",
                        "input_frame_rows": input_rows,
                        "team_a_valid_fraction": 0.0,
                        "team_b_valid_fraction": 0.0,
                        "temporal_feature_rows_written": 0,
                        "summary_rows_written": 0,
                        "five_second_rows_written": 0,
                        "output_directory": "",
                        "error_message": str(error),
                    }
                )

        summary_path = ensure_output_parent(derived_root / "shape_temporal_build_summary.csv")
        with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=BUILD_SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

        print("Shape temporal build summary:")
        for row in rows:
            print(
                f"{row['clip_id']}: {row['status']} feature_rows={row['temporal_feature_rows_written']} "
                f"5s_rows={row['five_second_rows_written']} error={row['error_message']}"
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
