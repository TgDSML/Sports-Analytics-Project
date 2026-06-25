"""Export weak pass event rows without touching existing outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import ensure_output_parent, read_csv_with_header, reject_outputs_path, utc_now_iso  # noqa: E402


TEAM_LABELS = {"Team A", "Team B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export weak pass events from possession transitions.")
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--possession", required=True)
    parser.add_argument("--possession-debug")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-possession-frames", type=int, default=6)
    parser.add_argument("--min-possession-duration", type=float, default=0.20)
    parser.add_argument("--max-transfer-gap", type=float, default=2.0)
    return parser.parse_args()


def _frame_level_possession(possession_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frame, group in possession_df.sort_values(["frame", "timestamp"]).groupby("frame", sort=True):
        active = group[group["team"].isin(TEAM_LABELS)].dropna(subset=["nearest_player_id"])
        if active.empty:
            first = group.iloc[0]
            rows.append(
                {
                    "frame": int(frame),
                    "timestamp": float(first["timestamp"]),
                    "team": "None",
                    "nearest_player_id": pd.NA,
                }
            )
            continue
        active = active.copy()
        active["distance_to_ball"] = pd.to_numeric(active["distance_to_ball"], errors="coerce")
        chosen = active.sort_values(["distance_to_ball", "timestamp"], na_position="last").iloc[0]
        rows.append(
            {
                "frame": int(frame),
                "timestamp": float(chosen["timestamp"]),
                "team": str(chosen["team"]),
                "nearest_player_id": int(chosen["nearest_player_id"]),
            }
        )
    return pd.DataFrame(rows)


def _stable_segments(frame_possession: pd.DataFrame, min_frames: int, min_duration: float) -> list[dict]:
    segments = []
    current = None
    for row in frame_possession.itertuples(index=False):
        player_id = None if pd.isna(row.nearest_player_id) else int(row.nearest_player_id)
        owner = (row.team, player_id) if row.team in TEAM_LABELS and player_id is not None else None
        if owner is None:
            if current is not None:
                _append_segment(segments, current, min_frames, min_duration)
                current = None
            continue
        if current is None or current["team"] != owner[0] or current["player_id"] != owner[1]:
            if current is not None:
                _append_segment(segments, current, min_frames, min_duration)
            current = {
                "team": owner[0],
                "player_id": owner[1],
                "start_frame": int(row.frame),
                "end_frame": int(row.frame),
                "start_time": float(row.timestamp),
                "end_time": float(row.timestamp),
                "frames": 1,
            }
        else:
            current["end_frame"] = int(row.frame)
            current["end_time"] = float(row.timestamp)
            current["frames"] += 1
    if current is not None:
        _append_segment(segments, current, min_frames, min_duration)
    return segments


def _append_segment(segments: list[dict], segment: dict, min_frames: int, min_duration: float) -> None:
    duration = max(0.0, float(segment["end_time"]) - float(segment["start_time"]))
    if segment["frames"] < min_frames or duration < min_duration:
        return
    stable = dict(segment)
    stable["duration"] = duration
    segments.append(stable)


def build_pass_events(
    clip_id: str,
    possession_df: pd.DataFrame,
    min_frames: int,
    min_duration: float,
    max_transfer_gap: float,
) -> pd.DataFrame:
    frame_possession = _frame_level_possession(possession_df)
    segments = _stable_segments(frame_possession, min_frames, min_duration)
    rows = []
    for previous, current in zip(segments, segments[1:]):
        if previous["team"] != current["team"]:
            continue
        if previous["player_id"] == current["player_id"]:
            continue
        gap = max(0.0, float(current["start_time"]) - float(previous["end_time"]))
        if gap > max_transfer_gap:
            continue
        rows.append(
            {
                "clip_id": clip_id,
                "pass_id": len(rows) + 1,
                "start_frame": int(previous["end_frame"]),
                "end_frame": int(current["start_frame"]),
                "start_timestamp": float(previous["end_time"]),
                "end_timestamp": float(current["start_time"]),
                "from_player_id": int(previous["player_id"]),
                "to_player_id": int(current["player_id"]),
                "team": current["team"],
                "confidence": 0.5,
                "quality_flag": f"stable_segments_gap_{gap:.3f}s",
                "label_source": "possession_transition_heuristic",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "clip_id",
            "pass_id",
            "start_frame",
            "end_frame",
            "start_timestamp",
            "end_timestamp",
            "from_player_id",
            "to_player_id",
            "team",
            "confidence",
            "quality_flag",
            "label_source",
        ],
    )


def main() -> int:
    args = parse_args()
    try:
        reject_outputs_path(args.output)
        possession_df, possession_header = read_csv_with_header(args.possession)
        debug_header = []
        if args.possession_debug:
            _, debug_header = read_csv_with_header(args.possession_debug)
        events = build_pass_events(
            clip_id=args.clip_id,
            possession_df=possession_df,
            min_frames=args.min_possession_frames,
            min_duration=args.min_possession_duration,
            max_transfer_gap=args.max_transfer_gap,
        )
        output = ensure_output_parent(args.output)
        events.to_csv(output, index=False)
        schema_path = output.with_name("passes_weak_schema.json")
        schema = {
            "clip_id": args.clip_id,
            "build_timestamp": utc_now_iso(),
            "input_paths": {
                "possession": str(Path(args.possession)),
                "possession_debug": str(Path(args.possession_debug)) if args.possession_debug else "",
            },
            "detected_source_headers": {
                "possession": possession_header,
                "possession_debug": debug_header,
            },
            "event_count": int(len(events)),
            "output_columns": list(events.columns),
            "label_source": "possession_transition_heuristic",
            "assumptions": [
                "Existing passing_stats.py keeps pass event dictionaries internally but does not export event rows.",
                "Weak pass events are same-team stable possession transitions from player A to player B.",
                "Weak pass events are not ground truth.",
            ],
            "settings": {
                "min_possession_frames": args.min_possession_frames,
                "min_possession_duration": args.min_possession_duration,
                "max_transfer_gap": args.max_transfer_gap,
            },
        }
        schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Output path: {output}")
        print(f"Schema path: {schema_path}")
        print(f"Event count: {len(events)}")
        print("Source: conservative possession-transition fallback")
        print("First 10 rows:")
        print(events.head(10).to_string(index=False))
        return 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

