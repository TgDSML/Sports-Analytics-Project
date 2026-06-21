"""Estimate pass counts from possession output."""

import argparse
import csv
from pathlib import Path

import pandas as pd


SUMMARY_COLUMNS = ["metric", "value"]
TEAM_LABELS = {"Team A", "Team B"}


def write_summary(
    possession_csv: Path,
    output_csv: Path,
    output_md: Path,
    min_possession_frames: int = 6,
    min_possession_duration: float = 0.20,
    max_transfer_gap: float = 2.0,
) -> dict[str, float]:
    """Write pass summary files from a possession CSV."""
    possession_df = pd.read_csv(possession_csv)
    frame_possession = _frame_level_possession(possession_df)
    segments = _stable_possession_segments(
        frame_possession,
        min_possession_frames=min_possession_frames,
        min_possession_duration=min_possession_duration,
    )
    pass_events = _pass_events(segments, max_transfer_gap=max_transfer_gap)
    success = sum(1 for event in pass_events if event["from_team"] == event["to_team"])
    failed = len(pass_events) - success
    total = success + failed
    summary = {
        "successful_passes": success,
        "failed_passes": failed,
        "total_passes": total,
        "pass_accuracy": (success / total) * 100 if total else 0.0,
        "stable_possession_segments": len(segments),
        "min_possession_frames": int(min_possession_frames),
        "min_possession_duration": float(min_possession_duration),
        "max_transfer_gap": float(max_transfer_gap),
    }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for metric, value in summary.items():
            writer.writerow({"metric": metric, "value": value})

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        "\n".join(
            [
                "# Passing Summary",
                "",
                f"- Successful passes: {summary['successful_passes']}",
                f"- Failed passes: {summary['failed_passes']}",
                f"- Total passes: {summary['total_passes']}",
                f"- Pass accuracy: {summary['pass_accuracy']:.2f}%",
                f"- Stable possession segments: {summary['stable_possession_segments']}",
                f"- Minimum possession frames: {summary['min_possession_frames']}",
                f"- Minimum possession duration: {summary['min_possession_duration']:.3f} s",
                f"- Maximum transfer gap: {summary['max_transfer_gap']:.3f} s",
            ]
        ),
        encoding="utf-8",
    )
    return summary


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
        active = active.sort_values(["distance_to_ball", "timestamp"], na_position="last")
        chosen = active.iloc[0]
        rows.append(
            {
                "frame": int(frame),
                "timestamp": float(chosen["timestamp"]),
                "team": str(chosen["team"]),
                "nearest_player_id": int(chosen["nearest_player_id"]),
            }
        )
    return pd.DataFrame(rows)


def _stable_possession_segments(
    frame_possession: pd.DataFrame,
    min_possession_frames: int,
    min_possession_duration: float,
) -> list[dict]:
    segments = []
    current = None
    for row in frame_possession.itertuples(index=False):
        player_id = None if pd.isna(row.nearest_player_id) else int(row.nearest_player_id)
        owner = (row.team, player_id) if row.team in TEAM_LABELS and player_id is not None else None
        if owner is None:
            if current is not None:
                _append_stable_segment(segments, current, min_possession_frames, min_possession_duration)
                current = None
            continue
        if current is None or current["team"] != owner[0] or current["player_id"] != owner[1]:
            if current is not None:
                _append_stable_segment(segments, current, min_possession_frames, min_possession_duration)
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
        _append_stable_segment(segments, current, min_possession_frames, min_possession_duration)
    return segments


def _append_stable_segment(
    segments: list[dict],
    segment: dict,
    min_possession_frames: int,
    min_possession_duration: float,
) -> None:
    duration = max(0.0, float(segment["end_time"]) - float(segment["start_time"]))
    if segment["frames"] < min_possession_frames:
        return
    if duration < min_possession_duration:
        return
    stable = dict(segment)
    stable["duration"] = duration
    segments.append(stable)


def _pass_events(segments: list[dict], max_transfer_gap: float) -> list[dict]:
    events = []
    for previous, current in zip(segments, segments[1:]):
        if previous["player_id"] == current["player_id"]:
            continue
        gap = max(0.0, float(current["start_time"]) - float(previous["end_time"]))
        if gap > max_transfer_gap:
            continue
        events.append(
            {
                "from_player": previous["player_id"],
                "to_player": current["player_id"],
                "from_team": previous["team"],
                "to_team": current["team"],
                "gap": gap,
            }
        )
    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate pass counts from possession output")
    parser.add_argument(
        "--possession-csv",
        type=Path,
        default=Path("outputs/possession_30s_720p.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/passing_summary_30s_720p.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/passing_summary_30s_720p.md"),
    )
    parser.add_argument("--min-possession-frames", type=int, default=6)
    parser.add_argument("--min-possession-duration", type=float, default=0.20)
    parser.add_argument("--max-transfer-gap", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = write_summary(
        args.possession_csv,
        args.output_csv,
        args.output_md,
        min_possession_frames=args.min_possession_frames,
        min_possession_duration=args.min_possession_duration,
        max_transfer_gap=args.max_transfer_gap,
    )
    print(f"Passing summary saved to: {args.output_csv}")
    print(f"Passing report saved to: {args.output_md}")
    print(f"Successful passes: {summary['successful_passes']}")
    print(f"Failed passes: {summary['failed_passes']}")
    print(f"Pass accuracy: {summary['pass_accuracy']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
