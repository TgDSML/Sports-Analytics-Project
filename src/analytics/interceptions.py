import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

INTERCEPTION_COLUMNS = [
    "interception_id",
    "frame",
    "timestamp",
    "winner_player_id",
    "winner_team",
    "previous_player_id",
    "previous_team",
    "ball_x",
    "ball_y",
    "distance_to_ball",
    "interception_type",
    "quality_flag",
]

SUMMARY_COLUMNS = ["section", "metric", "value"]


def build_interceptions_csv(
    possession_csv_path: Path,
    possession_debug_csv_path: Path,
    output_csv_path: Path,
    summary_csv_path: Path,
    summary_md_path: Path,
    min_prev_frames: int = 3,
    min_new_frames: int = 2,
) -> dict:
    possession_rows = _read_csv(
        possession_csv_path,
        required={"frame", "timestamp", "nearest_player_id", "team", "distance_to_ball"},
    )
    debug_rows = _read_csv(
        possession_debug_csv_path,
        required={"frame", "ball_center_x", "ball_center_y", "team", "nearest_player_id", "distance_to_ball"},
    )

    debug_by_frame = {int(row["frame"]): row for row in debug_rows}
    merged_rows = []
    for row in possession_rows:
        frame = int(row["frame"])
        debug = debug_by_frame.get(frame, {})
        merged_rows.append(
            {
                "frame": frame,
                "timestamp": float(row["timestamp"]),
                "player_id": str(row.get("nearest_player_id") or "").strip(),
                "team": str(row.get("team") or "None"),
                "distance_to_ball": _float_or_none(row.get("distance_to_ball")),
                "ball_x": _float_or_none(debug.get("ball_center_x")),
                "ball_y": _float_or_none(debug.get("ball_center_y")),
            }
        )

    segments = _build_segments(merged_rows)
    events = _detect_interceptions(segments, min_prev_frames=min_prev_frames, min_new_frames=min_new_frames)
    summary = _build_summary(events)

    _write_rows(events, output_csv_path, INTERCEPTION_COLUMNS)
    _write_summary(summary, summary_csv_path, summary_md_path)

    print(f"Interceptions CSV saved to: {output_csv_path}")
    print(f"Interceptions summary CSV saved to: {summary_csv_path}")
    print(f"Interceptions summary report saved to: {summary_md_path}")
    return summary


def _build_segments(rows: list[dict]) -> list[dict]:
    segments = []
    current = []
    for row in rows:
        if not row["player_id"] or row["team"] == "None":
            if current:
                segments.append(_segment_from_rows(current))
                current = []
            continue

        if not current:
            current = [row]
            continue

        contiguous = row["frame"] == current[-1]["frame"] + 1
        same_owner = row["player_id"] == current[-1]["player_id"] and row["team"] == current[-1]["team"]
        if contiguous and same_owner:
            current.append(row)
        else:
            segments.append(_segment_from_rows(current))
            current = [row]

    if current:
        segments.append(_segment_from_rows(current))
    return segments


def _segment_from_rows(rows: list[dict]) -> dict:
    return {
        "player_id": rows[0]["player_id"],
        "team": rows[0]["team"],
        "start_frame": rows[0]["frame"],
        "end_frame": rows[-1]["frame"],
        "frames": len(rows),
        "rows": rows,
    }


def _detect_interceptions(segments: list[dict], min_prev_frames: int, min_new_frames: int) -> list[dict]:
    events = []
    interception_id = 1
    for index in range(1, len(segments)):
        previous = segments[index - 1]
        current = segments[index]

        if previous["team"] not in {"Team A", "Team B"} or current["team"] not in {"Team A", "Team B"}:
            continue
        if previous["team"] == current["team"]:
            continue
        if previous["frames"] < max(1, min_prev_frames):
            continue
        if current["frames"] < max(1, min_new_frames):
            continue

        event_row = current["rows"][0]
        quality_flags = []
        if previous["frames"] < 5:
            quality_flags.append("short_previous_segment")
        if current["frames"] < 3:
            quality_flags.append("short_new_segment")
        if not quality_flags:
            quality_flags.append("ok")

        events.append(
            {
                "interception_id": interception_id,
                "frame": event_row["frame"],
                "timestamp": f"{event_row['timestamp']:.6f}",
                "winner_player_id": current["player_id"],
                "winner_team": current["team"],
                "previous_player_id": previous["player_id"],
                "previous_team": previous["team"],
                "ball_x": _fmt(event_row.get("ball_x")),
                "ball_y": _fmt(event_row.get("ball_y")),
                "distance_to_ball": _fmt(event_row.get("distance_to_ball")),
                "interception_type": "opponent_possession_win",
                "quality_flag": "; ".join(quality_flags),
            }
        )
        interception_id += 1
    return events


def _build_summary(rows: list[dict]) -> dict:
    team_counts = Counter(row["winner_team"] for row in rows)
    player_counts = Counter(row["winner_player_id"] for row in rows)
    return {
        "total_interceptions": len(rows),
        "team_counts": team_counts,
        "player_counts": player_counts,
    }


def _write_summary(summary: dict, csv_path: Path, md_path: Path) -> None:
    csv_rows = [{"section": "counts", "metric": "total_interceptions", "value": summary["total_interceptions"]}]
    for team, count in sorted(summary["team_counts"].items()):
        csv_rows.append({"section": "teams", "metric": team, "value": count})
    for player_id, count in summary["player_counts"].most_common(20):
        csv_rows.append({"section": "players", "metric": player_id, "value": count})
    _write_rows(csv_rows, csv_path, SUMMARY_COLUMNS)

    lines = [
        "# Interceptions Summary",
        "",
        f"- Total interceptions: {summary['total_interceptions']}",
        "",
        "## Interceptions by Team",
        "",
    ]
    lines.extend(f"- {team}: {count}" for team, count in sorted(summary["team_counts"].items()))
    lines.extend(["", "## Top Players", ""])
    lines.extend(f"- Player {player_id}: {count}" for player_id, count in summary["player_counts"].most_common(10))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _read_csv(path: Path, required: set[str]) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing column(s): {', '.join(sorted(missing))}")
        return list(reader)


def _write_rows(rows: list[dict], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float_or_none(value):
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _fmt(value):
    return "" if value is None else f"{float(value):.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build interceptions CSV required by interception_map.py")
    parser.add_argument("--possession-csv", type=Path, required=True)
    parser.add_argument("--possession-debug-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/interceptions_30s_720p.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/interceptions_summary_30s_720p.csv"))
    parser.add_argument("--summary-md", type=Path, default=Path("outputs/interceptions_summary_30s_720p.md"))
    parser.add_argument("--min-prev-frames", type=int, default=3)
    parser.add_argument("--min-new-frames", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_interceptions_csv(
        possession_csv_path=args.possession_csv,
        possession_debug_csv_path=args.possession_debug_csv,
        output_csv_path=args.output_csv,
        summary_csv_path=args.summary_csv,
        summary_md_path=args.summary_md,
        min_prev_frames=args.min_prev_frames,
        min_new_frames=args.min_new_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())