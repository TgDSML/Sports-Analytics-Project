import argparse
import csv
from collections import Counter, defaultdict
from math import dist
from pathlib import Path

CARRY_COLUMNS = [
    "carry_id",
    "player_id",
    "team",
    "start_frame",
    "end_frame",
    "start_time",
    "end_time",
    "frames",
    "duration_seconds",
    "start_ball_x",
    "start_ball_y",
    "end_ball_x",
    "end_ball_y",
    "ball_distance_px",
    "player_distance_px",
    "mean_distance_to_ball",
    "max_distance_to_ball",
    "mean_player_box_height",
    "carry_distance_threshold",
    "interpolated_frame_count",
    "carry_quality_flag",
]

SUMMARY_COLUMNS = ["section", "metric", "value"]


def estimate_carries(
    possession_csv_path: Path,
    possession_debug_csv_path: Path,
    player_tracks_csv_path: Path,
    output_csv_path: Path,
    summary_csv_path: Path,
    summary_md_path: Path,
    min_carry_frames: int = 5,
    min_carry_duration: float = 0.20,
    min_ball_distance_px: float = 20.0,
    min_player_distance_px: float = 10.0,
    max_frame_gap: int = 1,
    carry_distance_multiplier: float = 0.8,
    max_interpolated_share: float = 0.5,
) -> dict:
    possession_rows = _read_csv(
        possession_csv_path,
        required={"frame", "timestamp", "nearest_player_id", "team", "distance_to_ball"},
    )
    debug_rows = _read_csv(
        possession_debug_csv_path,
        required={
            "frame", "ball_center_x", "ball_center_y", "is_interpolated",
            "nearest_player_id", "distance_to_ball", "team"
        },
    )
    player_tracks = _read_player_tracks(player_tracks_csv_path)

    debug_by_frame = {int(r["frame"]): r for r in debug_rows}
    player_positions = _player_positions_by_frame(player_tracks)
    merged_rows = _merge_possession_with_debug(possession_rows, debug_by_frame, player_positions)
    segments = _build_possession_segments(merged_rows, max_frame_gap=max_frame_gap)
    carry_rows = _segments_to_carries(
        segments,
        min_carry_frames=min_carry_frames,
        min_carry_duration=min_carry_duration,
        min_ball_distance_px=min_ball_distance_px,
        min_player_distance_px=min_player_distance_px,
        carry_distance_multiplier=carry_distance_multiplier,
        max_interpolated_share=max_interpolated_share,
    )
    summary = _build_summary(carry_rows, carry_distance_multiplier)
    _write_rows(carry_rows, output_csv_path, CARRY_COLUMNS)
    _write_summary(summary, summary_csv_path, summary_md_path)
    print(f"Carry events CSV saved to: {output_csv_path}")
    print(f"Carry summary CSV saved to: {summary_csv_path}")
    print(f"Carry summary report saved to: {summary_md_path}")
    return summary


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


def _read_player_tracks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing = {"frame", "track_id", "center_x", "center_y"} - fieldnames
        if missing:
            raise ValueError(f"{path} missing column(s): {', '.join(sorted(missing))}")
        if "box_height" in fieldnames:
            height_column = "box_height"
        elif "height" in fieldnames:
            height_column = "height"
        else:
            raise ValueError(f"{path} missing column(s): box_height or height")
        rows = list(reader)
    for row in rows:
        row["box_height"] = row.get(height_column)
    return rows


def _player_positions_by_frame(player_tracks: list[dict]) -> dict[int, dict[int, tuple[float, float, float]]]:
    result: dict[int, dict[int, tuple[float, float, float]]] = defaultdict(dict)
    for row in player_tracks:
        frame = int(row["frame"])
        track_id = int(row["track_id"])
        result[frame][track_id] = (float(row["center_x"]), float(row["center_y"]), float(row["box_height"]))
    return result


def _merge_possession_with_debug(
    possession_rows: list[dict],
    debug_by_frame: dict[int, dict],
    player_positions: dict[int, dict[int, tuple[float, float, float]]],
) -> list[dict]:
    merged = []
    for row in possession_rows:
        frame = int(row["frame"])
        debug = debug_by_frame.get(frame, {})
        player_id = row.get("nearest_player_id")
        player_pos = None
        if player_id:
            try:
                player_pos = player_positions.get(frame, {}).get(int(player_id))
            except ValueError:
                player_pos = None
        merged.append(
            {
                "frame": frame,
                "timestamp": float(row["timestamp"]),
                "player_id": str(player_id or "").strip(),
                "team": row.get("team") or "None",
                "distance_to_ball": _float_or_none(row.get("distance_to_ball")),
                "ball_x": _float_or_none(debug.get("ball_center_x")),
                "ball_y": _float_or_none(debug.get("ball_center_y")),
                "is_interpolated": _truthy(debug.get("is_interpolated")),
                "player_x": player_pos[0] if player_pos else None,
                "player_y": player_pos[1] if player_pos else None,
                "player_box_height": player_pos[2] if player_pos else None,
            }
        )
    return merged


def _build_possession_segments(rows: list[dict], max_frame_gap: int) -> list[list[dict]]:
    segments = []
    current = []
    last_frame = None
    last_player = None
    last_team = None

    for row in rows:
        player_id = row["player_id"]
        team = row["team"]
        if not player_id or team == "None":
            if current:
                segments.append(current)
                current = []
            last_frame = None
            last_player = None
            last_team = None
            continue

        if not current:
            current = [row]
        else:
            frame_gap = row["frame"] - (last_frame if last_frame is not None else row["frame"])
            same_owner = player_id == last_player and team == last_team
            if same_owner and frame_gap <= max_frame_gap:
                current.append(row)
            else:
                segments.append(current)
                current = [row]
        last_frame = row["frame"]
        last_player = player_id
        last_team = team

    if current:
        segments.append(current)
    return segments


def _segments_to_carries(
    segments: list[list[dict]],
    min_carry_frames: int,
    min_carry_duration: float,
    min_ball_distance_px: float,
    min_player_distance_px: float,
    carry_distance_multiplier: float,
    max_interpolated_share: float,
) -> list[dict]:
    carries = []
    carry_id = 1
    for segment in segments:
        features = _compute_segment_features(segment, carry_distance_multiplier)
        if not _is_valid_carry(
            features,
            min_carry_frames=min_carry_frames,
            min_carry_duration=min_carry_duration,
            min_ball_distance_px=min_ball_distance_px,
            min_player_distance_px=min_player_distance_px,
            max_interpolated_share=max_interpolated_share,
        ):
            continue
        features["carry_id"] = carry_id
        carries.append(_format_carry_row(features))
        carry_id += 1
    return carries


def _compute_segment_features(segment: list[dict], carry_distance_multiplier: float) -> dict:
    first = segment[0]
    last = segment[-1]
    distances = [r["distance_to_ball"] for r in segment if r["distance_to_ball"] is not None]
    box_heights = [r["player_box_height"] for r in segment if r["player_box_height"] is not None]
    mean_box_height = sum(box_heights) / len(box_heights) if box_heights else 0.0
    carry_distance_threshold = mean_box_height * carry_distance_multiplier
    interpolated_count = sum(1 for r in segment if r["is_interpolated"])

    start_ball = _first_valid_point(segment, "ball")
    end_ball = _last_valid_point(segment, "ball")
    start_player = _first_valid_point(segment, "player")
    end_player = _last_valid_point(segment, "player")

    ball_distance = _point_distance(start_ball, end_ball)
    player_distance = _point_distance(start_player, end_player)
    duration = max(0.0, float(last["timestamp"]) - float(first["timestamp"]))
    frames = len(segment)
    interp_share = interpolated_count / frames if frames else 0.0

    quality = []
    if interp_share > 0:
        quality.append("contains_interpolation")
    if distances and carry_distance_threshold > 0 and sum(distances) / len(distances) > carry_distance_threshold:
        quality.append("loose_ball_player_distance")
    if not quality:
        quality.append("ok")

    return {
        "player_id": first["player_id"],
        "team": first["team"],
        "start_frame": first["frame"],
        "end_frame": last["frame"],
        "start_time": first["timestamp"],
        "end_time": last["timestamp"],
        "frames": frames,
        "duration_seconds": duration,
        "start_ball_x": start_ball[0] if start_ball else None,
        "start_ball_y": start_ball[1] if start_ball else None,
        "end_ball_x": end_ball[0] if end_ball else None,
        "end_ball_y": end_ball[1] if end_ball else None,
        "ball_distance_px": ball_distance,
        "player_distance_px": player_distance,
        "mean_distance_to_ball": sum(distances) / len(distances) if distances else 0.0,
        "max_distance_to_ball": max(distances) if distances else 0.0,
        "mean_player_box_height": mean_box_height,
        "carry_distance_threshold": carry_distance_threshold,
        "interpolated_frame_count": interpolated_count,
        "interpolated_share": interp_share,
        "carry_quality_flag": "; ".join(quality),
    }


def _is_valid_carry(
    features: dict,
    min_carry_frames: int,
    min_carry_duration: float,
    min_ball_distance_px: float,
    min_player_distance_px: float,
    max_interpolated_share: float,
) -> bool:
    if features["frames"] < min_carry_frames:
        return False
    if features["duration_seconds"] < min_carry_duration:
        return False
    if features["ball_distance_px"] < min_ball_distance_px:
        return False
    if features["player_distance_px"] < min_player_distance_px:
        return False
    if features["carry_distance_threshold"] <= 0:
        return False
    if features["mean_distance_to_ball"] > features["carry_distance_threshold"]:
        return False
    if features["interpolated_share"] > max_interpolated_share:
        return False
    return True


def _format_carry_row(features: dict) -> dict:
    return {
        "carry_id": features["carry_id"],
        "player_id": features["player_id"],
        "team": features["team"],
        "start_frame": features["start_frame"],
        "end_frame": features["end_frame"],
        "start_time": f"{features['start_time']:.6f}",
        "end_time": f"{features['end_time']:.6f}",
        "frames": features["frames"],
        "duration_seconds": f"{features['duration_seconds']:.3f}",
        "start_ball_x": _fmt(features["start_ball_x"]),
        "start_ball_y": _fmt(features["start_ball_y"]),
        "end_ball_x": _fmt(features["end_ball_x"]),
        "end_ball_y": _fmt(features["end_ball_y"]),
        "ball_distance_px": f"{features['ball_distance_px']:.2f}",
        "player_distance_px": f"{features['player_distance_px']:.2f}",
        "mean_distance_to_ball": f"{features['mean_distance_to_ball']:.2f}",
        "max_distance_to_ball": f"{features['max_distance_to_ball']:.2f}",
        "mean_player_box_height": f"{features['mean_player_box_height']:.2f}",
        "carry_distance_threshold": f"{features['carry_distance_threshold']:.2f}",
        "interpolated_frame_count": features["interpolated_frame_count"],
        "carry_quality_flag": features["carry_quality_flag"],
    }


def _build_summary(carry_rows: list[dict], carry_distance_multiplier: float) -> dict:
    total = len(carry_rows)
    team_counts = Counter(r["team"] for r in carry_rows)
    player_counts = Counter(r["player_id"] for r in carry_rows)
    durations = [float(r["duration_seconds"]) for r in carry_rows]
    ball_distances = [float(r["ball_distance_px"]) for r in carry_rows]
    box_heights = [float(r["mean_player_box_height"]) for r in carry_rows]
    thresholds = [float(r["carry_distance_threshold"]) for r in carry_rows]
    return {
        "total_carries": total,
        "carry_distance_multiplier": carry_distance_multiplier,
        "team_counts": team_counts,
        "player_counts": player_counts,
        "avg_duration_seconds": sum(durations) / len(durations) if durations else 0.0,
        "avg_ball_distance_px": sum(ball_distances) / len(ball_distances) if ball_distances else 0.0,
        "max_ball_distance_px": max(ball_distances) if ball_distances else 0.0,
        "avg_mean_player_box_height": sum(box_heights) / len(box_heights) if box_heights else 0.0,
        "avg_carry_distance_threshold": sum(thresholds) / len(thresholds) if thresholds else 0.0,
    }


def _write_summary(summary: dict, csv_path: Path, md_path: Path) -> None:
    rows = [
        {"section": "counts", "metric": "total_carries", "value": summary["total_carries"]},
        {"section": "dynamic_distance", "metric": "carry_distance_multiplier", "value": f"{summary['carry_distance_multiplier']:.2f}"},
        {"section": "metrics", "metric": "avg_duration_seconds", "value": f"{summary['avg_duration_seconds']:.3f}"},
        {"section": "metrics", "metric": "avg_ball_distance_px", "value": f"{summary['avg_ball_distance_px']:.2f}"},
        {"section": "metrics", "metric": "max_ball_distance_px", "value": f"{summary['max_ball_distance_px']:.2f}"},
        {"section": "dynamic_distance", "metric": "avg_mean_player_box_height", "value": f"{summary['avg_mean_player_box_height']:.2f}"},
        {"section": "dynamic_distance", "metric": "avg_carry_distance_threshold", "value": f"{summary['avg_carry_distance_threshold']:.2f}"},
    ]
    for team, count in sorted(summary["team_counts"].items()):
        rows.append({"section": "teams", "metric": team, "value": count})
    for player_id, count in summary["player_counts"].most_common(20):
        rows.append({"section": "players", "metric": player_id, "value": count})
    _write_rows(rows, csv_path, SUMMARY_COLUMNS)

    lines = [
        "# Carry Summary",
        "",
        f"- Total carries: {summary['total_carries']}",
        f"- Carry distance multiplier: {summary['carry_distance_multiplier']:.2f}",
        f"- Average carry duration: {summary['avg_duration_seconds']:.3f} s",
        f"- Average carry ball distance: {summary['avg_ball_distance_px']:.2f} px",
        f"- Max carry ball distance: {summary['max_ball_distance_px']:.2f} px",
        f"- Average player box height: {summary['avg_mean_player_box_height']:.2f} px",
        f"- Average carry distance threshold: {summary['avg_carry_distance_threshold']:.2f} px",
        "",
        "## Carries by Team",
        "",
    ]
    lines.extend(f"- {team}: {count}" for team, count in sorted(summary["team_counts"].items()))
    lines.extend(["", "## Top Players", ""])
    lines.extend(f"- Player {player_id}: {count}" for player_id, count in summary["player_counts"].most_common(10))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _first_valid_point(segment: list[dict], prefix: str):
    x_key = f"{prefix}_x"
    y_key = f"{prefix}_y"
    for row in segment:
        if row.get(x_key) is not None and row.get(y_key) is not None:
            return (float(row[x_key]), float(row[y_key]))
    return None


def _last_valid_point(segment: list[dict], prefix: str):
    x_key = f"{prefix}_x"
    y_key = f"{prefix}_y"
    for row in reversed(segment):
        if row.get(x_key) is not None and row.get(y_key) is not None:
            return (float(row[x_key]), float(row[y_key]))
    return None


def _point_distance(p1, p2) -> float:
    if p1 is None or p2 is None:
        return 0.0
    return dist(p1, p2)


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float_or_none(value):
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _fmt(value):
    return "" if value is None else f"{float(value):.2f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate player carries from possession outputs")
    parser.add_argument("--possession-csv", type=Path, required=True)
    parser.add_argument("--possession-debug-csv", type=Path, required=True)
    parser.add_argument("--player-tracks-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/carry_events_30s_720p.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/carry_summary_30s_720p.csv"))
    parser.add_argument("--summary-md", type=Path, default=Path("outputs/carry_summary_30s_720p.md"))
    parser.add_argument("--min-carry-frames", type=int, default=5)
    parser.add_argument("--min-carry-duration", type=float, default=0.20)
    parser.add_argument("--min-ball-distance-px", type=float, default=20.0)
    parser.add_argument("--min-player-distance-px", type=float, default=10.0)
    parser.add_argument("--max-frame-gap", type=int, default=1)
    parser.add_argument("--carry-distance-multiplier", type=float, default=0.8)
    parser.add_argument("--max-interpolated-share", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    estimate_carries(
        possession_csv_path=args.possession_csv,
        possession_debug_csv_path=args.possession_debug_csv,
        player_tracks_csv_path=args.player_tracks_csv,
        output_csv_path=args.output_csv,
        summary_csv_path=args.summary_csv,
        summary_md_path=args.summary_md,
        min_carry_frames=args.min_carry_frames,
        min_carry_duration=args.min_carry_duration,
        min_ball_distance_px=args.min_ball_distance_px,
        min_player_distance_px=args.min_player_distance_px,
        max_frame_gap=args.max_frame_gap,
        carry_distance_multiplier=args.carry_distance_multiplier,
        max_interpolated_share=args.max_interpolated_share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())