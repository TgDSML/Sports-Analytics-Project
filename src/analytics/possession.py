"""Experimental baseline possession estimation from ball and player tracks."""

import argparse
import csv
from collections import Counter
from math import dist
from pathlib import Path

import cv2


POSSESSION_COLUMNS = [
    "frame",
    "timestamp",
    "ball_track_id",
    "nearest_player_id",
    "team",
    "distance_to_ball",
]

POSSESSION_DEBUG_COLUMNS = [
    "frame",
    "timestamp",
    "ball_track_id",
    "ball_center_x",
    "ball_center_y",
    "ball_confidence",
    "is_interpolated",
    "candidate_team",
    "team",
    "nearest_player_id",
    "nearest_player_role",
    "nearest_player_team",
    "nearest_player_confidence",
    "nearest_player_center_x",
    "nearest_player_center_y",
    "nearest_player_box_height",
    "distance_to_ball",
    "dynamic_distance_threshold",
    "possession_reason",
]

ELIGIBLE_POSSESSION_ROLES = {"team_a_player", "team_b_player"}
TEAM_LABELS = ("Team A", "Team B", "Unknown", "None")


def estimate_possession(
    player_tracks_csv_path: Path,
    teams_csv_path: Path,
    ball_tracks_csv_path: Path,
    output_csv_path: Path,
    summary_csv_path: Path,
    summary_md_path: Path,
    debug_csv_path: Path,
    qa_summary_md_path: Path,
    dynamic_distance_multiplier: float = 1.5,
    min_track_confidence: float = 0.10,
    min_ball_confidence: float = 0.25,
    assign_interpolated: bool = False,
    switch_confirmation_frames: int = 3,
    eligible_roles: set[str] | None = None,
) -> dict:
    """Estimate frame-level possession with conservative QA gates."""
    player_tracks = _read_player_tracks(player_tracks_csv_path)
    ball_tracks = _read_csv(
        ball_tracks_csv_path,
        required={"track_id", "frame", "timestamp", "center_x", "center_y", "confidence", "is_interpolated"},
    )
    team_info = _read_team_assignments(teams_csv_path)
    eligible_roles = eligible_roles or ELIGIBLE_POSSESSION_ROLES

    players_by_frame = _players_by_frame(
        player_tracks=player_tracks,
        team_info=team_info,
        min_track_confidence=min_track_confidence,
        eligible_roles=eligible_roles,
    )
    debug_rows = _build_candidate_rows(
        ball_tracks=ball_tracks,
        players_by_frame=players_by_frame,
        team_info=team_info,
        dynamic_distance_multiplier=dynamic_distance_multiplier,
        min_ball_confidence=min_ball_confidence,
        assign_interpolated=assign_interpolated,
    )
    _apply_possession_smoothing(debug_rows, switch_confirmation_frames)
    possession_rows = _compact_possession_rows(debug_rows)

    _write_rows(possession_rows, output_csv_path, POSSESSION_COLUMNS)
    _write_rows(debug_rows, debug_csv_path, POSSESSION_DEBUG_COLUMNS)
    summary = _build_summary(debug_rows)
    _write_summary(summary, summary_csv_path, summary_md_path)
    _write_qa_summary(
        summary=summary,
        qa_summary_md_path=qa_summary_md_path,
        dynamic_distance_multiplier=dynamic_distance_multiplier,
        min_track_confidence=min_track_confidence,
        min_ball_confidence=min_ball_confidence,
        assign_interpolated=assign_interpolated,
        switch_confirmation_frames=switch_confirmation_frames,
        eligible_roles=eligible_roles,
    )
    print(f"Possession CSV saved to: {output_csv_path}")
    print(f"Possession debug CSV saved to: {debug_csv_path}")
    print(f"Possession summary saved to: {summary_csv_path}")
    print(f"Possession report saved to: {summary_md_path}")
    print(f"Possession QA summary saved to: {qa_summary_md_path}")
    return summary


def create_possession_video(
    video_path: Path,
    possession_csv_path: Path,
    output_video_path: Path,
) -> None:
    """Overlay frame-level possession labels on a video."""
    rows = _read_csv(possession_csv_path, required={"frame", "team"})
    possession_by_frame = {int(row["frame"]): row for row in rows}
    _write_possession_video(
        video_path=video_path,
        rows_by_frame=possession_by_frame,
        output_video_path=output_video_path,
        debug=False,
    )
    print(f"Possession video saved to: {output_video_path}")


def create_possession_debug_video(
    video_path: Path,
    debug_csv_path: Path,
    output_video_path: Path,
) -> None:
    """Overlay possession QA details on a video."""
    rows = _read_csv(debug_csv_path, required={"frame", "team", "possession_reason"})
    rows_by_frame = {int(row["frame"]): row for row in rows}
    _write_possession_video(
        video_path=video_path,
        rows_by_frame=rows_by_frame,
        output_video_path=output_video_path,
        debug=True,
    )
    print(f"Possession debug video saved to: {output_video_path}")


def _players_by_frame(
    player_tracks: list[dict],
    team_info: dict[int, dict[str, str]],
    min_track_confidence: float,
    eligible_roles: set[str],
) -> dict[int, list[dict]]:
    players_by_frame = {}
    for row in player_tracks:
        confidence = float(row.get("confidence") or 0.0)
        if confidence < min_track_confidence:
            continue
        track_id = int(row["track_id"])
        info = team_info.get(track_id, {"team": "Unknown", "role": "unknown"})
        role = info["role"]
        team = info["team"]
        if role not in eligible_roles or team not in {"Team A", "Team B"}:
            continue
        player = dict(row)
        player["team"] = team
        player["role"] = role
        players_by_frame.setdefault(int(row["frame"]), []).append(player)
    return players_by_frame


def _build_candidate_rows(
    ball_tracks: list[dict],
    players_by_frame: dict[int, list[dict]],
    team_info: dict[int, dict[str, str]],
    dynamic_distance_multiplier: float,
    min_ball_confidence: float,
    assign_interpolated: bool,
) -> list[dict]:
    rows = []
    for ball in ball_tracks:
        frame = int(ball["frame"])
        ball_confidence = float(ball.get("confidence") or 0.0)
        is_interpolated = _truthy(ball.get("is_interpolated"))
        nearest = _nearest_player(ball, players_by_frame.get(frame, []))
        dynamic_threshold = _dynamic_distance_threshold(nearest, dynamic_distance_multiplier)
        row = _base_debug_row(ball, nearest, team_info, dynamic_threshold)

        if is_interpolated and not assign_interpolated:
            row["candidate_team"] = "None"
            row["possession_reason"] = "interpolated_ball_skipped"
        elif ball_confidence < min_ball_confidence:
            row["candidate_team"] = "None"
            row["possession_reason"] = "ball_confidence_below_threshold"
        elif nearest is None:
            row["candidate_team"] = "None"
            row["possession_reason"] = "no_eligible_player_in_frame"
        elif dynamic_threshold is None or nearest["distance"] > dynamic_threshold:
            row["candidate_team"] = "None"
            row["possession_reason"] = "nearest_player_too_far"
        else:
            row["candidate_team"] = nearest["player"]["team"]
            row["possession_reason"] = "nearest_eligible_player_within_threshold"
        rows.append(row)
    return rows


def _base_debug_row(
    ball: dict,
    nearest: dict | None,
    team_info: dict[int, dict[str, str]],
    dynamic_threshold: float | None,
) -> dict:
    nearest_player = nearest["player"] if nearest else None
    nearest_id = int(nearest_player["track_id"]) if nearest_player else ""
    nearest_info = team_info.get(nearest_id, {"team": "Unknown", "role": "unknown"}) if nearest_player else {}
    return {
        "frame": int(ball["frame"]),
        "timestamp": ball["timestamp"],
        "ball_track_id": ball["track_id"],
        "ball_center_x": f"{float(ball['center_x']):.2f}",
        "ball_center_y": f"{float(ball['center_y']):.2f}",
        "ball_confidence": f"{float(ball.get('confidence') or 0.0):.6f}",
        "is_interpolated": int(_truthy(ball.get("is_interpolated"))),
        "candidate_team": "None",
        "team": "None",
        "nearest_player_id": nearest_id,
        "nearest_player_role": nearest_info.get("role", ""),
        "nearest_player_team": nearest_info.get("team", ""),
        "nearest_player_confidence": (
            f"{float(nearest_player.get('confidence') or 0.0):.6f}" if nearest_player else ""
        ),
        "nearest_player_center_x": (
            f"{float(nearest_player['center_x']):.2f}" if nearest_player else ""
        ),
        "nearest_player_center_y": (
            f"{float(nearest_player['center_y']):.2f}" if nearest_player else ""
        ),
        "nearest_player_box_height": (
            f"{float(nearest_player['box_height']):.2f}" if nearest_player else ""
        ),
        "distance_to_ball": f"{nearest['distance']:.2f}" if nearest else "",
        "dynamic_distance_threshold": f"{dynamic_threshold:.2f}" if dynamic_threshold is not None else "",
        "possession_reason": "",
    }


def _apply_possession_smoothing(rows: list[dict], switch_confirmation_frames: int) -> None:
    if not rows:
        return

    required = max(1, int(switch_confirmation_frames))
    current = "None"
    pending = None
    pending_count = 0

    for row in rows:
        candidate = row["candidate_team"]
        if candidate == "None":
            current = "None"
            pending = None
            pending_count = 0
            row["team"] = "None"
            continue

        if candidate == current:
            pending = None
            pending_count = 0
            row["team"] = current
            continue

        if candidate != pending:
            pending = candidate
            pending_count = 1
        else:
            pending_count += 1

        if pending_count >= required:
            current = candidate
            pending = None
            pending_count = 0
            row["team"] = current
            row["possession_reason"] = f"{row['possession_reason']}; smoothed_switch_confirmed"
        else:
            row["team"] = current
            row["possession_reason"] = (
                f"{row['possession_reason']}; smoothed_pending_switch_to_{candidate}"
            )


def _compact_possession_rows(debug_rows: list[dict]) -> list[dict]:
    rows = []
    for row in debug_rows:
        rows.append(
            {
                "frame": row["frame"],
                "timestamp": row["timestamp"],
                "ball_track_id": row["ball_track_id"],
                "nearest_player_id": row["nearest_player_id"] if row["team"] != "None" else "",
                "team": row["team"],
                "distance_to_ball": row["distance_to_ball"] if row["team"] != "None" else "",
            }
        )
    return rows


def _read_csv(path: Path, required: set[str]) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing column(s): {', '.join(sorted(missing))}")
        return list(reader)


def _read_player_tracks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = set(reader.fieldnames or [])
        missing = {"frame", "track_id", "center_x", "center_y", "confidence"} - fieldnames
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


def _read_team_assignments(path: Path) -> dict[int, dict[str, str]]:
    rows = _read_csv(path, required={"track_id", "team"})
    result = {}
    for row in rows:
        if not row.get("track_id"):
            continue
        role = str(row.get("role") or "unknown")
        result[int(row["track_id"])] = {
            "team": str(row.get("team") or "Unknown"),
            "role": role,
        }
    return result


def _nearest_player(ball: dict, players: list[dict]) -> dict | None:
    if not players:
        return None
    ball_center = (float(ball["center_x"]), float(ball["center_y"]))
    nearest = None
    for player in players:
        distance = dist(ball_center, (float(player["center_x"]), float(player["center_y"])))
        if nearest is None or distance < nearest["distance"]:
            nearest = {"player": player, "distance": distance}
    return nearest


def _dynamic_distance_threshold(nearest: dict | None, dynamic_distance_multiplier: float) -> float | None:
    if nearest is None:
        return None
    return float(nearest["player"]["box_height"]) * dynamic_distance_multiplier


def _write_rows(rows: list[dict], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _build_summary(rows: list[dict]) -> dict:
    counts = Counter(row["team"] for row in rows)
    candidate_counts = Counter(row["candidate_team"] for row in rows)
    reasons = Counter(_primary_reason(row["possession_reason"]) for row in rows)
    total = len(rows)

    def pct(label: str) -> float:
        return 100 * counts.get(label, 0) / total if total else 0.0

    distances = [
        float(row["distance_to_ball"])
        for row in rows
        if row.get("distance_to_ball") and row["team"] != "None"
    ]
    return {
        "total_ball_frames": total,
        "team_a_frames": counts.get("Team A", 0),
        "team_b_frames": counts.get("Team B", 0),
        "unknown_frames": counts.get("Unknown", 0),
        "none_frames": counts.get("None", 0),
        "team_a_percent": pct("Team A"),
        "team_b_percent": pct("Team B"),
        "unknown_percent": pct("Unknown"),
        "none_percent": pct("None"),
        "candidate_counts": candidate_counts,
        "reasons": reasons,
        "assigned_distance_avg": sum(distances) / len(distances) if distances else 0.0,
        "assigned_distance_max": max(distances) if distances else 0.0,
        "interpolated_rows": sum(_truthy(row["is_interpolated"]) for row in rows),
        "low_confidence_rows": reasons.get("ball_confidence_below_threshold", 0),
    }


def _write_summary(summary: dict, csv_path: Path, md_path: Path) -> None:
    rows = [
        ("counts", "total_ball_frames", summary["total_ball_frames"]),
        ("counts", "team_a_frames", summary["team_a_frames"]),
        ("counts", "team_b_frames", summary["team_b_frames"]),
        ("counts", "unknown_frames", summary["unknown_frames"]),
        ("counts", "frames_with_no_possession", summary["none_frames"]),
        ("counts", "interpolated_ball_rows", summary["interpolated_rows"]),
        ("counts", "low_ball_confidence_rows", summary["low_confidence_rows"]),
        ("distances", "assigned_distance_avg", f"{summary['assigned_distance_avg']:.2f}"),
        ("distances", "assigned_distance_max", f"{summary['assigned_distance_max']:.2f}"),
        ("percentages", "team_a_possession_percent", f"{summary['team_a_percent']:.2f}"),
        ("percentages", "team_b_possession_percent", f"{summary['team_b_percent']:.2f}"),
        ("percentages", "unknown_possession_percent", f"{summary['unknown_percent']:.2f}"),
        ("percentages", "no_possession_percent", f"{summary['none_percent']:.2f}"),
    ]
    for reason, count in sorted(summary["reasons"].items()):
        rows.append(("reasons", reason, count))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows({"section": section, "metric": metric, "value": value} for section, metric, value in rows)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        "\n".join(
            [
                "# Possession Summary",
                "",
                f"- Team A possession: {summary['team_a_percent']:.2f}%",
                f"- Team B possession: {summary['team_b_percent']:.2f}%",
                f"- Unknown possession: {summary['unknown_percent']:.2f}%",
                f"- Frames with no possession: {summary['none_frames']}",
                f"- No possession: {summary['none_percent']:.2f}%",
                f"- Interpolated ball rows: {summary['interpolated_rows']}",
                f"- Low ball-confidence rows: {summary['low_confidence_rows']}",
                f"- Assigned distance avg/max: {summary['assigned_distance_avg']:.2f} / {summary['assigned_distance_max']:.2f}",
                "",
                "## Assignment Reasons",
                "",
                *[f"- {reason}: {count}" for reason, count in sorted(summary["reasons"].items())],
            ]
        ),
        encoding="utf-8",
    )


def _write_qa_summary(
    summary: dict,
    qa_summary_md_path: Path,
    dynamic_distance_multiplier: float,
    min_track_confidence: float,
    min_ball_confidence: float,
    assign_interpolated: bool,
    switch_confirmation_frames: int,
    eligible_roles: set[str],
) -> None:
    qa_summary_md_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Possession QA Summary",
        "",
        "## Status",
        "",
        "Experimental baseline. The output is suitable for QA and reporting as a heuristic, not as validated possession analytics.",
        "",
        "## Current Gates",
        "",
        f"- dynamic_distance_multiplier: {dynamic_distance_multiplier}",
        f"- min_track_confidence: {min_track_confidence}",
        f"- min_ball_confidence: {min_ball_confidence}",
        f"- assign_interpolated: {assign_interpolated}",
        f"- switch_confirmation_frames: {switch_confirmation_frames}",
        f"- eligible_roles: {', '.join(sorted(eligible_roles))}",
        "",
        "## Results",
        "",
        f"- Team A possession: {summary['team_a_percent']:.2f}%",
        f"- Team B possession: {summary['team_b_percent']:.2f}%",
        f"- Unknown possession: {summary['unknown_percent']:.2f}%",
        f"- No possession: {summary['none_percent']:.2f}%",
        f"- Frames with no possession: {summary['none_frames']}",
        "",
        "## Root Cause Notes",
        "",
        "- Ball detection/tracking remains the main risk: false positives or jumps can place the tracked ball away from the real football.",
        "- Possession is distance-only in image coordinates, so a nearby player is not always the player controlling the ball.",
        "- Unknown, referee, and goalkeeper roles are excluded by default; this improves robustness but can miss real possession by those roles.",
        "- Interpolated ball points are excluded by default because they are tracker guesses rather than detector observations.",
        "",
        "## Reason Counts",
        "",
    ]
    lines.extend(f"- {reason}: {count}" for reason, count in sorted(summary["reasons"].items()))
    qa_summary_md_path.write_text("\n".join(lines), encoding="utf-8")


def _write_possession_video(
    video_path: Path,
    rows_by_frame: dict[int, dict],
    output_video_path: Path,
    debug: bool,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 30

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_video_path}")

    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break

        row = rows_by_frame.get(frame_index)
        if row is not None:
            _draw_possession_overlay(frame, frame_index, row, debug=debug)
        else:
            _draw_label_block(frame, [f"Frame {frame_index}", "Possession: None"], (80, 80, 80))
        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()


def _draw_possession_overlay(frame, frame_index: int, row: dict, debug: bool) -> None:
    team = row.get("team") or "None"
    color = _team_color(team)
    lines = [f"Frame {frame_index}", f"Possession: {team}"]
    if debug:
        lines.extend(
            [
                f"Candidate: {row.get('candidate_team', team)}",
                f"Distance: {row.get('distance_to_ball') or 'n/a'}",
                f"Threshold: {row.get('dynamic_distance_threshold') or 'n/a'}",
                f"Ball conf: {row.get('ball_confidence') or 'n/a'}",
                f"Interpolated: {row.get('is_interpolated') or 0}",
                f"Reason: {_short_reason(row.get('possession_reason', ''))}",
            ]
        )
        _draw_debug_line(frame, row)
    _draw_label_block(frame, lines, color)


def _draw_debug_line(frame, row: dict) -> None:
    if not row.get("nearest_player_center_x") or not row.get("distance_to_ball"):
        return
    try:
        player_center = (
            int(round(float(row["nearest_player_center_x"]))),
            int(round(float(row["nearest_player_center_y"]))),
        )
    except (TypeError, ValueError):
        return

    try:
        ball_center = (
            int(round(float(row["ball_center_x"]))),
            int(round(float(row["ball_center_y"]))),
        )
    except (KeyError, TypeError, ValueError):
        ball_center = None

    if ball_center is not None:
        cv2.line(frame, ball_center, player_center, (255, 255, 255), 2)
        cv2.circle(frame, ball_center, 7, _team_color(row.get("team", "None")), 2)
    cv2.circle(frame, player_center, 6, (255, 255, 255), 2)
    cv2.putText(
        frame,
        f"nearest ID {row.get('nearest_player_id')} d={row.get('distance_to_ball')}",
        (max(0, player_center[0] - 80), max(20, player_center[1] - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_label_block(frame, lines: list[str], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.48, frame.shape[0] / 1050)
    thickness = max(1, int(frame.shape[0] / 420))
    padding = 10
    line_height = int(24 * max(1.0, frame.shape[0] / 720))
    text_width = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines)
    block_height = line_height * len(lines) + padding
    x = padding
    y = padding
    cv2.rectangle(
        frame,
        (x - 4, y - 4),
        (x + text_width + 2 * padding, y + block_height),
        color,
        -1,
    )
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x + padding // 2, y + (index + 1) * line_height),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )


def _team_color(team: str) -> tuple[int, int, int]:
    return {
        "Team A": (0, 220, 255),
        "Team B": (255, 80, 80),
        "Unknown": (180, 180, 180),
        "None": (80, 80, 80),
    }.get(team, (80, 80, 80))


def _short_reason(reason: str) -> str:
    return reason.split(";")[0][:44] if reason else "n/a"


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _primary_reason(reason: str) -> str:
    return str(reason).split(";")[0].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate experimental baseline team possession")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--player-tracks-csv", type=Path, required=True)
    parser.add_argument("--teams-csv", type=Path, required=True)
    parser.add_argument("--ball-tracks-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/possession_30s_720p.csv"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/possession_summary_30s_720p.csv"))
    parser.add_argument("--summary-md", type=Path, default=Path("outputs/possession_summary_30s_720p.md"))
    parser.add_argument("--output-video", type=Path, default=Path("outputs/possession_30s_720p.mp4"))
    parser.add_argument("--debug-csv", type=Path, default=Path("outputs/possession_debug_30s_720p.csv"))
    parser.add_argument("--debug-video", type=Path, default=Path("outputs/possession_debug_30s_720p.mp4"))
    parser.add_argument("--qa-summary-md", type=Path, default=Path("outputs/possession_qa_summary.md"))
    parser.add_argument("--dynamic-distance-multiplier", type=float, default=1.5)
    parser.add_argument("--min-track-confidence", type=float, default=0.10)
    parser.add_argument("--min-ball-confidence", type=float, default=0.25)
    parser.add_argument(
        "--assign-interpolated",
        action="store_true",
        help="Allow interpolated ball points to assign possession. Default skips them.",
    )
    parser.add_argument("--switch-confirmation-frames", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    estimate_possession(
        player_tracks_csv_path=args.player_tracks_csv,
        teams_csv_path=args.teams_csv,
        ball_tracks_csv_path=args.ball_tracks_csv,
        output_csv_path=args.output_csv,
        summary_csv_path=args.summary_csv,
        summary_md_path=args.summary_md,
        debug_csv_path=args.debug_csv,
        qa_summary_md_path=args.qa_summary_md,
        dynamic_distance_multiplier=args.dynamic_distance_multiplier,
        min_track_confidence=args.min_track_confidence,
        min_ball_confidence=args.min_ball_confidence,
        assign_interpolated=args.assign_interpolated,
        switch_confirmation_frames=args.switch_confirmation_frames,
    )
    create_possession_video(
        video_path=args.video,
        possession_csv_path=args.output_csv,
        output_video_path=args.output_video,
    )
    create_possession_debug_video(
        video_path=args.video,
        debug_csv_path=args.debug_csv,
        output_video_path=args.debug_video,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
