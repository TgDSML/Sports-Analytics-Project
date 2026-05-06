"""Generate player trajectory plots from tracking CSV files."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def generate_trajectory_plot(
    tracks_csv_path,
    output_path,
    frame_width,
    frame_height,
    track_id=None,
):
    """Generate a football-style trajectory plot from tracked centroids."""
    tracks_path = Path(tracks_csv_path)
    if not tracks_path.exists():
        raise FileNotFoundError(f"Tracks CSV not found: {tracks_path}")

    frame_width = int(frame_width)
    frame_height = int(frame_height)
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be positive")

    tracks = pd.read_csv(tracks_path)
    required_columns = {"center_x", "center_y", "track_id"}
    missing_columns = required_columns - set(tracks.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Tracks CSV missing required column(s): {columns}")

    if track_id is not None:
        track_id = int(track_id)
        tracks = tracks[tracks["track_id"] == track_id]

    sort_columns = [column for column in ("track_id", "frame", "timestamp") if column in tracks.columns]
    if sort_columns:
        tracks = tracks.sort_values(sort_columns)

    canvas = _draw_field_background(frame_width, frame_height)
    colors = _track_colors(tracks["track_id"].dropna().unique())

    for current_track_id, group in tracks.groupby("track_id"):
        points = _valid_points(group, frame_width, frame_height)
        if len(points) == 0:
            continue

        color = colors.get(current_track_id, (255, 255, 255))
        thickness = max(1, int(min(frame_width, frame_height) / 150))

        if len(points) > 1:
            cv2.polylines(canvas, [points], False, color, thickness, cv2.LINE_AA)

        cv2.circle(canvas, tuple(points[0]), max(2, thickness + 1), color, -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(points[-1]), max(3, thickness + 2), color, -1, cv2.LINE_AA)
        _draw_track_label(canvas, str(int(current_track_id)), tuple(points[-1]), color)

    title = (
        f"Player {track_id} Trajectory"
        if track_id is not None
        else "Player Trajectories"
    )
    _draw_title(canvas, title)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Could not write trajectory image: {output}")


def _valid_points(group, width: int, height: int) -> np.ndarray:
    points = []
    for center_x, center_y in group[["center_x", "center_y"]].dropna().to_numpy():
        x = int(round(center_x))
        y = int(round(center_y))
        if 0 <= x < width and 0 <= y < height:
            points.append((x, y))
    return np.array(points, dtype=np.int32)


def _track_colors(track_ids) -> dict:
    colors = {}
    sorted_ids = sorted(int(track_id) for track_id in track_ids)
    count = max(1, len(sorted_ids))
    for index, track_id in enumerate(sorted_ids):
        hue = int((index * 179) / count)
        hsv = np.uint8([[[hue, 210, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors[track_id] = tuple(int(value) for value in bgr)
    return colors


def _draw_field_background(width: int, height: int):
    field = np.full((height, width, 3), (24, 68, 38), dtype=np.uint8)
    line_color = (170, 205, 182)
    thickness = max(1, int(min(width, height) / 220))

    margin_x = max(8, int(width * 0.04))
    margin_y = max(8, int(height * 0.07))
    left = margin_x
    right = width - margin_x
    top = margin_y
    bottom = height - margin_y
    center_x = width // 2
    center_y = height // 2

    cv2.rectangle(field, (left, top), (right, bottom), line_color, thickness)
    cv2.line(field, (center_x, top), (center_x, bottom), line_color, thickness)
    cv2.circle(field, (center_x, center_y), max(8, int(height * 0.12)), line_color, thickness)

    box_depth = max(12, int(width * 0.12))
    box_half_height = max(12, int(height * 0.22))
    cv2.rectangle(
        field,
        (left, center_y - box_half_height),
        (left + box_depth, center_y + box_half_height),
        line_color,
        thickness,
    )
    cv2.rectangle(
        field,
        (right - box_depth, center_y - box_half_height),
        (right, center_y + box_half_height),
        line_color,
        thickness,
    )
    return field


def _draw_title(image, title: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width = image.shape[:2]
    font_scale = max(0.45, height / 950)
    thickness = max(1, int(height / 360))
    padding = max(6, int(height * 0.018))

    text_size, baseline = cv2.getTextSize(title, font, font_scale, thickness)
    text_width, text_height = text_size
    x = max(padding, (width - text_width) // 2)
    y = padding + text_height

    cv2.rectangle(
        image,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        (10, 28, 18),
        -1,
    )
    cv2.putText(image, title, (x, y), font, font_scale, (245, 245, 245), thickness, cv2.LINE_AA)


def _draw_track_label(image, label: str, point: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width = image.shape[:2]
    font_scale = max(0.35, height / 1000)
    thickness = 1
    padding = 2
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_width, text_height = text_size

    x = min(max(point[0] + 3, 0), max(0, width - text_width - 2 * padding))
    y = min(max(point[1] - 3, text_height + padding), height - baseline - padding)

    cv2.rectangle(
        image,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        (8, 24, 16),
        -1,
    )
    cv2.putText(image, label, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def main() -> None:
    """Parse CLI arguments and generate trajectories from an existing tracks CSV."""
    parser = argparse.ArgumentParser(
        description="Generate player trajectory plots from a tracks CSV"
    )
    parser.add_argument("--tracks-csv", required=True, help="Path to the tracking CSV file")
    parser.add_argument("--output", required=True, help="Path to save the trajectory image")
    parser.add_argument(
        "--frame-width",
        type=int,
        default=398,
        help="Output trajectory width in pixels",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=224,
        help="Output trajectory height in pixels",
    )
    parser.add_argument("--track-id", type=int, help="Optional track ID for one player")
    args = parser.parse_args()

    try:
        generate_trajectory_plot(
            tracks_csv_path=args.tracks_csv,
            output_path=args.output,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            track_id=args.track_id,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return

    print(f"Trajectory plot saved to: {args.output}")


if __name__ == "__main__":
    main()
