"""Generate player movement heatmaps from tracking CSV files."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def generate_heatmap(
    tracks_csv_path,
    output_path,
    frame_width,
    frame_height,
    track_id=None,
):
    """Generate a football-style movement heatmap from tracked centroids."""
    tracks_path = Path(tracks_csv_path)
    if not tracks_path.exists():
        raise FileNotFoundError(f"Tracks CSV not found: {tracks_path}")

    tracks = pd.read_csv(tracks_path)
    required_columns = {"center_x", "center_y", "track_id"}
    missing_columns = required_columns - set(tracks.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Tracks CSV missing required column(s): {columns}")

    frame_width, frame_height = _resolve_frame_size(tracks, frame_width, frame_height)

    if track_id is not None:
        track_id = int(track_id)
        tracks = tracks[tracks["track_id"] == track_id]

    density = np.zeros((frame_height, frame_width), dtype=np.float32)
    for center_x, center_y in tracks[["center_x", "center_y"]].dropna().to_numpy():
        x = int(round(center_x))
        y = int(round(center_y))
        if 0 <= x < frame_width and 0 <= y < frame_height:
            density[y, x] += 1

    if density.max() > 0:
        blur_size = _odd_kernel_size(min(frame_width, frame_height) // 18)
        density = cv2.GaussianBlur(density, (blur_size, blur_size), 0)
        density = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX)

    heatmap = cv2.applyColorMap(density.astype(np.uint8), cv2.COLORMAP_JET)
    field = _draw_field_background(frame_width, frame_height)
    overlay = cv2.addWeighted(field, 0.65, heatmap, 0.55, 0)

    title = f"Player {track_id} Heatmap" if track_id is not None else "Player Movement Heatmap"
    _draw_title(overlay, title)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), overlay):
        raise RuntimeError(f"Could not write heatmap image: {output}")


def _draw_field_background(width: int, height: int):
    """Draw a dark football-field-style background at frame scale."""
    field = np.full((height, width, 3), (28, 75, 42), dtype=np.uint8)
    line_color = (175, 210, 185)
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

    goal_depth = max(6, int(width * 0.035))
    goal_half_height = max(8, int(height * 0.09))
    cv2.rectangle(
        field,
        (left, center_y - goal_half_height),
        (left + goal_depth, center_y + goal_half_height),
        line_color,
        thickness,
    )
    cv2.rectangle(
        field,
        (right - goal_depth, center_y - goal_half_height),
        (right, center_y + goal_half_height),
        line_color,
        thickness,
    )

    return field


def _draw_title(image, title: str) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width = image.shape[:2]
    font_scale = max(0.5, height / 900)
    thickness = max(1, int(height / 360))
    padding = max(8, int(height * 0.02))

    text_size, baseline = cv2.getTextSize(title, font, font_scale, thickness)
    text_width, text_height = text_size
    x = max(padding, (width - text_width) // 2)
    y = padding + text_height

    cv2.rectangle(
        image,
        (x - padding, y - text_height - padding),
        (x + text_width + padding, y + baseline + padding),
        (12, 32, 20),
        -1,
    )
    cv2.putText(image, title, (x, y), font, font_scale, (245, 245, 245), thickness, cv2.LINE_AA)


def _odd_kernel_size(value: int) -> int:
    value = max(15, int(value))
    return value if value % 2 == 1 else value + 1


def _resolve_frame_size(tracks: pd.DataFrame, frame_width, frame_height) -> tuple[int, int]:
    if frame_width is not None and frame_height is not None:
        frame_width = int(frame_width)
        frame_height = int(frame_height)
    else:
        width_source = "x2" if "x2" in tracks.columns else "center_x"
        height_source = "y2" if "y2" in tracks.columns else "center_y"
        frame_width = int(np.ceil(tracks[width_source].max())) + 1
        frame_height = int(np.ceil(tracks[height_source].max())) + 1

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame_width and frame_height must be positive")
    return frame_width, frame_height


def main() -> None:
    """Parse CLI arguments and generate a heatmap from an existing tracks CSV."""
    parser = argparse.ArgumentParser(
        description="Generate a player movement heatmap from a tracks CSV"
    )
    parser.add_argument(
        "--tracks-csv",
        required=True,
        help="Path to the tracking CSV file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the generated heatmap image",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        help="Output heatmap width in pixels. Inferred from tracks when omitted.",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        help="Output heatmap height in pixels. Inferred from tracks when omitted.",
    )
    parser.add_argument(
        "--track-id",
        type=int,
        help="Optional track ID for a single-player heatmap",
    )
    args = parser.parse_args()

    try:
        generate_heatmap(
            tracks_csv_path=args.tracks_csv,
            output_path=args.output,
            frame_width=args.frame_width,
            frame_height=args.frame_height,
            track_id=args.track_id,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return

    print(f"Heatmap saved to: {args.output}")


if __name__ == "__main__":
    main()
