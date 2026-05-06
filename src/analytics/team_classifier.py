"""Simple jersey-color team classification from tracked player boxes."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


TEAM_COLORS = {
    "Team A": (0, 220, 255),
    "Team B": (255, 80, 80),
}


def classify_teams(video_path, tracks_csv_path, output_csv_path) -> pd.DataFrame:
    """Classify tracked players into two teams using average jersey color."""
    video = Path(video_path)
    tracks_path = Path(tracks_csv_path)
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video}")
    if not tracks_path.exists():
        raise FileNotFoundError(f"Tracks CSV not found: {tracks_path}")

    tracks = _read_tracks(tracks_path)
    color_samples = _collect_track_colors(video, tracks)
    if color_samples.empty:
        raise ValueError("No valid player crops found for team classification")

    team_rows = _cluster_track_colors(color_samples)
    output = Path(output_csv_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    team_rows.to_csv(output, index=False)
    return team_rows


def create_team_video(video_path, tracks_csv_path, teams_csv_path, output_video_path) -> None:
    """Draw team-colored tracked boxes and labels on a video."""
    video = Path(video_path)
    tracks = _read_tracks(tracks_csv_path)
    teams = _read_teams(teams_csv_path)
    team_by_track_id = dict(zip(teams["track_id"], teams["team"]))

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 30

    output = Path(output_video_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output}")

    tracks_by_frame = {frame: group for frame, group in tracks.groupby("frame")}
    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_tracks = tracks_by_frame.get(frame_index)
        if frame_tracks is not None:
            _draw_team_tracks(frame, frame_tracks, team_by_track_id)

        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()


def generate_team_heatmaps(
    tracks_csv_path,
    teams_csv_path,
    team_a_output_path,
    team_b_output_path,
    frame_width=398,
    frame_height=224,
) -> None:
    """Generate one movement heatmap for each classified team."""
    tracks = _read_tracks(tracks_csv_path)
    teams = _read_teams(teams_csv_path)
    team_by_track_id = dict(zip(teams["track_id"], teams["team"]))

    tracks = tracks.copy()
    tracks["team"] = tracks["track_id"].map(team_by_track_id)
    _generate_team_heatmap(
        tracks[tracks["team"] == "Team A"],
        team_a_output_path,
        frame_width,
        frame_height,
        "Team A Heatmap",
    )
    _generate_team_heatmap(
        tracks[tracks["team"] == "Team B"],
        team_b_output_path,
        frame_width,
        frame_height,
        "Team B Heatmap",
    )


def _read_tracks(tracks_csv_path) -> pd.DataFrame:
    tracks = pd.read_csv(tracks_csv_path)
    required_columns = {
        "frame",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
    }
    missing_columns = required_columns - set(tracks.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Tracks CSV missing required column(s): {columns}")

    tracks = tracks.dropna(subset=list(required_columns)).copy()
    tracks["frame"] = tracks["frame"].astype(int)
    tracks["track_id"] = tracks["track_id"].astype(int)
    return tracks


def _read_teams(teams_csv_path) -> pd.DataFrame:
    teams = pd.read_csv(teams_csv_path)
    required_columns = {"track_id", "team"}
    missing_columns = required_columns - set(teams.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Teams CSV missing required column(s): {columns}")

    teams = teams.dropna(subset=["track_id", "team"]).copy()
    teams["track_id"] = teams["track_id"].astype(int)
    return teams


def _collect_track_colors(video_path: Path, tracks: pd.DataFrame) -> pd.DataFrame:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracks_by_frame = {frame: group for frame, group in tracks.groupby("frame")}
    samples = []
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_tracks = tracks_by_frame.get(frame_index)
        if frame_tracks is not None:
            for _, row in frame_tracks.iterrows():
                color = _average_jersey_color(row, frame, frame_width, frame_height)
                if color is None:
                    continue
                samples.append(
                    {
                        "track_id": int(row["track_id"]),
                        "r": color[0],
                        "g": color[1],
                        "b": color[2],
                    }
                )

        frame_index += 1

    capture.release()
    return pd.DataFrame(samples, columns=["track_id", "r", "g", "b"])


def _average_jersey_color(row, frame, frame_width: int, frame_height: int):
    x1 = int(round(row["x1"]))
    y1 = int(round(row["y1"]))
    x2 = int(round(row["x2"]))
    y2 = int(round(row["y2"]))

    x1 = max(0, min(x1, frame_width - 1))
    x2 = max(0, min(x2, frame_width))
    y1 = max(0, min(y1, frame_height - 1))
    y2 = max(0, min(y2, frame_height))
    if x2 <= x1 or y2 <= y1:
        return None

    upper_body_bottom = y1 + max(1, (y2 - y1) // 2)
    crop = frame[y1:upper_body_bottom, x1:x2]
    if crop.size == 0:
        return None

    crop_hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = crop_hsv[:, :, 1]
    value = crop_hsv[:, :, 2]
    mask = (saturation > 30) & (value > 40)
    if mask.sum() < 5:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mean_rgb = crop_rgb[mask].mean(axis=0)
    return tuple(float(value) for value in mean_rgb)


def _cluster_track_colors(color_samples: pd.DataFrame) -> pd.DataFrame:
    track_colors = (
        color_samples.groupby("track_id")
        .agg(avg_r=("r", "mean"), avg_g=("g", "mean"), avg_b=("b", "mean"), frames_used=("r", "count"))
        .reset_index()
    )
    if len(track_colors) < 2:
        track_colors["team"] = "Team A"
        return track_colors[["track_id", "team", "avg_r", "avg_g", "avg_b", "frames_used"]]

    features = track_colors[["avg_r", "avg_g", "avg_b"]].to_numpy()
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    centers = kmeans.cluster_centers_
    brightness = centers.mean(axis=1)
    ordered_labels = list(np.argsort(brightness))
    team_names = {
        ordered_labels[0]: "Team A",
        ordered_labels[1]: "Team B",
    }
    track_colors["team"] = [team_names[label] for label in labels]

    for column in ["avg_r", "avg_g", "avg_b"]:
        track_colors[column] = track_colors[column].round(2)
    return track_colors[
        ["track_id", "team", "avg_r", "avg_g", "avg_b", "frames_used"]
    ].sort_values(["team", "track_id"])


def _draw_team_tracks(frame, tracks: pd.DataFrame, team_by_track_id: dict) -> None:
    frame_height, frame_width = frame.shape[:2]
    scale = frame_height / 720
    font_scale = max(0.25, 0.45 * scale)
    thickness = max(1, int(2 * scale))
    padding = max(2, int(4 * scale))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for _, row in tracks.iterrows():
        track_id = int(row["track_id"])
        team = team_by_track_id.get(track_id)
        if team is None:
            continue

        x1 = max(0, min(int(round(row["x1"])), frame_width - 1))
        y1 = max(0, min(int(round(row["y1"])), frame_height - 1))
        x2 = max(0, min(int(round(row["x2"])), frame_width - 1))
        y2 = max(0, min(int(round(row["y2"])), frame_height - 1))
        color = TEAM_COLORS.get(team, (255, 255, 255))
        label = f"{team} ID {track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        _draw_label(frame, label, x1, y1, color, font, font_scale, thickness, padding)


def _draw_label(frame, label, x1, y1, color, font, font_scale, thickness, padding) -> None:
    frame_height, frame_width = frame.shape[:2]
    label_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_width, text_height = label_size
    label_x = max(0, min(x1, frame_width - text_width - 2 * padding))
    label_y = y1 - padding
    if label_y - text_height - baseline - padding < 0:
        label_y = y1 + text_height + baseline + padding
    label_y = min(label_y, frame_height - baseline - padding)

    background_top = max(0, label_y - text_height - baseline - padding)
    background_bottom = min(frame_height, label_y + baseline + padding)
    background_right = min(frame_width, label_x + text_width + 2 * padding)

    cv2.rectangle(
        frame,
        (label_x, background_top),
        (background_right, background_bottom),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (label_x + padding, label_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def _generate_team_heatmap(
    tracks: pd.DataFrame,
    output_path,
    frame_width: int,
    frame_height: int,
    title: str,
) -> None:
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
    _draw_title(overlay, title)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), overlay):
        raise RuntimeError(f"Could not write team heatmap image: {output}")


def _draw_field_background(width: int, height: int):
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


def main() -> None:
    """Parse CLI arguments and run color-based team analytics."""
    parser = argparse.ArgumentParser(
        description="Classify teams using simple jersey-color clustering"
    )
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--tracks-csv", required=True, help="Path to the tracking CSV")
    parser.add_argument("--output", required=True, help="Path to save player teams CSV")
    parser.add_argument(
        "--team-video-output",
        help="Optional path to save team-colored tracked video",
    )
    parser.add_argument(
        "--team-a-heatmap",
        help="Optional path to save Team A heatmap",
    )
    parser.add_argument(
        "--team-b-heatmap",
        help="Optional path to save Team B heatmap",
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=398,
        help="Team heatmap width in pixels",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=224,
        help="Team heatmap height in pixels",
    )
    args = parser.parse_args()

    try:
        classify_teams(
            video_path=args.video,
            tracks_csv_path=args.tracks_csv,
            output_csv_path=args.output,
        )
        print(f"Player teams saved to: {args.output}")

        if args.team_video_output:
            create_team_video(
                video_path=args.video,
                tracks_csv_path=args.tracks_csv,
                teams_csv_path=args.output,
                output_video_path=args.team_video_output,
            )
            print(f"Team-colored video saved to: {args.team_video_output}")

        if args.team_a_heatmap and args.team_b_heatmap:
            generate_team_heatmaps(
                tracks_csv_path=args.tracks_csv,
                teams_csv_path=args.output,
                team_a_output_path=args.team_a_heatmap,
                team_b_output_path=args.team_b_heatmap,
                frame_width=args.frame_width,
                frame_height=args.frame_height,
            )
            print(f"Team A heatmap saved to: {args.team_a_heatmap}")
            print(f"Team B heatmap saved to: {args.team_b_heatmap}")
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return


if __name__ == "__main__":
    main()
