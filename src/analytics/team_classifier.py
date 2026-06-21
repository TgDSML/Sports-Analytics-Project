"""Simple jersey-color team classification from tracked player boxes."""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


TEAM_COLORS = {
    "Team A": (0, 220, 255),
    "Team B": (255, 80, 80),
    "Unknown": (180, 180, 180),
    "Other": (180, 180, 180),
}
ROLE_COLORS = {
    "team_a_player": (0, 220, 255),
    "team_b_player": (255, 80, 80),
    "team_a_goalkeeper": (40, 170, 255),
    "team_b_goalkeeper": (255, 170, 40),
    "goalkeeper_left": (40, 170, 255),
    "goalkeeper_right": (255, 170, 40),
    "referee": (255, 255, 255),
    "unknown": (180, 180, 180),
}
TEAM_LABELS = ("Team A", "Team B", "Unknown")
ROLE_LABELS = (
    "team_a_player",
    "team_b_player",
    "team_a_goalkeeper",
    "team_b_goalkeeper",
    "goalkeeper_left",
    "goalkeeper_right",
    "referee",
    "unknown",
)
TORSO_X_MARGIN = 0.20
TORSO_Y_TOP = 0.15
TORSO_Y_BOTTOM = 0.55
MIN_BOX_WIDTH = 18
MIN_BOX_HEIGHT = 36
MIN_VALID_PIXELS = 18
MIN_VALID_FRACTION = 0.08
MAX_GREEN_FRACTION = 0.55
MAX_BOX_WIDTH_FRACTION = 0.22
MAX_BOX_HEIGHT_FRACTION = 0.38
MAX_TORSO_AREA_FRACTION = 0.0075
MIN_TRACK_SAMPLES = 8
WEAK_TRACK_MIN_SAMPLES = 1
OUTLIER_MAD_MULTIPLIER = 3.5
MIN_DISTANCE_CUTOFF = 1.15
MAX_DISTANCE_CUTOFF = 2.75
MIN_CLUSTER_MARGIN = 0.05
FEATURE_COLUMNS = ["lab_a", "lab_b", "lab_l"]
DEFAULT_ROLE_CLUSTERS = 5
MIN_GOALKEEPER_TRACK_LENGTH = 60
REFEREE_DISTANCE_THRESHOLD = 2.1
GOAL_AREA_X_FRACTION = 0.18
CENTRAL_X_MIN = 0.25
CENTRAL_X_MAX = 0.75


def classify_teams(
    video_path,
    tracks_csv_path,
    output_csv_path,
    num_clusters: int = 2,
    debug_dir=None,
    min_track_samples: int = MIN_TRACK_SAMPLES,
    detect_roles: bool = False,
    role_clusters: int = DEFAULT_ROLE_CLUSTERS,
    min_goalkeeper_track_length: int = MIN_GOALKEEPER_TRACK_LENGTH,
    referee_distance_threshold: float = REFEREE_DISTANCE_THRESHOLD,
) -> pd.DataFrame:
    """Classify tracked players into teams using aggregated torso jersey color."""
    video = Path(video_path)
    tracks_path = Path(tracks_csv_path)
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video}")
    if not tracks_path.exists():
        raise FileNotFoundError(f"Tracks CSV not found: {tracks_path}")

    debug_path = Path(debug_dir) if debug_dir else None
    if debug_path is not None:
        _prepare_debug_dir(debug_path)

    tracks = _read_tracks(tracks_path)
    color_samples = _collect_track_colors(video, tracks, debug_path)
    if color_samples.empty:
        raise ValueError("No valid player torso crops found for team classification")
    if int(num_clusters) != 2:
        raise ValueError("Team classification currently supports exactly 2 clusters")

    team_rows = _cluster_track_colors(
        color_samples,
        num_clusters=num_clusters,
        min_track_samples=min_track_samples,
    )
    cluster_centers = team_rows.attrs.get("cluster_centers")
    team_rows = _add_missing_unknown_tracks(team_rows, tracks["track_id"].unique())
    if cluster_centers is not None:
        team_rows.attrs["cluster_centers"] = cluster_centers
    team_rows = _add_track_position_stats(team_rows, tracks)
    team_rows["samples"] = team_rows["frames_used"]
    if detect_roles:
        team_rows = _assign_roles(
            team_rows,
            frame_width=int(tracks["x2"].max()),
            frame_height=int(tracks["y2"].max()),
            role_clusters=role_clusters,
            min_goalkeeper_track_length=min_goalkeeper_track_length,
            referee_distance_threshold=referee_distance_threshold,
        )
    else:
        team_rows = _assign_default_roles(team_rows)

    output = Path(output_csv_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    team_rows.to_csv(output, index=False)

    if debug_path is not None:
        _write_debug_outputs(team_rows, debug_path)
    return team_rows


def create_team_video(video_path, tracks_csv_path, teams_csv_path, output_video_path) -> None:
    """Draw team-colored tracked boxes and labels on a video."""
    video = Path(video_path)
    tracks = _read_tracks(tracks_csv_path)
    teams = _read_teams(teams_csv_path)
    role_by_track_id = dict(zip(teams["track_id"], teams["role"]))

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
            _draw_team_tracks(frame, frame_tracks, role_by_track_id)

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
    include_goalkeepers: bool = False,
) -> None:
    """Generate one movement heatmap for each outfield team by default."""
    tracks = _read_tracks(tracks_csv_path)
    teams = _read_teams(teams_csv_path)
    role_by_track_id = dict(zip(teams["track_id"], teams["role"]))

    tracks = tracks.copy()
    tracks["role"] = tracks["track_id"].map(role_by_track_id).fillna("unknown")
    team_a_roles = {"team_a_player"}
    team_b_roles = {"team_b_player"}
    if include_goalkeepers:
        team_a_roles.add("team_a_goalkeeper")
        team_b_roles.add("team_b_goalkeeper")
    _generate_team_heatmap(
        tracks[tracks["role"].isin(team_a_roles)],
        team_a_output_path,
        frame_width,
        frame_height,
        "Team A Outfield Heatmap",
    )
    _generate_team_heatmap(
        tracks[tracks["role"].isin(team_b_roles)],
        team_b_output_path,
        frame_width,
        frame_height,
        "Team B Outfield Heatmap",
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
    required_columns = {"track_id", "team", "cluster_id"}
    missing_columns = required_columns - set(teams.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Teams CSV missing required column(s): {columns}")

    teams = teams.dropna(subset=["track_id", "team"]).copy()
    teams["track_id"] = teams["track_id"].astype(int)
    teams["cluster_id"] = teams["cluster_id"].astype(int)
    if "role" not in teams.columns:
        teams = _assign_default_roles(teams)
    return teams


def _collect_track_colors(
    video_path: Path,
    tracks: pd.DataFrame,
    debug_dir: Path | None,
) -> pd.DataFrame:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracks_by_frame = {frame: group for frame, group in tracks.groupby("frame")}
    representative_crops = {}
    samples = []
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_tracks = tracks_by_frame.get(frame_index)
        if frame_tracks is not None:
            for _, row in frame_tracks.iterrows():
                sample = _extract_jersey_sample(row, frame, frame_width, frame_height)
                if sample is None:
                    continue
                sample["frame"] = frame_index
                samples.append(sample)
                _keep_representative_crop(representative_crops, sample)

        frame_index += 1

    capture.release()

    if debug_dir is not None:
        _write_representative_crops(representative_crops, debug_dir / "crops")

    return pd.DataFrame(samples)


def _extract_jersey_sample(row, frame, frame_width: int, frame_height: int):
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

    box_width = x2 - x1
    box_height = y2 - y1
    if box_width < MIN_BOX_WIDTH or box_height < MIN_BOX_HEIGHT:
        return None
    if box_width > frame_width * MAX_BOX_WIDTH_FRACTION:
        return None
    if box_height > frame_height * MAX_BOX_HEIGHT_FRACTION:
        return None

    crop_x1 = x1 + int(round(box_width * TORSO_X_MARGIN))
    crop_x2 = x2 - int(round(box_width * TORSO_X_MARGIN))
    crop_y1 = y1 + int(round(box_height * TORSO_Y_TOP))
    crop_y2 = y1 + int(round(box_height * TORSO_Y_BOTTOM))
    crop_x1 = max(x1, min(crop_x1, x2 - 1))
    crop_x2 = max(crop_x1 + 1, min(crop_x2, x2))
    crop_y1 = max(y1, min(crop_y1, y2 - 1))
    crop_y2 = max(crop_y1 + 1, min(crop_y2, y2))

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return None
    crop_area = int(crop.shape[0] * crop.shape[1])
    if crop_area > frame_width * frame_height * MAX_TORSO_AREA_FRACTION:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    green = (hue >= 35) & (hue <= 85) & (saturation > 35) & (value > 35)
    green_fraction = float(green.mean())
    if green_fraction > MAX_GREEN_FRACTION:
        return None

    jersey_mask = (saturation > 28) & (value > 38) & (value < 248) & ~green
    valid_pixels = int(jersey_mask.sum())
    valid_fraction = valid_pixels / crop_area
    if valid_pixels < MIN_VALID_PIXELS or valid_fraction < MIN_VALID_FRACTION:
        return None

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hsv_values = hsv[jersey_mask]
    rgb_values = rgb[jersey_mask]
    lab_values = lab[jersey_mask]
    mean_rgb = rgb_values.mean(axis=0)
    median_rgb = np.median(rgb_values, axis=0)
    median_hsv = np.median(hsv_values, axis=0)
    median_lab = np.median(lab_values, axis=0)

    return {
        "track_id": int(row["track_id"]),
        "confidence": float(row.get("confidence", 0.0)),
        "r": float(median_rgb[0]),
        "g": float(median_rgb[1]),
        "b": float(median_rgb[2]),
        "mean_r": float(mean_rgb[0]),
        "mean_g": float(mean_rgb[1]),
        "mean_b": float(mean_rgb[2]),
        "h": float(median_hsv[0]),
        "s": float(median_hsv[1]),
        "v": float(median_hsv[2]),
        "lab_l": float(median_lab[0]),
        "lab_a": float(median_lab[1]),
        "lab_b": float(median_lab[2]),
        "green_fraction": green_fraction,
        "valid_pixels": valid_pixels,
        "valid_fraction": float(valid_fraction),
        "crop_area": crop_area,
        "box_width": box_width,
        "box_height": box_height,
        "crop": crop.copy(),
    }


def _keep_representative_crop(representative_crops: dict, sample: dict) -> None:
    track_id = sample["track_id"]
    previous = representative_crops.get(track_id)
    if previous is None or sample["valid_pixels"] > previous["valid_pixels"]:
        representative_crops[track_id] = sample


def _cluster_track_colors(
    color_samples: pd.DataFrame,
    num_clusters: int = 2,
    min_track_samples: int = MIN_TRACK_SAMPLES,
) -> pd.DataFrame:
    track_colors = (
        color_samples.groupby("track_id")
        .agg(
            avg_r=("r", "median"),
            avg_g=("g", "median"),
            avg_b=("b", "median"),
            hsv_h=("h", "median"),
            hsv_s=("s", "median"),
            hsv_v=("v", "median"),
            lab_l=("lab_l", "median"),
            lab_a=("lab_a", "median"),
            lab_b=("lab_b", "median"),
            frames_used=("r", "count"),
            avg_confidence=("confidence", "median"),
            green_fraction=("green_fraction", "mean"),
            valid_fraction=("valid_fraction", "median"),
            valid_pixels=("valid_pixels", "median"),
            crop_area=("crop_area", "median"),
            box_width=("box_width", "median"),
            box_height=("box_height", "median"),
        )
        .reset_index()
    )
    track_colors["team"] = "Unknown"
    track_colors["cluster_id"] = -1
    track_colors["cluster_distance"] = np.nan
    track_colors["cluster_margin"] = np.nan
    track_colors["assignment_confidence"] = 0.0
    track_colors["assignment_reason"] = "insufficient_samples"

    reliable = _reliable_fit_mask(track_colors, int(min_track_samples))
    if reliable.sum() < 2:
        weak = track_colors["frames_used"] >= WEAK_TRACK_MIN_SAMPLES
        track_colors.loc[weak, "team"] = "Team A"
        track_colors.loc[weak, "cluster_id"] = 0
        track_colors.loc[weak, "assignment_reason"] = "fallback_single_cluster"
        track_colors.loc[weak, "assignment_confidence"] = 0.5
        return _format_team_rows(track_colors)

    fit_rows = track_colors.loc[reliable].copy()
    scaler = _fit_robust_scaler(fit_rows[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    fit_features = _transform_features(
        fit_rows[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        scaler,
    )
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=30)
    fit_labels = kmeans.fit_predict(fit_features)
    fit_distances = pairwise_distances(fit_features, kmeans.cluster_centers_)
    keep_fit = _inlier_mask(fit_distances, fit_labels)

    if keep_fit.sum() >= 2 and len(set(fit_labels[keep_fit])) == 2:
        fit_features = fit_features[keep_fit]
        fit_rows = fit_rows.iloc[np.flatnonzero(keep_fit)].copy()
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=30)
        fit_labels = kmeans.fit_predict(fit_features)

    team_names = _assign_two_team_names(fit_rows, fit_labels)
    all_features = _transform_features(
        track_colors[FEATURE_COLUMNS].to_numpy(dtype=np.float32),
        scaler,
    )
    all_distances = pairwise_distances(all_features, kmeans.cluster_centers_)
    nearest = all_distances.argmin(axis=1)
    nearest_distance = all_distances[np.arange(len(track_colors)), nearest]
    sorted_distances = np.sort(all_distances, axis=1)
    margins = sorted_distances[:, 1] - sorted_distances[:, 0]
    distance_cutoffs = _cluster_distance_cutoffs(fit_features, fit_labels, kmeans.cluster_centers_)

    for row_position, row_index in enumerate(track_colors.index):
        label = int(nearest[row_position])
        distance = float(nearest_distance[row_position])
        margin = float(margins[row_position])
        frames_used = int(track_colors.at[row_index, "frames_used"])
        cutoff = distance_cutoffs[label]

        track_colors.at[row_index, "cluster_id"] = label
        track_colors.at[row_index, "cluster_distance"] = distance
        track_colors.at[row_index, "cluster_margin"] = margin
        track_colors.at[row_index, "assignment_confidence"] = _assignment_confidence(
            distance,
            margin,
            cutoff,
            frames_used,
        )

        if frames_used < WEAK_TRACK_MIN_SAMPLES:
            continue
        if distance > cutoff:
            track_colors.at[row_index, "assignment_reason"] = "far_from_team_centroid"
            track_colors.at[row_index, "cluster_id"] = -1
            continue
        if frames_used < int(min_track_samples) and margin < MIN_CLUSTER_MARGIN:
            track_colors.at[row_index, "assignment_reason"] = "ambiguous_weak_track"
            track_colors.at[row_index, "cluster_id"] = -1
            continue

        track_colors.at[row_index, "team"] = team_names[label]
        track_colors.at[row_index, "assignment_reason"] = (
            "reliable_track" if frames_used >= int(min_track_samples) else "weak_track_near_centroid"
        )

    result = _format_team_rows(track_colors)
    result.attrs["cluster_centers"] = _build_cluster_center_rows(
        fit_rows,
        fit_labels,
        team_names,
        distance_cutoffs,
    )
    return result


def _reliable_fit_mask(track_colors: pd.DataFrame, min_track_samples: int) -> pd.Series:
    return (
        (track_colors["frames_used"] >= min_track_samples)
        & (track_colors["valid_fraction"] >= MIN_VALID_FRACTION)
        & (track_colors["green_fraction"] <= MAX_GREEN_FRACTION)
        & (track_colors["crop_area"] <= track_colors["crop_area"].quantile(0.90))
    )


def _fit_robust_scaler(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(features, axis=0)
    mad = np.median(np.abs(features - median), axis=0)
    scale = 1.4826 * mad
    scale[scale < 1e-6] = features.std(axis=0)[scale < 1e-6]
    scale[scale < 1e-6] = 1.0
    return median, scale


def _transform_features(
    features: np.ndarray,
    scaler: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    median, scale = scaler
    transformed = (features - median) / scale
    transformed[:, 2] *= 0.65
    return transformed


def _inlier_mask(distances: np.ndarray, labels: np.ndarray) -> np.ndarray:
    mask = np.ones(len(labels), dtype=bool)
    for label in sorted(set(int(value) for value in labels)):
        label_distances = distances[labels == label, label]
        if len(label_distances) < 4:
            continue
        median_distance = float(np.median(label_distances))
        mad = float(np.median(np.abs(label_distances - median_distance)))
        cutoff = median_distance + max(MIN_DISTANCE_CUTOFF, OUTLIER_MAD_MULTIPLIER * mad)
        mask[labels == label] = label_distances <= cutoff
    return mask


def _cluster_distance_cutoffs(
    fit_features: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
) -> dict[int, float]:
    distances = pairwise_distances(fit_features, centers)
    cutoffs = {}
    for label in sorted(set(int(value) for value in labels)):
        label_distances = distances[labels == label, label]
        median_distance = float(np.median(label_distances))
        mad = float(np.median(np.abs(label_distances - median_distance)))
        cutoff = median_distance + max(MIN_DISTANCE_CUTOFF, OUTLIER_MAD_MULTIPLIER * mad)
        cutoffs[label] = min(MAX_DISTANCE_CUTOFF, max(MIN_DISTANCE_CUTOFF, cutoff))
    return cutoffs


def _assign_two_team_names(track_colors: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    labeled = track_colors.copy()
    labeled["cluster_id"] = labels
    centers = labeled.groupby("cluster_id")[["lab_a", "lab_b", "lab_l"]].median()
    ordered = centers.sort_values(["lab_b", "lab_a", "lab_l"], ascending=[False, True, False]).index.tolist()
    return {int(ordered[0]): "Team A", int(ordered[1]): "Team B"}


def _assignment_confidence(
    distance: float,
    margin: float,
    cutoff: float,
    frames_used: int,
) -> float:
    distance_score = max(0.0, 1.0 - (distance / max(cutoff, 1e-6)))
    margin_score = min(1.0, max(0.0, margin / max(cutoff, 1e-6)))
    sample_score = min(1.0, frames_used / max(MIN_TRACK_SAMPLES, 1))
    return float((0.55 * distance_score) + (0.25 * margin_score) + (0.20 * sample_score))


def _build_cluster_center_rows(
    fit_rows: pd.DataFrame,
    labels: np.ndarray,
    team_names: dict[int, str],
    distance_cutoffs: dict[int, float],
) -> pd.DataFrame:
    rows = []
    labeled = fit_rows.copy()
    labeled["cluster_id"] = labels
    for cluster_id, group in labeled.groupby("cluster_id"):
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "team": team_names[int(cluster_id)],
                "tracks_used": int(len(group)),
                "center_r": float(group["avg_r"].median()),
                "center_g": float(group["avg_g"].median()),
                "center_b": float(group["avg_b"].median()),
                "center_h": float(group["hsv_h"].median()),
                "center_s": float(group["hsv_s"].median()),
                "center_v": float(group["hsv_v"].median()),
                "center_lab_l": float(group["lab_l"].median()),
                "center_lab_a": float(group["lab_a"].median()),
                "center_lab_b": float(group["lab_b"].median()),
                "distance_cutoff": float(distance_cutoffs[int(cluster_id)]),
            }
        )
    return pd.DataFrame(rows).sort_values("team")


def _format_team_rows(track_colors: pd.DataFrame) -> pd.DataFrame:
    """Round and order team-classification output rows."""
    for column in [
        "avg_r",
        "avg_g",
        "avg_b",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "lab_l",
        "lab_a",
        "lab_b",
        "avg_confidence",
        "green_fraction",
        "valid_fraction",
        "valid_pixels",
        "crop_area",
        "box_width",
        "box_height",
        "cluster_distance",
        "cluster_margin",
        "assignment_confidence",
    ]:
        track_colors[column] = track_colors[column].round(3)
    return track_colors[
        [
            "track_id",
            "team",
            "avg_r",
            "avg_g",
            "avg_b",
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "lab_l",
            "lab_a",
            "lab_b",
            "frames_used",
            "avg_confidence",
            "green_fraction",
            "valid_fraction",
            "valid_pixels",
            "crop_area",
            "box_width",
            "box_height",
            "cluster_id",
            "cluster_distance",
            "cluster_margin",
            "assignment_confidence",
            "assignment_reason",
        ]
    ].sort_values(["team", "track_id"])


def _add_missing_unknown_tracks(team_rows: pd.DataFrame, track_ids) -> pd.DataFrame:
    known_track_ids = set(team_rows["track_id"].astype(int))
    missing_track_ids = sorted(int(track_id) for track_id in track_ids if int(track_id) not in known_track_ids)
    if not missing_track_ids:
        return team_rows

    unknown_rows = pd.DataFrame(
        [
            {
                "track_id": track_id,
                "team": "Unknown",
                "avg_r": np.nan,
                "avg_g": np.nan,
                "avg_b": np.nan,
                "hsv_h": np.nan,
                "hsv_s": np.nan,
                "hsv_v": np.nan,
                "lab_l": np.nan,
                "lab_a": np.nan,
                "lab_b": np.nan,
                "frames_used": 0,
                "avg_confidence": np.nan,
                "green_fraction": np.nan,
                "valid_fraction": 0,
                "valid_pixels": 0,
                "crop_area": 0,
                "box_width": 0,
                "box_height": 0,
                "cluster_id": -1,
                "cluster_distance": np.nan,
                "cluster_margin": np.nan,
                "assignment_confidence": 0,
                "assignment_reason": "no_valid_torso_samples",
            }
            for track_id in missing_track_ids
        ]
    )
    return pd.concat([team_rows, unknown_rows], ignore_index=True).sort_values(
        ["team", "track_id"]
    )


def _add_track_position_stats(team_rows: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    """Attach movement and position summaries used by role heuristics."""
    ordered_tracks = tracks.sort_values(["track_id", "frame"]).copy()
    ordered_tracks["delta_x"] = ordered_tracks.groupby("track_id")["center_x"].diff()
    ordered_tracks["delta_y"] = ordered_tracks.groupby("track_id")["center_y"].diff()
    ordered_tracks["step_distance"] = np.hypot(
        ordered_tracks["delta_x"].fillna(0),
        ordered_tracks["delta_y"].fillna(0),
    )
    ordered_tracks["box_width_raw"] = ordered_tracks["x2"] - ordered_tracks["x1"]
    ordered_tracks["box_height_raw"] = ordered_tracks["y2"] - ordered_tracks["y1"]

    position_stats = (
        ordered_tracks.groupby("track_id")
        .agg(
            track_frames=("frame", "count"),
            first_frame=("frame", "min"),
            last_frame=("frame", "max"),
            median_x=("center_x", "median"),
            median_y=("center_y", "median"),
            min_x=("center_x", "min"),
            max_x=("center_x", "max"),
            min_y=("center_y", "min"),
            max_y=("center_y", "max"),
            movement_distance=("step_distance", "sum"),
            median_box_width=("box_width_raw", "median"),
            median_box_height=("box_height_raw", "median"),
        )
        .reset_index()
    )
    position_stats["span_x"] = position_stats["max_x"] - position_stats["min_x"]
    position_stats["span_y"] = position_stats["max_y"] - position_stats["min_y"]

    attrs = dict(team_rows.attrs)
    result = team_rows.merge(position_stats, on="track_id", how="left")
    result.attrs.update(attrs)
    return result


def _assign_default_roles(team_rows: pd.DataFrame) -> pd.DataFrame:
    """Add role columns without running role-specific heuristics."""
    result = team_rows.copy()
    result["role"] = result["team"].map(
        {
            "Team A": "team_a_player",
            "Team B": "team_b_player",
        }
    ).fillna("unknown")
    result["color_cluster"] = result["cluster_id"] if "cluster_id" in result.columns else -1
    result["role_confidence"] = (
        result["assignment_confidence"] if "assignment_confidence" in result.columns else 0.0
    )
    result["role_reason"] = (
        result["assignment_reason"] if "assignment_reason" in result.columns else "legacy_team_label"
    )
    return result


def _assign_roles(
    team_rows: pd.DataFrame,
    frame_width: int,
    frame_height: int,
    role_clusters: int,
    min_goalkeeper_track_length: int,
    referee_distance_threshold: float,
) -> pd.DataFrame:
    """Separate outfield players, goalkeeper candidates, referees, and unknowns."""
    result = _assign_default_roles(team_rows)
    result["role_reason"] = result["role"].map(
        {
            "team_a_player": "outfield_color_cluster",
            "team_b_player": "outfield_color_cluster",
        }
    ).fillna(result["assignment_reason"])

    color_cluster_rows = _assign_role_color_clusters(result, role_clusters)
    result = color_cluster_rows["rows"]
    result.attrs["role_color_clusters"] = color_cluster_rows["summary"]

    outfield_clusters = _outfield_color_clusters(result)
    color_outlier = ~result["color_cluster"].isin(outfield_clusters)
    result["role_color_outlier"] = color_outlier

    goalkeeper_rows = _select_goalkeeper_candidates(
        result,
        frame_width=frame_width,
        min_goalkeeper_track_length=min_goalkeeper_track_length,
        color_outlier=color_outlier,
    )
    result = _apply_goalkeeper_roles(result, goalkeeper_rows, frame_width)

    referee_rows = _select_referee_candidates(
        result,
        frame_width=frame_width,
        color_outlier=color_outlier,
        referee_distance_threshold=referee_distance_threshold,
    )
    for row_index in referee_rows.index:
        if "goalkeeper" in str(result.at[row_index, "role"]):
            continue
        result.at[row_index, "team"] = "Unknown"
        result.at[row_index, "role"] = "referee"
        result.at[row_index, "role_confidence"] = max(
            float(result.at[row_index, "role_confidence"]),
            float(referee_rows.at[row_index, "role_score"]),
        )
        result.at[row_index, "role_reason"] = "color_outlier_central_official_candidate"

    result.loc[result["team"] == "Unknown", "role"] = result.loc[
        result["team"] == "Unknown", "role"
    ].where(result.loc[result["team"] == "Unknown", "role"].isin(["referee", "goalkeeper_left", "goalkeeper_right"]), "unknown")
    result.attrs.update(team_rows.attrs)
    return result


def _assign_role_color_clusters(team_rows: pd.DataFrame, role_clusters: int) -> dict:
    result = team_rows.copy()
    result["color_cluster"] = -1
    color_rows = result.dropna(subset=FEATURE_COLUMNS).copy()
    if color_rows.empty:
        return {"rows": result, "summary": pd.DataFrame()}

    cluster_count = min(max(2, int(role_clusters)), len(color_rows))
    features = color_rows[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    scaler = _fit_robust_scaler(features)
    transformed = _transform_features(features, scaler)
    labels = KMeans(n_clusters=cluster_count, random_state=42, n_init=30).fit_predict(transformed)
    result.loc[color_rows.index, "color_cluster"] = labels

    summary = _build_role_color_cluster_summary(result)
    return {"rows": result, "summary": summary}


def _build_role_color_cluster_summary(team_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    clustered = team_rows[team_rows["color_cluster"] >= 0]
    for color_cluster, group in clustered.groupby("color_cluster"):
        team_counts = group["team"].value_counts().to_dict()
        rows.append(
            {
                "color_cluster": int(color_cluster),
                "tracks": int(len(group)),
                "reliable_tracks": int((group["frames_used"] >= MIN_TRACK_SAMPLES).sum()),
                "team_a_tracks": int(team_counts.get("Team A", 0)),
                "team_b_tracks": int(team_counts.get("Team B", 0)),
                "unknown_tracks": int(team_counts.get("Unknown", 0)),
                "median_r": float(group["avg_r"].median()),
                "median_g": float(group["avg_g"].median()),
                "median_b": float(group["avg_b"].median()),
                "median_lab_l": float(group["lab_l"].median()),
                "median_lab_a": float(group["lab_a"].median()),
                "median_lab_b": float(group["lab_b"].median()),
                "median_x": float(group["median_x"].median()),
                "median_y": float(group["median_y"].median()),
                "median_movement": float(group["movement_distance"].median()),
            }
        )
    return pd.DataFrame(rows).sort_values("tracks", ascending=False)


def _outfield_color_clusters(team_rows: pd.DataFrame) -> set[int]:
    outfield_clusters = set()
    for team in ("Team A", "Team B"):
        team_rows_for_team = team_rows[
            (team_rows["team"] == team)
            & (team_rows["frames_used"] >= MIN_TRACK_SAMPLES)
            & (team_rows["color_cluster"] >= 0)
        ]
        if team_rows_for_team.empty:
            continue
        counts = team_rows_for_team["color_cluster"].value_counts()
        threshold = max(4, int(np.ceil(counts.max() * 0.40)))
        outfield_clusters.update(int(cluster) for cluster, count in counts.items() if count >= threshold)
    return outfield_clusters


def _select_goalkeeper_candidates(
    team_rows: pd.DataFrame,
    frame_width: int,
    min_goalkeeper_track_length: int,
    color_outlier: pd.Series,
) -> pd.DataFrame:
    candidates = team_rows[
        color_outlier
        & (team_rows["track_frames"] >= int(min_goalkeeper_track_length))
        & (team_rows["frames_used"] >= MIN_TRACK_SAMPLES)
        & (team_rows["median_x"].notna())
    ].copy()
    if candidates.empty:
        return candidates

    left_goal = candidates["median_x"] <= frame_width * GOAL_AREA_X_FRACTION
    right_goal = candidates["median_x"] >= frame_width * (1.0 - GOAL_AREA_X_FRACTION)
    candidates = candidates[left_goal | right_goal].copy()
    if candidates.empty:
        return candidates

    movement_reference = team_rows.loc[
        team_rows["team"].isin(["Team A", "Team B"]),
        "movement_distance",
    ].dropna()
    movement_cutoff = float(movement_reference.quantile(0.60)) if not movement_reference.empty else np.inf
    candidates["side"] = np.where(
        candidates["median_x"] <= frame_width * GOAL_AREA_X_FRACTION,
        "left",
        "right",
    )
    candidates["movement_score"] = 1.0 - (
        candidates["movement_distance"].fillna(movement_cutoff) / max(movement_cutoff, 1.0)
    ).clip(0, 1)
    candidates["length_score"] = (
        candidates["track_frames"] / max(int(min_goalkeeper_track_length), 1)
    ).clip(0, 1)
    candidates["role_score"] = (0.45 * candidates["length_score"]) + (0.35 * candidates["movement_score"]) + 0.20

    selected = []
    for _, group in candidates.groupby("side"):
        best = group.sort_values(["role_score", "track_frames"], ascending=False).head(1)
        if not best.empty and float(best.iloc[0]["role_score"]) >= 0.45:
            best.attrs = {}
            selected.append(best)
    return pd.concat(selected) if selected else candidates.iloc[0:0]


def _apply_goalkeeper_roles(
    team_rows: pd.DataFrame,
    goalkeeper_rows: pd.DataFrame,
    frame_width: int,
) -> pd.DataFrame:
    result = team_rows.copy()
    side_to_team = _infer_defensive_side_to_team(result, frame_width)
    for row_index, row in goalkeeper_rows.iterrows():
        side = row["side"]
        team = side_to_team.get(side)
        if team == "Team A":
            role = "team_a_goalkeeper"
        elif team == "Team B":
            role = "team_b_goalkeeper"
        else:
            role = f"goalkeeper_{side}"
            team = "Unknown"

        result.at[row_index, "team"] = team
        result.at[row_index, "role"] = role
        result.at[row_index, "role_confidence"] = max(
            float(result.at[row_index, "role_confidence"]),
            float(row["role_score"]),
        )
        result.at[row_index, "role_reason"] = f"color_outlier_near_{side}_goal"
    return result


def _infer_defensive_side_to_team(team_rows: pd.DataFrame, frame_width: int) -> dict[str, str]:
    team_a = team_rows[(team_rows["role"] == "team_a_player") & team_rows["median_x"].notna()]
    team_b = team_rows[(team_rows["role"] == "team_b_player") & team_rows["median_x"].notna()]
    if team_a.empty or team_b.empty:
        return {}

    team_a_median = float(team_a["median_x"].median())
    team_b_median = float(team_b["median_x"].median())
    if abs(team_a_median - team_b_median) < frame_width * 0.15:
        return {}
    if team_a_median < team_b_median:
        return {"left": "Team A", "right": "Team B"}
    return {"left": "Team B", "right": "Team A"}


def _select_referee_candidates(
    team_rows: pd.DataFrame,
    frame_width: int,
    color_outlier: pd.Series,
    referee_distance_threshold: float,
) -> pd.DataFrame:
    candidates = team_rows[
        color_outlier
        & (team_rows["color_cluster"] >= 0)
        & (team_rows["frames_used"] >= MIN_TRACK_SAMPLES)
        & (team_rows["track_frames"] >= max(20, MIN_TRACK_SAMPLES * 2))
        & (team_rows["median_x"] >= frame_width * CENTRAL_X_MIN)
        & (team_rows["median_x"] <= frame_width * CENTRAL_X_MAX)
        & (team_rows["cluster_distance"].fillna(np.inf) >= float(referee_distance_threshold))
    ].copy()
    if candidates.empty:
        return candidates

    movement_reference = team_rows.loc[
        team_rows["team"].isin(["Team A", "Team B"]),
        "movement_distance",
    ].dropna()
    movement_median = float(movement_reference.median()) if not movement_reference.empty else 1.0
    candidates["movement_score"] = (
        candidates["movement_distance"].fillna(0) / max(movement_median, 1.0)
    ).clip(0, 1)
    candidates["distance_score"] = (
        candidates["cluster_distance"].fillna(0) / max(float(referee_distance_threshold), 1.0)
    ).clip(0, 1)
    candidates["role_score"] = (0.55 * candidates["distance_score"]) + (0.45 * candidates["movement_score"])
    return candidates.sort_values(["role_score", "track_frames"], ascending=False).head(4)


def _prepare_debug_dir(debug_dir: Path) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = debug_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    for path in crops_dir.glob("track_*.jpg"):
        path.unlink()
    role_crops_dir = debug_dir / "role_crops"
    if role_crops_dir.exists():
        for role_dir in role_crops_dir.iterdir():
            if role_dir.is_dir():
                for path in role_dir.glob("track_*.jpg"):
                    path.unlink()
    role_crops_dir.mkdir(parents=True, exist_ok=True)


def _write_representative_crops(representative_crops: dict, crops_dir: Path) -> None:
    crops_dir.mkdir(parents=True, exist_ok=True)
    for track_id, sample in sorted(representative_crops.items()):
        output = crops_dir / f"track_{track_id:04d}.jpg"
        cv2.imwrite(str(output), sample["crop"])


def _write_debug_outputs(team_rows: pd.DataFrame, debug_dir: Path) -> None:
    assignments_path = debug_dir / "team_assignments.csv"
    team_rows.to_csv(assignments_path, index=False)
    _write_role_assignments(team_rows, debug_dir / "role_assignments.csv")
    _write_movement_summary(team_rows, debug_dir / "movement_summary.csv")
    cluster_centers = team_rows.attrs.get("cluster_centers")
    if cluster_centers is not None:
        cluster_centers.to_csv(debug_dir / "cluster_centers.csv", index=False)
    role_color_clusters = team_rows.attrs.get("role_color_clusters")
    if role_color_clusters is not None and not role_color_clusters.empty:
        role_color_clusters.to_csv(debug_dir / "color_clusters.csv", index=False)
        _write_color_clusters_palette(role_color_clusters, debug_dir / "color_clusters_palette.png")
    _write_role_crops(team_rows, debug_dir / "crops", debug_dir / "role_crops")
    _write_palette(team_rows, debug_dir / "team_palette.png")


def _write_role_assignments(team_rows: pd.DataFrame, output_path: Path) -> None:
    columns = [
        "track_id",
        "team",
        "role",
        "color_cluster",
        "samples",
        "frames_used",
        "track_frames",
        "median_x",
        "median_y",
        "movement_distance",
        "role_confidence",
        "role_reason",
    ]
    available_columns = [column for column in columns if column in team_rows.columns]
    team_rows[available_columns].to_csv(output_path, index=False)


def _write_movement_summary(team_rows: pd.DataFrame, output_path: Path) -> None:
    columns = [
        "track_id",
        "role",
        "track_frames",
        "first_frame",
        "last_frame",
        "median_x",
        "median_y",
        "min_x",
        "max_x",
        "min_y",
        "max_y",
        "span_x",
        "span_y",
        "movement_distance",
        "median_box_width",
        "median_box_height",
    ]
    available_columns = [column for column in columns if column in team_rows.columns]
    team_rows[available_columns].to_csv(output_path, index=False)


def _write_role_crops(team_rows: pd.DataFrame, crops_dir: Path, role_crops_dir: Path) -> None:
    role_crops_dir.mkdir(parents=True, exist_ok=True)
    for _, row in team_rows.iterrows():
        role = str(row.get("role", "unknown"))
        role_dir = role_crops_dir / role
        role_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crops_dir / f"track_{int(row['track_id']):04d}.jpg"
        if crop_path.exists():
            shutil.copy2(crop_path, role_dir / crop_path.name)


def _write_palette(team_rows: pd.DataFrame, output_path: Path) -> None:
    rows = team_rows.sort_values(["role", "track_id"]).reset_index(drop=True)
    swatch_width = 64
    swatch_height = 42
    label_width = 130
    height = max(swatch_height, len(rows) * swatch_height)
    width = swatch_width + label_width
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for index, row in rows.iterrows():
        y1 = index * swatch_height
        y2 = y1 + swatch_height
        if pd.isna(row["avg_r"]) or pd.isna(row["avg_g"]) or pd.isna(row["avg_b"]):
            rgb = (180, 180, 180)
        else:
            rgb = (int(row["avg_r"]), int(row["avg_g"]), int(row["avg_b"]))
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(canvas, (0, y1), (swatch_width, y2), bgr, -1)
        cv2.rectangle(canvas, (0, y1), (swatch_width, y2), (80, 80, 80), 1)
        label = f"ID {int(row['track_id'])} {row.get('role', row['team'])}"
        cv2.putText(
            canvas,
            label,
            (swatch_width + 6, y1 + 25),
            font,
            0.42,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


def _write_color_clusters_palette(clusters: pd.DataFrame, output_path: Path) -> None:
    rows = clusters.sort_values("color_cluster").reset_index(drop=True)
    swatch_width = 72
    row_height = 46
    label_width = 260
    height = max(row_height, len(rows) * row_height)
    width = swatch_width + label_width
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for index, row in rows.iterrows():
        y1 = index * row_height
        y2 = y1 + row_height
        rgb = (
            int(row["median_r"]),
            int(row["median_g"]),
            int(row["median_b"]),
        )
        bgr = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(canvas, (0, y1), (swatch_width, y2), bgr, -1)
        cv2.rectangle(canvas, (0, y1), (swatch_width, y2), (80, 80, 80), 1)
        label = (
            f"C{int(row['color_cluster'])} n={int(row['tracks'])} "
            f"A={int(row['team_a_tracks'])} B={int(row['team_b_tracks'])} "
            f"U={int(row['unknown_tracks'])}"
        )
        cv2.putText(
            canvas,
            label,
            (swatch_width + 6, y1 + 28),
            font,
            0.42,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


def _print_team_summary(team_rows: pd.DataFrame) -> None:
    """Print assignment counts for the supported labels."""
    counts = team_rows["team"].value_counts().to_dict()
    print("Team assignment counts:")
    for team in TEAM_LABELS:
        print(f"- {team}: {counts.get(team, 0)}")
    if "role" in team_rows.columns:
        role_counts = team_rows["role"].value_counts().to_dict()
        print("Role assignment counts:")
        for role in ROLE_LABELS:
            if role_counts.get(role, 0):
                print(f"- {role}: {role_counts.get(role, 0)}")


def _draw_team_tracks(frame, tracks: pd.DataFrame, role_by_track_id: dict) -> None:
    frame_height, frame_width = frame.shape[:2]
    scale = frame_height / 720
    font_scale = max(0.25, 0.45 * scale)
    thickness = max(1, int(2 * scale))
    padding = max(2, int(4 * scale))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for _, row in tracks.iterrows():
        track_id = int(row["track_id"])
        role = role_by_track_id.get(track_id, "unknown")

        x1 = max(0, min(int(round(row["x1"])), frame_width - 1))
        y1 = max(0, min(int(round(row["y1"])), frame_height - 1))
        x2 = max(0, min(int(round(row["x2"])), frame_width - 1))
        y2 = max(0, min(int(round(row["y2"])), frame_height - 1))
        color = ROLE_COLORS.get(role, ROLE_COLORS["unknown"])
        label = f"ID {track_id} {_short_role_label(role)}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        _draw_label(frame, label, x1, y1, color, font, font_scale, thickness, padding)


def _short_role_label(role: str) -> str:
    labels = {
        "team_a_player": "A",
        "team_b_player": "B",
        "team_a_goalkeeper": "A GK",
        "team_b_goalkeeper": "B GK",
        "goalkeeper_left": "GK L",
        "goalkeeper_right": "GK R",
        "referee": "REF",
        "unknown": "UNK",
    }
    return labels.get(role, role)


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
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=2,
        help="Number of color clusters for team classification. Defaults to 2.",
    )
    parser.add_argument(
        "--min-track-samples",
        type=int,
        default=MIN_TRACK_SAMPLES,
        help="Minimum valid torso samples before assigning a team.",
    )
    parser.add_argument(
        "--debug-dir",
        default="outputs/team_debug",
        help="Directory for team-classification debug crops and CSVs.",
    )
    parser.add_argument(
        "--detect-roles",
        action="store_true",
        help="Separate outfield players, goalkeeper candidates, referees, and unknowns.",
    )
    parser.add_argument(
        "--role-clusters",
        type=int,
        default=DEFAULT_ROLE_CLUSTERS,
        help="Number of color clusters used for role outlier detection.",
    )
    parser.add_argument(
        "--min-goalkeeper-track-length",
        type=int,
        default=MIN_GOALKEEPER_TRACK_LENGTH,
        help="Minimum tracked frames before a color outlier can be a goalkeeper.",
    )
    parser.add_argument(
        "--referee-distance-threshold",
        type=float,
        default=REFEREE_DISTANCE_THRESHOLD,
        help="Minimum color distance from outfield centroids before referee consideration.",
    )
    parser.add_argument(
        "--include-goalkeepers-in-heatmaps",
        action="store_true",
        help="Include team goalkeeper roles in team heatmaps. Default is outfield only.",
    )
    args = parser.parse_args()

    try:
        team_rows = classify_teams(
            video_path=args.video,
            tracks_csv_path=args.tracks_csv,
            output_csv_path=args.output,
            num_clusters=args.num_clusters,
            debug_dir=args.debug_dir,
            min_track_samples=args.min_track_samples,
            detect_roles=args.detect_roles,
            role_clusters=args.role_clusters,
            min_goalkeeper_track_length=args.min_goalkeeper_track_length,
            referee_distance_threshold=args.referee_distance_threshold,
        )
        print(f"Player teams saved to: {args.output}")
        _print_team_summary(team_rows)
        print(f"Team debug outputs saved to: {args.debug_dir}")

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
                include_goalkeepers=args.include_goalkeepers_in_heatmaps,
            )
            print(f"Team A heatmap saved to: {args.team_a_heatmap}")
            print(f"Team B heatmap saved to: {args.team_b_heatmap}")
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return


if __name__ == "__main__":
    main()
