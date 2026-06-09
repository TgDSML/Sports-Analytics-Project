"""Baseline ball tracking from filtered ball detections."""

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, field
from math import dist
from pathlib import Path

import cv2
import numpy as np


BALL_TRACK_COLUMNS = [
    "track_id",
    "frame",
    "timestamp",
    "center_x",
    "center_y",
    "confidence",
    "is_interpolated",
]


@dataclass
class BallPoint:
    """One ball track point."""

    track_id: int
    frame: int
    timestamp: float
    center_x: float
    center_y: float
    confidence: float
    is_interpolated: bool = False


@dataclass
class ActiveBallTrack:
    """Mutable state for an active ball track."""

    track_id: int
    points: list[BallPoint] = field(default_factory=list)
    missing: int = 0

    @property
    def last_point(self) -> BallPoint:
        return self.points[-1]


@dataclass
class BallTrackingSummary:
    """Summary diagnostics for a ball tracking run."""

    total_tracks: int
    longest_track: int
    average_track_length: float
    interpolated_points: int
    gaps_filled: int


class BallTracker:
    """Link ball detections into short explainable tracks."""

    def __init__(
        self,
        max_distance: float = 90.0,
        max_gap: int = 8,
        confidence_weight: float = 25.0,
        min_track_length: int = 2,
    ) -> None:
        self.max_distance = float(max_distance)
        self.max_gap = int(max_gap)
        self.confidence_weight = float(confidence_weight)
        self.min_track_length = int(min_track_length)
        self.next_track_id = 1
        self.active_tracks: dict[int, ActiveBallTrack] = {}
        self.finished_tracks: list[ActiveBallTrack] = []

    def update(self, frame: int, timestamp: float, detections: list[dict]) -> None:
        """Update active tracks from detections for one frame."""
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        candidates = []

        for track_id, track in self.active_tracks.items():
            frame_gap = max(1, frame - track.last_point.frame)
            allowed_distance = self.max_distance * frame_gap
            for detection_index, detection in enumerate(detections):
                distance = dist(
                    (track.last_point.center_x, track.last_point.center_y),
                    (float(detection["center_x"]), float(detection["center_y"])),
                )
                if distance > allowed_distance:
                    continue
                confidence = float(detection.get("confidence", 0.0))
                score = distance - (confidence * self.confidence_weight)
                candidates.append((score, distance, track_id, detection_index))

        for _, _, track_id, detection_index in sorted(candidates):
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            track = self.active_tracks[track_id]
            detection = detections[detection_index]
            track.points.append(
                BallPoint(
                    track_id=track_id,
                    frame=frame,
                    timestamp=timestamp,
                    center_x=float(detection["center_x"]),
                    center_y=float(detection["center_y"]),
                    confidence=float(detection.get("confidence", 0.0)),
                )
            )
            track.missing = 0
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        for track_id in list(self.active_tracks):
            if track_id in matched_tracks:
                continue
            self.active_tracks[track_id].missing += 1
            if self.active_tracks[track_id].missing > self.max_gap:
                self._finish_track(track_id)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            self._start_track(frame, timestamp, detection)

    def finish(self) -> list[ActiveBallTrack]:
        """Finish all active tracks and return retained tracks."""
        for track_id in list(self.active_tracks):
            self._finish_track(track_id)
        return [
            track
            for track in self.finished_tracks
            if len([point for point in track.points if not point.is_interpolated]) >= self.min_track_length
        ]

    def _start_track(self, frame: int, timestamp: float, detection: dict) -> None:
        track = ActiveBallTrack(track_id=self.next_track_id)
        track.points.append(
            BallPoint(
                track_id=track.track_id,
                frame=frame,
                timestamp=timestamp,
                center_x=float(detection["center_x"]),
                center_y=float(detection["center_y"]),
                confidence=float(detection.get("confidence", 0.0)),
            )
        )
        self.active_tracks[track.track_id] = track
        self.next_track_id += 1

    def _finish_track(self, track_id: int) -> None:
        track = self.active_tracks.pop(track_id)
        self.finished_tracks.append(track)


def track_ball_detections(
    detections_csv_path: Path,
    output_csv_path: Path,
    summary_csv_path: Path,
    summary_md_path: Path,
    max_distance: float = 90.0,
    max_gap: int = 8,
    confidence_weight: float = 25.0,
    min_track_length: int = 2,
) -> BallTrackingSummary:
    """Build ball tracks from filtered detections and write outputs."""
    detections = _read_ball_detections(detections_csv_path)
    detections_by_frame = {}
    timestamp_by_frame = {}
    for row in detections:
        frame = int(row["frame"])
        detections_by_frame.setdefault(frame, []).append(row)
        timestamp_by_frame[frame] = float(row["timestamp"])

    tracker = BallTracker(
        max_distance=max_distance,
        max_gap=max_gap,
        confidence_weight=confidence_weight,
        min_track_length=min_track_length,
    )
    if detections_by_frame:
        first_frame = min(detections_by_frame)
        last_frame = max(detections_by_frame)
        for frame in range(first_frame, last_frame + 1):
            timestamp = timestamp_by_frame.get(frame)
            if timestamp is None:
                timestamp = frame * _estimate_frame_duration(timestamp_by_frame)
            tracker.update(frame, timestamp, detections_by_frame.get(frame, []))

    tracks = tracker.finish()
    gaps_filled = _interpolate_short_gaps(tracks, max_gap=max_gap)
    points = [point for track in tracks for point in track.points]
    points.sort(key=lambda point: (point.frame, point.track_id, point.is_interpolated))
    _write_ball_tracks(points, output_csv_path)

    summary = _build_summary(tracks, gaps_filled)
    _write_summary(summary, summary_csv_path, summary_md_path)
    print(f"Ball tracks saved to: {output_csv_path}")
    print(f"Ball tracking diagnostics saved to: {summary_csv_path}")
    print(f"Ball tracking report saved to: {summary_md_path}")
    return summary


def create_ball_tracking_video(
    video_path: Path,
    tracks_csv_path: Path,
    output_video_path: Path,
    tail_length: int = 18,
) -> None:
    """Draw ball track IDs and recent trajectory tails on a video."""
    tracks = _read_ball_tracks(tracks_csv_path)
    tracks_by_frame = {}
    for row in tracks:
        tracks_by_frame.setdefault(int(row["frame"]), []).append(row)

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

    history: dict[int, list[tuple[int, int]]] = {}
    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_points = tracks_by_frame.get(frame_index, [])
        for point in frame_points:
            track_id = int(point["track_id"])
            center = (int(round(float(point["center_x"]))), int(round(float(point["center_y"]))))
            history.setdefault(track_id, []).append(center)
            history[track_id] = history[track_id][-tail_length:]
            _draw_ball_track(frame, track_id, center, history[track_id], point["is_interpolated"])

        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()
    print(f"Ball tracking video saved to: {output_video_path}")


def _read_ball_detections(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Ball detections CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"frame", "timestamp", "center_x", "center_y", "confidence"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Ball detections CSV missing column(s): {', '.join(sorted(missing))}")
        return [row for row in reader if row.get("frame")]


def _read_ball_tracks(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


def _estimate_frame_duration(timestamp_by_frame: dict[int, float]) -> float:
    if len(timestamp_by_frame) < 2:
        return 1 / 30
    frames = sorted(timestamp_by_frame)
    durations = []
    for previous, current in zip(frames, frames[1:]):
        frame_gap = current - previous
        if frame_gap > 0:
            durations.append((timestamp_by_frame[current] - timestamp_by_frame[previous]) / frame_gap)
    durations = [duration for duration in durations if duration > 0]
    return float(np.median(durations)) if durations else 1 / 30


def _interpolate_short_gaps(tracks: list[ActiveBallTrack], max_gap: int) -> int:
    gaps_filled = 0
    for track in tracks:
        ordered = sorted(track.points, key=lambda point: point.frame)
        filled = []
        for previous, current in zip(ordered, ordered[1:]):
            filled.append(previous)
            gap = current.frame - previous.frame - 1
            if 0 < gap <= max_gap:
                gaps_filled += 1
                for step in range(1, gap + 1):
                    ratio = step / (gap + 1)
                    filled.append(
                        BallPoint(
                            track_id=track.track_id,
                            frame=previous.frame + step,
                            timestamp=previous.timestamp
                            + ratio * (current.timestamp - previous.timestamp),
                            center_x=previous.center_x + ratio * (current.center_x - previous.center_x),
                            center_y=previous.center_y + ratio * (current.center_y - previous.center_y),
                            confidence=min(previous.confidence, current.confidence),
                            is_interpolated=True,
                        )
                    )
        if ordered:
            filled.append(ordered[-1])
        track.points = sorted(filled, key=lambda point: point.frame)
    return gaps_filled


def _write_ball_tracks(points: list[BallPoint], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=BALL_TRACK_COLUMNS)
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "track_id": point.track_id,
                    "frame": point.frame,
                    "timestamp": f"{point.timestamp:.6f}",
                    "center_x": f"{point.center_x:.2f}",
                    "center_y": f"{point.center_y:.2f}",
                    "confidence": f"{point.confidence:.6f}",
                    "is_interpolated": int(point.is_interpolated),
                }
            )


def _build_summary(tracks: list[ActiveBallTrack], gaps_filled: int) -> BallTrackingSummary:
    lengths = [len(track.points) for track in tracks]
    interpolated_points = sum(point.is_interpolated for track in tracks for point in track.points)
    return BallTrackingSummary(
        total_tracks=len(tracks),
        longest_track=max(lengths) if lengths else 0,
        average_track_length=float(np.mean(lengths)) if lengths else 0.0,
        interpolated_points=int(interpolated_points),
        gaps_filled=gaps_filled,
    )


def _write_summary(summary: BallTrackingSummary, csv_path: Path, md_path: Path) -> None:
    rows = [
        ("counts", "total_ball_tracks", summary.total_tracks),
        ("counts", "longest_track", summary.longest_track),
        ("counts", "average_track_length", f"{summary.average_track_length:.2f}"),
        ("counts", "interpolated_points", summary.interpolated_points),
        ("counts", "gaps_filled", summary.gaps_filled),
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows({"section": section, "metric": metric, "value": value} for section, metric, value in rows)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(
        "\n".join(
            [
                "# Ball Tracking Diagnostics",
                "",
                f"- Total ball tracks: {summary.total_tracks}",
                f"- Longest track: {summary.longest_track}",
                f"- Average track length: {summary.average_track_length:.2f}",
                f"- Interpolated points: {summary.interpolated_points}",
                f"- Gaps filled: {summary.gaps_filled}",
            ]
        ),
        encoding="utf-8",
    )


def _draw_ball_track(
    frame,
    track_id: int,
    center: tuple[int, int],
    tail: list[tuple[int, int]],
    is_interpolated,
) -> None:
    frame_height = frame.shape[0]
    scale = frame_height / 720
    radius = max(5, int(7 * scale))
    thickness = max(2, int(2 * scale))
    color = (0, 255, 255) if str(is_interpolated) not in {"1", "True", "true"} else (0, 165, 255)

    for start, end in zip(tail, tail[1:]):
        cv2.line(frame, start, end, color, thickness)
    cv2.circle(frame, center, radius, color, thickness)
    cv2.putText(
        frame,
        f"Ball ID {track_id}",
        (min(center[0] + 10, frame.shape[1] - 110), max(center[1] - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.35, 0.48 * scale),
        color,
        thickness,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track filtered ball detections")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detections-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/ball_tracks_30s_720p.csv"))
    parser.add_argument("--output-video", type=Path, default=Path("outputs/ball_tracked_30s_720p.mp4"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/ball_debug/ball_tracking_summary.csv"))
    parser.add_argument("--summary-md", type=Path, default=Path("outputs/ball_debug/ball_tracking_summary.md"))
    parser.add_argument("--max-distance", type=float, default=90.0)
    parser.add_argument("--max-gap", type=int, default=8)
    parser.add_argument("--confidence-weight", type=float, default=25.0)
    parser.add_argument("--min-track-length", type=int, default=2)
    parser.add_argument("--tail-length", type=int, default=18)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    track_ball_detections(
        detections_csv_path=args.detections_csv,
        output_csv_path=args.output_csv,
        summary_csv_path=args.summary_csv,
        summary_md_path=args.summary_md,
        max_distance=args.max_distance,
        max_gap=args.max_gap,
        confidence_weight=args.confidence_weight,
        min_track_length=args.min_track_length,
    )
    create_ball_tracking_video(
        video_path=args.video,
        tracks_csv_path=args.output_csv,
        output_video_path=args.output_video,
        tail_length=args.tail_length,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
