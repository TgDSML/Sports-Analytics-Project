import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.analytics.heatmap import generate_heatmap
from src.analytics.player_stats import generate_player_stats
from src.analytics.trajectories import generate_trajectory_plot
from src.detection.detector import YOLODetector
from src.tracking.tracker import CentroidTracker
from src.utils.io import write_detections_csv, write_tracks_csv
from src.utils.video import find_random_video


def build_detection_row(detection, frame_index: int, fps: float) -> dict:
    """Build one CSV-ready row from a frame detection."""
    x1, y1, x2, y2 = detection.bbox
    width = x2 - x1
    height = y2 - y1

    return {
        "frame": frame_index,
        "timestamp": frame_index / fps,
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": detection.confidence,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "center_x": (x1 + x2) / 2,
        "center_y": (y1 + y2) / 2,
        "width": width,
        "height": height,
    }


def build_track_row(track, frame_index: int, fps: float) -> dict:
    """Build one CSV-ready row from an active track."""
    x1, y1, x2, y2 = track.bbox
    center_x, center_y = track.centroid

    return {
        "frame": frame_index,
        "timestamp": frame_index / fps,
        "track_id": track.track_id,
        "class_name": track.class_name,
        "confidence": track.confidence,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "center_x": center_x,
        "center_y": center_y,
    }


def draw_tracks(frame, tracks):
    """Draw tracked boxes and IDs on a copy of the frame."""
    annotated_frame = frame.copy()
    frame_height, frame_width = frame.shape[:2]
    scale = frame_height / 720
    font_scale = max(0.25, 0.45 * scale)
    thickness = max(1, int(2 * scale))
    padding = max(2, int(4 * scale))
    color = (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        label = f"ID {track.track_id}"

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

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
            annotated_frame,
            (label_x, background_top),
            (background_right, background_bottom),
            color,
            -1,
        )
        cv2.putText(
            annotated_frame,
            label,
            (label_x + padding, label_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return annotated_frame


def draw_track_box(frame, track_id: int, bbox: tuple[int, int, int, int], color) -> None:
    """Draw one tracked box and compact ID label in place."""
    frame_height, frame_width = frame.shape[:2]
    scale = frame_height / 720
    font_scale = max(0.25, 0.45 * scale)
    thickness = max(1, int(2 * scale))
    padding = max(2, int(4 * scale))
    font = cv2.FONT_HERSHEY_SIMPLEX
    x1, y1, x2, y2 = bbox
    label = f"ID {track_id}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

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


def process_video(
    video_path: Path,
    output_path: Path,
    model_path: str,
    conf: float,
    imgsz: int,
    csv_output_path: Path | None = None,
    enable_tracking: bool = False,
    tracks_csv_path: Path | None = None,
    tracker_type: str = "bytetrack",
    max_distance: float = 120,
    max_missing: int = 30,
    smoothing: float = 0.7,
    min_box_area: int = 100,
    show: bool = False,
) -> tuple[int, int]:
    """Run YOLO on a video and write annotated video plus optional CSV."""
    if enable_tracking and tracker_type == "bytetrack":
        return process_video_bytetrack(
            video_path=video_path,
            output_path=output_path,
            model_path=model_path,
            conf=conf,
            imgsz=imgsz,
            csv_output_path=csv_output_path,
            tracks_csv_path=tracks_csv_path,
            show=show,
        )

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    print(f"Processing video: {video_path}")
    print(f"Writing annotated output: {output_path}")

    detector = YOLODetector(model_path=model_path, conf=conf, imgsz=imgsz)
    tracker = (
        CentroidTracker(
            max_distance=max_distance,
            max_missing=max_missing,
            smoothing=smoothing,
            min_box_area=min_box_area,
        )
        if enable_tracking
        else None
    )
    detection_rows = []
    track_rows = []
    total_frames = 0
    total_detections_before_filtering = 0
    total_detections_after_filtering = 0
    total_tracks_per_frame = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_index = total_frames
        detections, raw_detection_count = detector.detect(frame)
        total_frames += 1
        total_detections_before_filtering += raw_detection_count
        total_detections_after_filtering += len(detections)

        if csv_output_path is not None:
            detection_rows.extend(
                build_detection_row(detection, frame_index, fps)
                for detection in detections
            )

        if tracker is not None:
            tracks = tracker.update(detections)
            total_tracks_per_frame += len(tracks)
            if tracks_csv_path is not None:
                track_rows.extend(
                    build_track_row(track, frame_index, fps) for track in tracks
                )
            annotated_frame = draw_tracks(frame, tracks)
        else:
            annotated_frame = detector.draw_detections(frame, detections)

        writer.write(annotated_frame)

        if show:
            cv2.imshow("YOLO Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    writer.release()

    if show:
        cv2.destroyAllWindows()

    average_detections = (
        total_detections_after_filtering / total_frames if total_frames else 0
    )
    print(f"Total frames processed: {total_frames}")
    print(f"Total detections before filtering: {total_detections_before_filtering}")
    print(f"Total detections after filtering: {total_detections_after_filtering}")
    print(f"Average detections per frame: {average_detections:.2f}")

    if tracker is not None:
        average_tracks = total_tracks_per_frame / total_frames if total_frames else 0
        print(f"Total unique track IDs created: {tracker.total_tracks_created}")
        print(f"Active tracks at end: {len(tracker.tracks)}")
        print(f"Average tracks per frame: {average_tracks:.2f}")

    if csv_output_path is not None:
        write_detections_csv(detection_rows, csv_output_path)
        print(f"Detection CSV saved to: {csv_output_path}")

    if tracker is not None and tracks_csv_path is not None:
        write_tracks_csv(track_rows, tracks_csv_path)
        print(f"Tracks CSV saved to: {tracks_csv_path}")

    return width, height


def process_video_bytetrack(
    video_path: Path,
    output_path: Path,
    model_path: str,
    conf: float,
    imgsz: int,
    csv_output_path: Path | None = None,
    tracks_csv_path: Path | None = None,
    show: bool = False,
) -> tuple[int, int]:
    """Run YOLO with ByteTrack and write annotated video plus tracks CSV."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0:
        fps = 30

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    print(f"Processing video with ByteTrack: {video_path}")
    print(f"Writing annotated output: {output_path}")

    model = YOLO(model_path)
    detection_rows = []
    track_rows = []
    total_frames = 0
    total_tracks_per_frame = 0
    unique_track_ids = set()
    track_id_map = {}
    next_track_id = 1

    results = model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        persist=True,
        conf=conf,
        imgsz=imgsz,
        classes=[0],
        stream=True,
        verbose=False,
    )

    for frame_index, result in enumerate(results):
        annotated_frame = result.orig_img.copy()
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id != 0:
                    continue

                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                width_box = x2 - x1
                height_box = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                if csv_output_path is not None:
                    detection_rows.append(
                        {
                            "frame": frame_index,
                            "timestamp": frame_index / fps,
                            "class_id": class_id,
                            "class_name": "person",
                            "confidence": confidence,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "center_x": center_x,
                            "center_y": center_y,
                            "width": width_box,
                            "height": height_box,
                        }
                    )

                if box.id is None:
                    continue

                raw_track_id = int(box.id[0])
                if raw_track_id not in track_id_map:
                    track_id_map[raw_track_id] = next_track_id
                    next_track_id += 1
                track_id = track_id_map[raw_track_id]
                unique_track_ids.add(track_id)
                draw_track_box(
                    annotated_frame,
                    track_id,
                    (x1, y1, x2, y2),
                    (0, 255, 255),
                )
                track_rows.append(
                    {
                        "frame": frame_index,
                        "timestamp": frame_index / fps,
                        "track_id": track_id,
                        "class_name": "person",
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "center_x": center_x,
                        "center_y": center_y,
                    }
                )

        writer.write(annotated_frame)

        frame_track_count = 0
        if boxes is not None and boxes.id is not None:
            frame_track_count = len(boxes.id)
        total_tracks_per_frame += frame_track_count
        total_frames += 1

        if show:
            cv2.imshow("YOLO ByteTrack", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    writer.release()
    if show:
        cv2.destroyAllWindows()

    average_tracks = total_tracks_per_frame / total_frames if total_frames else 0
    print(f"Total frames processed: {total_frames}")
    print(f"Total tracked rows: {len(track_rows)}")
    print(f"Total unique track IDs: {len(unique_track_ids)}")
    print(f"Average tracks per frame: {average_tracks:.2f}")

    if csv_output_path is not None:
        write_detections_csv(detection_rows, csv_output_path)
        print(f"Detection CSV saved to: {csv_output_path}")

    if tracks_csv_path is not None:
        write_tracks_csv(track_rows, tracks_csv_path)
        print(f"Tracks CSV saved to: {tracks_csv_path}")

    return width, height


def main():
    """Parse CLI arguments and run the video detection baseline."""
    parser = argparse.ArgumentParser(description="Run YOLO detection on a sports video")
    parser.add_argument(
        "--video",
        default="data/sample.mp4",
        help="Path to the input video file",
    )
    parser.add_argument(
        "--output",
        default="outputs/yolov8_baseline.mp4",
        help="Path to save the annotated output video",
    )
    parser.add_argument(
        "--random-soccernet",
        action="store_true",
        help="Select a random video from the local SoccerNet dataset",
    )
    parser.add_argument(
        "--soccernet-dir",
        default="data/SoccerNet",
        help="Path to the local SoccerNet dataset directory",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model path or model name",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="YOLO confidence threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display annotated frames while processing",
    )
    parser.add_argument(
        "--csv-output",
        help="Optional path to save detections as CSV",
    )
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        help="Enable tracking for person detections",
    )
    parser.add_argument(
        "--tracker-type",
        choices=["centroid", "bytetrack"],
        default="bytetrack",
        help="Tracking backend to use when tracking is enabled",
    )
    parser.add_argument(
        "--tracks-csv",
        default="outputs/tracks_30s.csv",
        help="Path to save tracks CSV when tracking is enabled",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=120,
        help="Maximum centroid distance for matching tracks",
    )
    parser.add_argument(
        "--max-missing",
        type=int,
        default=30,
        help="Maximum missing frames before a track is removed",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.7,
        help="Bounding box smoothing factor from 0.0 to 1.0",
    )
    parser.add_argument(
        "--min-box-area",
        type=int,
        default=100,
        help="Minimum person box area to keep for tracking",
    )
    parser.add_argument(
        "--generate-heatmap",
        action="store_true",
        help="Generate a player movement heatmap from the tracks CSV",
    )
    parser.add_argument(
        "--heatmap-output",
        default="outputs/heatmap_30s.png",
        help="Path to save the generated heatmap image",
    )
    parser.add_argument(
        "--heatmap-track-id",
        type=int,
        help="Optional track ID for a single-player heatmap",
    )
    parser.add_argument(
        "--generate-trajectories",
        action="store_true",
        help="Generate player trajectory plots from the tracks CSV",
    )
    parser.add_argument(
        "--trajectory-output",
        default="outputs/trajectories_all.png",
        help="Path to save the generated trajectory image",
    )
    parser.add_argument(
        "--trajectory-track-id",
        type=int,
        help="Optional track ID for a single-player trajectory plot",
    )
    parser.add_argument(
        "--generate-player-stats",
        action="store_true",
        help="Generate player movement statistics from the tracks CSV",
    )
    parser.add_argument(
        "--player-stats-output",
        default="outputs/player_stats_30s.csv",
        help="Path to save player movement statistics CSV",
    )
    args = parser.parse_args()

    try:
        analytics_requested = (
            args.generate_heatmap
            or args.generate_trajectories
            or args.generate_player_stats
        )
        if analytics_requested and not args.enable_tracking:
            raise RuntimeError(
                "--generate-heatmap, --generate-trajectories, and "
                "--generate-player-stats require --enable-tracking"
            )

        video_path = Path(args.video)
        if args.random_soccernet:
            video_path = find_random_video(args.soccernet_dir)
            print(f"Selected SoccerNet video: {video_path}")

        frame_width, frame_height = process_video(
            video_path=video_path,
            output_path=Path(args.output),
            model_path=args.model,
            conf=args.conf,
            imgsz=args.imgsz,
            csv_output_path=Path(args.csv_output) if args.csv_output else None,
            enable_tracking=args.enable_tracking,
            tracks_csv_path=Path(args.tracks_csv) if args.enable_tracking else None,
            tracker_type=args.tracker_type,
            max_distance=args.max_distance,
            max_missing=args.max_missing,
            smoothing=args.smoothing,
            min_box_area=args.min_box_area,
            show=args.show,
        )

        if args.generate_heatmap:
            generate_heatmap(
                tracks_csv_path=Path(args.tracks_csv),
                output_path=Path(args.heatmap_output),
                frame_width=frame_width,
                frame_height=frame_height,
                track_id=args.heatmap_track_id,
            )
            print(f"Heatmap saved to: {args.heatmap_output}")

        if args.generate_trajectories:
            generate_trajectory_plot(
                tracks_csv_path=Path(args.tracks_csv),
                output_path=Path(args.trajectory_output),
                frame_width=frame_width,
                frame_height=frame_height,
                track_id=args.trajectory_track_id,
            )
            print(f"Trajectory plot saved to: {args.trajectory_output}")

        if args.generate_player_stats:
            generate_player_stats(
                tracks_csv_path=Path(args.tracks_csv),
                output_csv_path=Path(args.player_stats_output),
            )
            print(f"Player stats saved to: {args.player_stats_output}")
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return

    print(f"Annotated video saved to: {args.output}")


if __name__ == "__main__":
    main()
