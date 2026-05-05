import argparse
from pathlib import Path

import cv2

from src.detection.detector import YOLODetector
from src.utils.video import find_random_video


def process_video(
    video_path: Path,
    output_path: Path,
    model_path: str,
    conf: float,
    imgsz: int,
    show: bool = False,
) -> None:
    """Run YOLO on a video and write an annotated copy."""
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
    total_frames = 0
    total_detections_before_filtering = 0
    total_detections_after_filtering = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        detections, raw_detection_count = detector.detect(frame)
        total_frames += 1
        total_detections_before_filtering += raw_detection_count
        total_detections_after_filtering += len(detections)

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
    args = parser.parse_args()

    try:
        video_path = Path(args.video)
        if args.random_soccernet:
            video_path = find_random_video(args.soccernet_dir)
            print(f"Selected SoccerNet video: {video_path}")

        process_video(
            video_path=video_path,
            output_path=Path(args.output),
            model_path=args.model,
            conf=args.conf,
            imgsz=args.imgsz,
            show=args.show,
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError) as error:
        print(f"Error: {error}")
        return

    print(f"Annotated video saved to: {args.output}")


if __name__ == "__main__":
    main()
