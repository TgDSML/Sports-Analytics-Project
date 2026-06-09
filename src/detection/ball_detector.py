"""Ball detection baseline using a YOLO model."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils.io import write_ball_detections_csv


BALL_CLASS_NAMES = {
    "ball",
    "football",
    "soccer ball",
    "sports ball",
}


@dataclass
class BallDetection:
    """Single ball candidate detection."""

    frame: int
    timestamp: float
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> tuple[float, float]:
        """Return detection center in image coordinates."""
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2, (y1 + y2) / 2

    @property
    def width(self) -> int:
        """Return bounding box width."""
        x1, _, x2, _ = self.bbox
        return x2 - x1

    @property
    def height(self) -> int:
        """Return bounding box height."""
        _, y1, _, y2 = self.bbox
        return y2 - y1

    @property
    def area(self) -> int:
        """Return bounding box area."""
        return self.width * self.height

    def to_csv_row(self) -> dict:
        """Return a stable CSV row."""
        x1, y1, x2, y2 = self.bbox
        center_x, center_y = self.center
        return {
            "frame": self.frame,
            "timestamp": self.timestamp,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "center_x": center_x,
            "center_y": center_y,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass
class BallFilterConfig:
    """Configurable post-detection filters for ball candidates."""

    min_area: int = 20
    max_area: int = 500
    min_width: int = 4
    max_width: int = 30
    min_height: int = 4
    max_height: int = 30
    max_detections_per_frame: int = 1
    exclude_top_ratio: float = 0.08
    exclude_bottom_ratio: float = 0.0


class YOLOBallDetector:
    """Small wrapper around YOLO for ball-capable detection models."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.10,
        imgsz: int = 1280,
        target_class_names: set[str] | None = None,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.target_class_names = target_class_names or BALL_CLASS_NAMES
        self.available_ball_classes = self._available_ball_classes()

    def detect(self, frame, frame_index: int, fps: float) -> tuple[list[BallDetection], int]:
        """Detect raw ball-class candidates in one frame."""
        results = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections = []
        raw_detection_count = 0
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                raw_detection_count += 1
                class_id = int(box.cls[0])
                class_name = str(self.model.names[class_id])
                if class_name.lower() not in self.target_class_names:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    BallDetection(
                        frame=frame_index,
                        timestamp=frame_index / fps,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(box.conf[0]),
                        class_id=class_id,
                        class_name=class_name,
                    )
                )

        return detections, raw_detection_count

    def _available_ball_classes(self) -> dict[int, str]:
        """Return model classes that look ball-capable."""
        return {
            int(class_id): str(class_name)
            for class_id, class_name in self.model.names.items()
            if str(class_name).lower() in self.target_class_names
        }


def process_ball_detection_video(
    video_path: Path,
    raw_output_csv_path: Path,
    filtered_output_csv_path: Path,
    output_video_path: Path,
    summary_csv_path: Path,
    summary_md_path: Path,
    model_path: str = "yolov8n.pt",
    conf: float = 0.10,
    imgsz: int = 1280,
    filter_config: BallFilterConfig | None = None,
    debug_dir: Path | None = None,
    debug_frame_stride: int = 0,
) -> tuple[int, int]:
    """Run ball detection, export raw/filtered rows, diagnostics, and video."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    filter_config = filter_config or BallFilterConfig()
    raw_output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_md_path.parent.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None and debug_frame_stride > 0:
        debug_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 30

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_video_path}")

    detector = YOLOBallDetector(model_path=model_path, conf=conf, imgsz=imgsz)
    print(f"Processing ball detection video: {video_path}")
    print(f"Writing raw ball detections CSV: {raw_output_csv_path}")
    print(f"Writing filtered ball detections CSV: {filtered_output_csv_path}")
    print(f"Writing filtered ball annotated output: {output_video_path}")
    if detector.available_ball_classes:
        classes = ", ".join(
            f"{class_id}:{class_name}"
            for class_id, class_name in detector.available_ball_classes.items()
        )
        print(f"Ball-capable model classes found: {classes}")
    else:
        print(
            "Warning: this model does not expose a ball-like class "
            "(ball, football, soccer ball, sports ball). The CSV may be empty."
        )

    raw_rows = []
    filtered_rows = []
    raw_by_frame = {}
    filtered_by_frame = {}
    removal_reasons = Counter()
    total_frames = 0
    total_raw_detections = 0
    total_raw_ball_detections = 0
    total_filtered_ball_detections = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        raw_detections, raw_detection_count = detector.detect(frame, total_frames, fps)
        filtered_detections, reasons = filter_ball_detections(
            raw_detections,
            frame_width=width,
            frame_height=height,
            config=filter_config,
        )
        total_raw_detections += raw_detection_count
        total_raw_ball_detections += len(raw_detections)
        total_filtered_ball_detections += len(filtered_detections)
        raw_by_frame[total_frames] = raw_detections
        filtered_by_frame[total_frames] = filtered_detections
        removal_reasons.update(reasons)
        raw_rows.extend(detection.to_csv_row() for detection in raw_detections)
        filtered_rows.extend(detection.to_csv_row() for detection in filtered_detections)

        annotated_frame = draw_ball_detections(frame, filtered_detections)
        writer.write(annotated_frame)
        if debug_dir is not None and debug_frame_stride > 0:
            if total_frames % debug_frame_stride == 0 and raw_detections:
                debug_frame = draw_ball_detections(
                    frame,
                    raw_detections,
                    label_prefix="raw",
                    color=(0, 165, 255),
                )
                debug_frame = draw_ball_detections(
                    debug_frame,
                    filtered_detections,
                    label_prefix="kept",
                    color=(0, 255, 255),
                    copy_frame=False,
                )
                cv2.imwrite(str(debug_dir / f"ball_debug_frame_{total_frames:04d}.jpg"), debug_frame)
        total_frames += 1

    capture.release()
    writer.release()

    write_ball_detections_csv(raw_rows, raw_output_csv_path)
    write_ball_detections_csv(filtered_rows, filtered_output_csv_path)
    write_ball_detection_diagnostics(
        summary_csv_path=summary_csv_path,
        summary_md_path=summary_md_path,
        raw_by_frame=raw_by_frame,
        filtered_by_frame=filtered_by_frame,
        frame_count=total_frames,
        filter_config=filter_config,
        removal_reasons=removal_reasons,
    )
    average_raw = total_raw_ball_detections / total_frames if total_frames else 0
    average_filtered = total_filtered_ball_detections / total_frames if total_frames else 0
    print(f"Total frames processed for ball detection: {total_frames}")
    print(f"Total raw YOLO detections before ball-class filtering: {total_raw_detections}")
    print(f"Raw ball-class detections: {total_raw_ball_detections}")
    print(f"Filtered ball detections: {total_filtered_ball_detections}")
    print(f"Average raw ball detections per frame: {average_raw:.3f}")
    print(f"Average filtered ball detections per frame: {average_filtered:.3f}")
    print(f"Ball diagnostics saved to: {summary_csv_path}")
    print(f"Ball diagnostics report saved to: {summary_md_path}")
    if total_filtered_ball_detections == 0:
        print(
            "No ball detections were exported. With generic COCO YOLO weights, "
            "this usually means the football was too small/blurred/occluded or "
            "the model did not fire on the sports ball class."
        )

    return total_raw_ball_detections, total_filtered_ball_detections


def filter_ball_detections(
    detections: list[BallDetection],
    frame_width: int,
    frame_height: int,
    config: BallFilterConfig,
) -> tuple[list[BallDetection], list[str]]:
    """Apply configurable quality filters without linking detections over time."""
    kept = []
    reasons = []
    top_limit = frame_height * config.exclude_top_ratio
    bottom_limit = frame_height * (1 - config.exclude_bottom_ratio)

    for detection in detections:
        _, center_y = detection.center
        if detection.area < config.min_area:
            reasons.append("area_below_min")
            continue
        if detection.area > config.max_area:
            reasons.append("area_above_max")
            continue
        if detection.width < config.min_width:
            reasons.append("width_below_min")
            continue
        if detection.width > config.max_width:
            reasons.append("width_above_max")
            continue
        if detection.height < config.min_height:
            reasons.append("height_below_min")
            continue
        if detection.height > config.max_height:
            reasons.append("height_above_max")
            continue
        if center_y < top_limit:
            reasons.append("excluded_top_region")
            continue
        if config.exclude_bottom_ratio > 0 and center_y > bottom_limit:
            reasons.append("excluded_bottom_region")
            continue
        kept.append(detection)

    if config.max_detections_per_frame > 0 and len(kept) > config.max_detections_per_frame:
        kept = sorted(kept, key=lambda item: item.confidence, reverse=True)
        removed = len(kept) - config.max_detections_per_frame
        reasons.extend(["per_frame_confidence_cap"] * removed)
        kept = kept[: config.max_detections_per_frame]

    return kept, reasons


def write_ball_detection_diagnostics(
    summary_csv_path: Path,
    summary_md_path: Path,
    raw_by_frame: dict[int, list[BallDetection]],
    filtered_by_frame: dict[int, list[BallDetection]],
    frame_count: int,
    filter_config: BallFilterConfig,
    removal_reasons: Counter,
) -> None:
    """Write ball detection count and distribution diagnostics."""
    import csv

    rows = []
    raw_counts = [len(raw_by_frame.get(frame, [])) for frame in range(frame_count)]
    filtered_counts = [len(filtered_by_frame.get(frame, [])) for frame in range(frame_count)]
    raw_detections = [d for detections in raw_by_frame.values() for d in detections]
    filtered_detections = [d for detections in filtered_by_frame.values() for d in detections]

    def add(section: str, metric: str, value) -> None:
        rows.append({"section": section, "metric": metric, "value": value})

    add("counts", "frames", frame_count)
    add("counts", "raw_detections", len(raw_detections))
    add("counts", "filtered_detections", len(filtered_detections))
    add("counts", "raw_frames_zero", sum(count == 0 for count in raw_counts))
    add("counts", "raw_frames_one", sum(count == 1 for count in raw_counts))
    add("counts", "raw_frames_multiple", sum(count > 1 for count in raw_counts))
    add("counts", "filtered_frames_zero", sum(count == 0 for count in filtered_counts))
    add("counts", "filtered_frames_one", sum(count == 1 for count in filtered_counts))
    add("counts", "filtered_frames_multiple", sum(count > 1 for count in filtered_counts))

    for key, value in filter_config.__dict__.items():
        add("filters", key, value)
    for reason, count in sorted(removal_reasons.items()):
        add("removal_reasons", reason, count)

    for label, detections in (("raw", raw_detections), ("filtered", filtered_detections)):
        for metric, values in _distribution_values(detections).items():
            add(f"{label}_distribution", metric, values)

    for frame in range(frame_count):
        add("raw_frame_counts", frame, raw_counts[frame])
        add("filtered_frame_counts", frame, filtered_counts[frame])

    with summary_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    summary_md_path.write_text(
        _build_diagnostics_markdown(
            frame_count=frame_count,
            raw_counts=raw_counts,
            filtered_counts=filtered_counts,
            raw_detections=raw_detections,
            filtered_detections=filtered_detections,
            filter_config=filter_config,
            removal_reasons=removal_reasons,
        ),
        encoding="utf-8",
    )


def _distribution_values(detections: list[BallDetection]) -> dict[str, str]:
    """Return compact distribution metrics for confidence and box geometry."""
    if not detections:
        return {
            "confidence": "empty",
            "width": "empty",
            "height": "empty",
            "area": "empty",
        }

    metrics = {
        "confidence": np.array([d.confidence for d in detections], dtype=float),
        "width": np.array([d.width for d in detections], dtype=float),
        "height": np.array([d.height for d in detections], dtype=float),
        "area": np.array([d.area for d in detections], dtype=float),
    }
    return {name: _format_distribution(values) for name, values in metrics.items()}


def _format_distribution(values: np.ndarray) -> str:
    """Format min/p25/median/p75/max for diagnostics."""
    percentiles = np.percentile(values, [0, 25, 50, 75, 100])
    return (
        f"min={percentiles[0]:.4g},p25={percentiles[1]:.4g},"
        f"median={percentiles[2]:.4g},p75={percentiles[3]:.4g},max={percentiles[4]:.4g}"
    )


def _build_diagnostics_markdown(
    frame_count: int,
    raw_counts: list[int],
    filtered_counts: list[int],
    raw_detections: list[BallDetection],
    filtered_detections: list[BallDetection],
    filter_config: BallFilterConfig,
    removal_reasons: Counter,
) -> str:
    """Build a readable diagnostics report."""
    lines = [
        "# Ball Detection Diagnostics",
        "",
        "## Counts",
        "",
        f"- Frames: {frame_count}",
        f"- Raw ball detections: {len(raw_detections)}",
        f"- Filtered ball detections: {len(filtered_detections)}",
        f"- Raw frames with 0 / 1 / multiple detections: "
        f"{sum(c == 0 for c in raw_counts)} / {sum(c == 1 for c in raw_counts)} / {sum(c > 1 for c in raw_counts)}",
        f"- Filtered frames with 0 / 1 / multiple detections: "
        f"{sum(c == 0 for c in filtered_counts)} / {sum(c == 1 for c in filtered_counts)} / {sum(c > 1 for c in filtered_counts)}",
        "",
        "## Filters",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in filter_config.__dict__.items())
    lines.extend(["", "## Removal Reasons", ""])
    if removal_reasons:
        lines.extend(f"- {reason}: {count}" for reason, count in sorted(removal_reasons.items()))
    else:
        lines.append("- None")
    lines.extend(["", "## Distributions", ""])
    for label, detections in (("Raw", raw_detections), ("Filtered", filtered_detections)):
        lines.append(f"### {label}")
        for metric, value in _distribution_values(detections).items():
            lines.append(f"- {metric}: {value}")
        lines.append("")
    return "\n".join(lines)


def draw_ball_detections(
    frame,
    detections: list[BallDetection],
    label_prefix: str = "ball",
    color=(0, 255, 255),
    copy_frame: bool = True,
):
    """Draw compact ball marks on a copy of the frame."""
    annotated_frame = frame.copy() if copy_frame else frame
    frame_height, frame_width = frame.shape[:2]
    scale = frame_height / 720
    radius = max(5, int(7 * scale))
    thickness = max(2, int(2 * scale))
    font_scale = max(0.3, 0.45 * scale)
    padding = max(2, int(4 * scale))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        center_x, center_y = detection.center
        center = (int(center_x), int(center_y))
        label = f"{label_prefix} {detection.confidence:.2f}"

        cv2.circle(annotated_frame, center, radius, color, thickness)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, max(1, thickness - 1))

        label_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_width, text_height = label_size
        label_x = max(0, min(center[0] + radius + padding, frame_width - text_width - 2 * padding))
        label_y = max(text_height + padding, center[1] - radius - padding)
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
