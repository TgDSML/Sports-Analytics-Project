"""YOLO object detection helpers."""

from dataclasses import dataclass

import cv2
from ultralytics import YOLO


@dataclass
class Detection:
    """Single filtered object detection."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Backward-compatible alias for the bounding box."""
        return self.bbox


class YOLODetector:
    """Thin wrapper around Ultralytics YOLO for sports analytics detection."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.15,
        imgsz: int = 640,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.target_class_names = {"person"}

    def detect(self, frame) -> tuple[list[Detection], int]:
        """Detect target classes in a frame.

        Returns filtered detections and the raw YOLO detection count before
        class filtering.
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections = []
        raw_detection_count = 0
        for result in results:
            for box in result.boxes:
                raw_detection_count += 1
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                if class_name not in self.target_class_names:
                    continue

                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                    )
                )

        return detections, raw_detection_count

    def draw_detections(self, frame, detections: list[Detection]):
        """Draw scaled boxes and labels on a copy of the frame."""
        annotated_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        scale = frame_height / 720
        font_scale = max(0.3, 0.5 * scale)
        thickness = max(1, int(2 * scale))
        color = (0, 255, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            label = f"Player {detection.confidence:.2f}"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

            label_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
            text_width, text_height = label_size
            padding = max(2, int(4 * scale))

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
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return annotated_frame
