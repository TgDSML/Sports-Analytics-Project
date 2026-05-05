"""Simple centroid-based player tracking."""

from dataclasses import dataclass
from math import dist


@dataclass
class Track:
    """Current state for one tracked player."""

    track_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    missing: int = 0


class CentroidTracker:
    """Assign stable IDs to person detections using nearest centroids."""

    def __init__(
        self,
        max_distance: float = 120,
        max_missing: int = 30,
        smoothing: float = 0.7,
        min_box_area: int = 100,
    ) -> None:
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.smoothing = min(1.0, max(0.0, smoothing))
        self.min_box_area = min_box_area
        self.next_track_id = 1
        self.tracks: dict[int, Track] = {}

    @property
    def total_tracks_created(self) -> int:
        """Return the number of track IDs allocated so far."""
        return self.next_track_id - 1

    def update(self, detections) -> list[Track]:
        """Update tracks from person detections and return active matches."""
        person_detections = [
            detection
            for detection in detections
            if detection.class_name == "person"
            and self._box_area(detection.bbox) >= self.min_box_area
        ]

        if not person_detections:
            self._mark_missing(set())
            return []

        detections_with_centroids = [
            (detection, self._centroid(detection.bbox)) for detection in person_detections
        ]

        matched_track_ids: set[int] = set()
        matched_detection_indexes: set[int] = set()
        active_tracks: list[Track] = []

        candidates = []
        for track_id, track in self.tracks.items():
            for detection_index, (_, centroid) in enumerate(detections_with_centroids):
                distance = dist(track.centroid, centroid)
                candidates.append((distance, track_id, detection_index))

        for distance, track_id, detection_index in sorted(candidates):
            if distance > self.max_distance:
                continue
            if track_id in matched_track_ids or detection_index in matched_detection_indexes:
                continue

            detection, centroid = detections_with_centroids[detection_index]
            track = self._update_track(track_id, detection, centroid)
            matched_track_ids.add(track_id)
            matched_detection_indexes.add(detection_index)
            active_tracks.append(track)

        self._mark_missing(matched_track_ids)

        for detection_index, (detection, centroid) in enumerate(detections_with_centroids):
            if detection_index in matched_detection_indexes:
                continue
            active_tracks.append(self._register_track(detection, centroid))

        return active_tracks

    def _register_track(self, detection, centroid: tuple[float, float]) -> Track:
        track = Track(
            track_id=self.next_track_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            bbox=detection.bbox,
            centroid=centroid,
        )
        self.tracks[track.track_id] = track
        self.next_track_id += 1
        return track

    def _update_track(self, track_id: int, detection, centroid: tuple[float, float]) -> Track:
        track = self.tracks[track_id]
        smoothed_bbox = self._smooth_bbox(track.bbox, detection.bbox)
        track.class_name = detection.class_name
        track.confidence = detection.confidence
        track.bbox = smoothed_bbox
        track.centroid = self._centroid(smoothed_bbox)
        track.missing = 0
        return track

    def _mark_missing(self, matched_track_ids: set[int]) -> None:
        for track_id in list(self.tracks):
            if track_id in matched_track_ids:
                continue
            self.tracks[track_id].missing += 1
            if self.tracks[track_id].missing > self.max_missing:
                del self.tracks[track_id]

    @staticmethod
    def _centroid(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @staticmethod
    def _box_area(bbox: tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _smooth_bbox(
        self,
        previous_bbox: tuple[int, int, int, int],
        new_bbox: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        smoothed = [
            (self.smoothing * previous) + ((1 - self.smoothing) * new)
            for previous, new in zip(previous_bbox, new_bbox)
        ]
        return tuple(int(round(value)) for value in smoothed)
