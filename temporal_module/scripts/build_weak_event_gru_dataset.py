"""Build weakly labeled sequence windows for a pilot GRU event classifier."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLASSES = ["background", "carry", "pass", "turnover", "shot"]
EVENT_CLASSES = ["carry", "pass", "turnover", "shot"]
PRECEDENCE = {"shot": 4, "turnover": 3, "pass": 2, "carry": 1, "background": 0}
DEFAULT_OUTPUT_DIR = Path("temporal_module") / "data" / "modeling" / "weak_event_gru"
LEAKAGE_TOKENS = [
    "event",
    "label",
    "candidate",
    "confidence_tier",
    "quality",
    "review",
    "gold",
    "manual",
    "score",
    "source",
    "priority",
    "overlap",
    "refinement",
]
ID_COLUMNS = {"clip_id", "frame", "timestamp", "possessor_track_id"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a weakly supervised GRU dataset from existing temporal/event artifacts."
    )
    parser.add_argument("--derived-root", default=str(Path("temporal_module") / "data" / "derived"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--label-region-seconds", type=float, default=1.0)
    parser.add_argument("--default-fps", type=float, default=25.0)
    parser.add_argument("--stride-seconds", type=float, default=2.0)
    parser.add_argument("--background-min-distance-seconds", type=float, default=2.0)
    parser.add_argument("--background-center-margin-seconds", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-clips", type=int, default=5)
    parser.add_argument("--val-clips", type=int, default=1)
    parser.add_argument("--test-clips", type=int, default=2)
    parser.add_argument(
        "--balance-strategy",
        choices=["undersample", "none"],
        default="undersample",
        help="Use deterministic per-class undersampling by default.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def infer_fps(frames: pd.DataFrame, default_fps: float) -> float:
    if "timestamp" not in frames.columns:
        return default_fps
    timestamps = pd.to_numeric(frames["timestamp"], errors="coerce").dropna().to_numpy()
    if len(timestamps) < 3:
        return default_fps
    diffs = np.diff(timestamps)
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if len(diffs) == 0:
        return default_fps
    fps = 1.0 / float(np.median(diffs))
    if not math.isfinite(fps) or fps <= 0:
        return default_fps
    return fps


def truthy(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def parse_int(value: Any, default: int = 0) -> int:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return default
    return int(round(float(parsed)))


def clip_dirs(derived_root: Path) -> list[Path]:
    return sorted(
        path
        for path in derived_root.iterdir()
        if path.is_dir() and (path / "temporal_frames.csv").exists()
    )


def normalize_event_type(value: Any) -> str:
    text = str(value).strip().casefold()
    if text.startswith("pass"):
        return "pass"
    if text.startswith("turnover") or text.startswith("interception"):
        return "turnover"
    if text.startswith("shot"):
        return "shot"
    if text.startswith("carry"):
        return "carry"
    return ""


def load_events(clip_dir: Path) -> list[dict[str, Any]]:
    path = clip_dir / "events" / "event_candidates_unified.csv"
    if not path.exists():
        return []
    events = read_csv(path)
    rows: list[dict[str, Any]] = []
    for idx, row in events.iterrows():
        target = normalize_event_type(row.get("event_type", ""))
        if target not in EVENT_CLASSES:
            continue
        duplicate_of = str(row.get("duplicate_of_unified_event_id", "")).strip()
        if duplicate_of and duplicate_of.lower() != "nan":
            continue
        center = parse_int(row.get("center_frame", row.get("start_frame", 0)))
        start = parse_int(row.get("start_frame", center), center)
        end = parse_int(row.get("end_frame", center), center)
        event_id = row.get("unified_event_id", row.get("canonical_event_id", idx + 1))
        rows.append(
            {
                "target_class": target,
                "event_start": min(start, end),
                "event_end": max(start, end),
                "center_frame": center,
                "candidate_id": str(event_id),
                "label_source": "weak_unified_event_candidates",
                "source_event_type": str(row.get("source_event_type", row.get("event_type", ""))),
                "is_eligible_for_weak_label": int(truthy(row.get("is_eligible_for_weak_label", ""))),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            item["center_frame"],
            -PRECEDENCE[item["target_class"]],
            item["candidate_id"],
        ),
    )


def clamp_window(center: int, window_frames: int, min_frame: int, max_frame: int) -> tuple[int, int]:
    half = window_frames // 2
    start = center - half
    end = start + window_frames - 1
    if start < min_frame:
        end += min_frame - start
        start = min_frame
    if end > max_frame:
        start -= end - max_frame
        end = max_frame
    start = max(min_frame, start)
    end = min(max_frame, end)
    return int(start), int(end)


def centered_window_with_padding(
    center: int,
    window_frames: int,
    min_frame: int,
    max_frame: int,
) -> tuple[int, int, int, int, int, int]:
    half = window_frames // 2
    desired_start = int(center) - half
    desired_end = desired_start + window_frames - 1
    valid_start = max(min_frame, desired_start)
    valid_end = min(max_frame, desired_end)
    pad_before = max(0, min_frame - desired_start)
    pad_after = max(0, desired_end - max_frame)
    return int(desired_start), int(desired_end), int(valid_start), int(valid_end), int(pad_before), int(pad_after)


def centered_label_region(center: int, label_region_frames: int) -> tuple[int, int]:
    half = label_region_frames // 2
    start = int(center) - half
    end = start + label_region_frames - 1
    return int(start), int(end)


def deduplicate_same_center(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for event in events:
        center = int(event["center_frame"])
        current = selected.get(center)
        if current is None:
            selected[center] = event
            continue
        current_key = (-PRECEDENCE[current["target_class"]], current["candidate_id"])
        new_key = (-PRECEDENCE[event["target_class"]], event["candidate_id"])
        if new_key < current_key:
            selected[center] = event
    return [selected[key] for key in sorted(selected)]


def spans_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a <= end_b and start_b <= end_a


def overlapping_events(
    start_frame: int,
    end_frame: int,
    events: list[dict[str, Any]],
    margin_frames: int = 0,
) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if spans_overlap(
            start_frame,
            end_frame,
            int(event["event_start"]) - margin_frames,
            int(event["event_end"]) + margin_frames,
        )
    ]


def clip_class_report(clips: list[str], windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_clip: dict[str, Counter[str]] = {clip: Counter() for clip in clips}
    for window in windows:
        by_clip[window["clip_id"]][window["target_class"]] += 1
    for clip in clips:
        counts = by_clip[clip]
        row: dict[str, Any] = {
            "clip_id": clip,
            "total_windows": sum(counts.values()),
            "classes_present": ",".join(class_name for class_name in CLASSES if counts.get(class_name, 0) > 0),
        }
        for class_name in CLASSES:
            row[f"{class_name}_windows"] = counts.get(class_name, 0)
        rows.append(row)
    return rows


def generate_windows_for_clip(
    clip_dir: Path,
    frames: pd.DataFrame,
    events: list[dict[str, Any]],
    window_frames: int,
    label_region_frames: int,
    stride_frames: int,
    background_center_margin_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clip_id = clip_dir.name
    frame_values = pd.to_numeric(frames["frame"], errors="coerce").dropna().astype(int)
    rejection_rows: list[dict[str, Any]] = []
    if frame_values.empty:
        rejection_rows.append(
            {
                "clip_id": clip_id,
                "window_kind": "clip",
                "candidate_start_frame": "",
                "candidate_end_frame": "",
                "reason": "missing_temporal_data",
                "detail": "temporal_frames.csv has no numeric frame values",
            }
        )
        return [], rejection_rows
    min_frame = int(frame_values.min())
    max_frame = int(frame_values.max())
    available_frames = max_frame - min_frame + 1
    windows: list[dict[str, Any]] = []
    resolved_events = deduplicate_same_center(events)
    for event in resolved_events:
        start, end, valid_start, valid_end, pad_before, pad_after = centered_window_with_padding(
            event["center_frame"],
            window_frames,
            min_frame,
            max_frame,
        )
        label_start, label_end = centered_label_region(event["center_frame"], label_region_frames)
        if not (label_start <= int(event["center_frame"]) <= label_end):
            rejection_rows.append(
                {
                    "clip_id": clip_id,
                    "window_kind": "event",
                    "candidate_start_frame": label_start,
                    "candidate_end_frame": label_end,
                    "reason": "event_midpoint_outside_label_region",
                    "detail": f"event {event['candidate_id']} midpoint is outside its label region",
                }
            )
            continue
        observed_frames = max(0, valid_end - valid_start + 1)
        padding_frames = pad_before + pad_after
        if observed_frames + padding_frames != window_frames:
            rejection_rows.append(
                {
                    "clip_id": clip_id,
                    "window_kind": "event",
                    "candidate_start_frame": valid_start,
                    "candidate_end_frame": valid_end,
                    "reason": "insufficient_frames",
                    "detail": (
                        f"event window has {observed_frames} observed frames and "
                        f"{padding_frames} padding frames; requires {window_frames}"
                    ),
                }
            )
            continue
        windows.append(
            {
                "clip_id": clip_id,
                "input_start_frame": start,
                "input_end_frame": end,
                "start_frame": start,
                "end_frame": end,
                "valid_start_frame": valid_start,
                "valid_end_frame": valid_end,
                "center_frame": int(event["center_frame"]),
                "label_start_frame": label_start,
                "label_end_frame": label_end,
                "label_center_frame": int(event["center_frame"]),
                "window_frames": window_frames,
                "label_region_frames": label_region_frames,
                "observed_frames": observed_frames,
                "padding_frames": padding_frames,
                "padding_before_frames": pad_before,
                "padding_after_frames": pad_after,
                "padding_percent": round((padding_frames / window_frames) * 100.0, 6),
                "context_contains_other_event": int(
                    any(
                        other["candidate_id"] != event["candidate_id"]
                        for other in overlapping_events(start, end, events)
                    )
                ),
                "target_class": event["target_class"],
                "label_source": event["label_source"],
                "candidate_id": event["candidate_id"],
                "source_event_type": event["source_event_type"],
                "is_eligible_for_weak_label": event["is_eligible_for_weak_label"],
                "overlap_resolution": "same_center_precedence_shot_turnover_pass_carry",
                "selected_after_balancing": 0,
                "split": "",
            }
        )

    if available_frames < window_frames:
        rejection_rows.append(
            {
                "clip_id": clip_id,
                "window_kind": "background",
                "candidate_start_frame": "",
                "candidate_end_frame": "",
                "reason": "insufficient_frames",
                "detail": f"clip has {available_frames} frames; requires {window_frames}",
            }
        )
        return sorted(
            windows,
            key=lambda item: (
                item["clip_id"],
                item["center_frame"],
                -PRECEDENCE[item["target_class"]],
                item["candidate_id"],
            ),
        ), rejection_rows

    background_count = 0
    background_with_context_event_count = 0
    last_start = max_frame - window_frames + 1
    for start in range(min_frame, last_start + 1, max(1, stride_frames)):
        end = start + window_frames - 1
        center = start + window_frames // 2
        label_start, label_end = centered_label_region(center, label_region_frames)
        center_overlaps = overlapping_events(label_start, label_end, events)
        if center_overlaps:
            rejection_rows.append(
                {
                    "clip_id": clip_id,
                    "window_kind": "background",
                    "candidate_start_frame": label_start,
                    "candidate_end_frame": label_end,
                    "reason": "center_label_region_overlaps_event",
                    "detail": ";".join(event["candidate_id"] for event in center_overlaps),
                }
            )
            continue
        nearby_events = overlapping_events(label_start, label_end, events, background_center_margin_frames)
        if nearby_events:
            rejection_rows.append(
                {
                    "clip_id": clip_id,
                    "window_kind": "background",
                    "candidate_start_frame": label_start,
                    "candidate_end_frame": label_end,
                    "reason": "center_label_region_too_close_to_event",
                    "detail": ";".join(event["candidate_id"] for event in nearby_events),
                }
            )
            continue
        context_events = overlapping_events(start, end, events)
        context_contains_other_event = int(bool(context_events))
        if context_contains_other_event:
            background_with_context_event_count += 1
            rejection_rows.append(
                {
                    "clip_id": clip_id,
                    "window_kind": "background",
                    "candidate_start_frame": start,
                    "candidate_end_frame": end,
                    "reason": "accepted_with_noncentral_context_event",
                    "detail": ";".join(event["candidate_id"] for event in context_events),
                }
            )
        background_count += 1
        windows.append(
            {
                "clip_id": clip_id,
                "input_start_frame": start,
                "input_end_frame": end,
                "start_frame": start,
                "end_frame": end,
                "valid_start_frame": start,
                "valid_end_frame": end,
                "center_frame": center,
                "label_start_frame": label_start,
                "label_end_frame": label_end,
                "label_center_frame": center,
                "window_frames": window_frames,
                "label_region_frames": label_region_frames,
                "observed_frames": window_frames,
                "padding_frames": 0,
                "padding_before_frames": 0,
                "padding_after_frames": 0,
                "padding_percent": 0.0,
                "context_contains_other_event": context_contains_other_event,
                "target_class": "background",
                "label_source": "weak_background_center_label_region",
                "candidate_id": "",
                "source_event_type": "",
                "is_eligible_for_weak_label": 1,
                "overlap_resolution": "background_label_region_excludes_event_spans_with_margin",
                "selected_after_balancing": 0,
                "split": "",
            }
        )
    if background_count == 0:
        rejection_rows.append(
            {
                "clip_id": clip_id,
                "window_kind": "background",
                "candidate_start_frame": "",
                "candidate_end_frame": "",
                "reason": "no_valid_background_window",
                "detail": "all stride-aligned background starts were rejected or unavailable",
            }
        )
    return sorted(
        windows,
        key=lambda item: (
            item["clip_id"],
            item["center_frame"],
            -PRECEDENCE[item["target_class"]],
            item["candidate_id"],
        ),
    ), rejection_rows


def numeric_feature_columns(frames_by_clip: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
    excluded: set[str] = set()
    candidates: list[str] = []
    first = next(iter(frames_by_clip.values()))
    for column in first.columns:
        lowered = column.casefold()
        if column in ID_COLUMNS or column.endswith("_track_id") or lowered.endswith("_id"):
            excluded.add(column)
            continue
        if any(token in lowered for token in LEAKAGE_TOKENS):
            excluded.add(column)
            continue
        series = pd.to_numeric(first[column], errors="coerce")
        if series.notna().any():
            candidates.append(column)
        else:
            excluded.add(column)
    stable: list[str] = []
    for column in candidates:
        if all(column in df.columns for df in frames_by_clip.values()):
            stable.append(column)
        else:
            excluded.add(column)
    return stable, sorted(excluded)


def balance_windows(windows: list[dict[str, Any]], strategy: str, seed: int) -> list[dict[str, Any]]:
    for window in windows:
        window["selected_after_balancing"] = 0
    if strategy == "none":
        for window in windows:
            window["selected_after_balancing"] = 1
        return windows
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        by_class[window["target_class"]].append(window)
    nonzero_counts = [len(by_class[class_name]) for class_name in CLASSES if by_class[class_name]]
    if not nonzero_counts:
        return windows
    cap = min(nonzero_counts)
    rng = random.Random(seed)
    selected_ids: set[int] = set()
    for class_name in CLASSES:
        class_windows = list(by_class[class_name])
        class_windows.sort(key=lambda item: (item["clip_id"], item["center_frame"], item["candidate_id"]))
        if len(class_windows) > cap:
            class_windows = rng.sample(class_windows, cap)
        for window in class_windows:
            selected_ids.add(id(window))
    for window in windows:
        window["selected_after_balancing"] = int(id(window) in selected_ids)
    return windows


def balance_windows_by_split(windows: list[dict[str, Any]], strategy: str, seed: int) -> list[dict[str, Any]]:
    for window in windows:
        window["selected_after_balancing"] = 0
    for split_index, split in enumerate(["train", "val", "test"]):
        split_windows = [window for window in windows if window.get("split") == split]
        balance_windows(split_windows, strategy, seed + split_index)
    return windows


def split_clips(
    clips: list[str],
    windows: list[dict[str, Any]],
    seed: int,
    train_count: int,
    val_count: int,
    test_count: int,
) -> tuple[dict[str, str], list[str]]:
    if len(clips) < train_count + val_count + test_count:
        raise RuntimeError(
            f"Need at least {train_count + val_count + test_count} clips for the default split; found {len(clips)}."
        )
    clip_classes: dict[str, set[str]] = {clip: set() for clip in clips}
    for window in windows:
        clip_classes[window["clip_id"]].add(window["target_class"])

    all_classes_available = set().union(*(clip_classes[clip] for clip in clips))
    impossible_missing = [class_name for class_name in CLASSES if class_name not in all_classes_available]
    if impossible_missing:
        raise RuntimeError(
            "No valid split can provide all required training classes. "
            f"Missing from all processable clips: {', '.join(impossible_missing)}. "
            f"Class-by-clip report: {json.dumps(clip_class_report(clips, windows), sort_keys=True)}"
        )

    rng = random.Random(seed)
    clip_rank = {clip: index for index, clip in enumerate(clips)}
    shuffled = clips[:]
    rng.shuffle(shuffled)
    random_rank = {clip: index for index, clip in enumerate(shuffled)}
    valid_assignments: list[tuple[tuple[Any, ...], list[str], list[str], list[str]]] = []
    for train_tuple in itertools.combinations(clips, train_count):
        train = list(train_tuple)
        train_classes = set().union(*(clip_classes[clip] for clip in train))
        if not all(class_name in train_classes for class_name in CLASSES):
            continue
        remaining = [clip for clip in clips if clip not in train]
        for val_tuple in itertools.combinations(remaining, val_count):
            val = list(val_tuple)
            test = [clip for clip in remaining if clip not in val]
            if len(test) != test_count:
                continue
            val_classes = set().union(*(clip_classes[clip] for clip in val)) if val else set()
            test_classes = set().union(*(clip_classes[clip] for clip in test)) if test else set()
            score = (
                -len(train_classes),
                -len(val_classes),
                -len(test_classes),
                sum(random_rank[clip] for clip in train),
                sum(random_rank[clip] for clip in val),
                tuple(random_rank[clip] for clip in train),
                tuple(random_rank[clip] for clip in val),
                tuple(clip_rank[clip] for clip in train),
            )
            valid_assignments.append((score, train, val, test))
    if not valid_assignments:
        raise RuntimeError(
            "No valid 5/1/2 split can provide all required training classes. "
            f"Class-by-clip report: {json.dumps(clip_class_report(clips, windows), sort_keys=True)}"
        )
    _, train, val, test = sorted(valid_assignments, key=lambda item: item[0])[0]
    assignment = {clip: "train" for clip in train}
    assignment.update({clip: "val" for clip in val})
    assignment.update({clip: "test" for clip in test})
    warnings: list[str] = []
    for window in windows:
        window["split"] = assignment.get(window["clip_id"], "")
    train_classes = {window["target_class"] for window in windows if assignment.get(window["clip_id"]) == "train"}
    missing_train = [class_name for class_name in CLASSES if class_name not in train_classes]
    if missing_train:
        raise RuntimeError(
            f"Required class(es) absent from train split: {', '.join(missing_train)}. "
            f"Class-by-clip report: {json.dumps(clip_class_report(clips, windows), sort_keys=True)}"
        )
    for split in ["val", "test"]:
        split_classes = {window["target_class"] for window in windows if assignment.get(window["clip_id"]) == split}
        missing = [class_name for class_name in CLASSES if class_name not in split_classes]
        if missing:
            warnings.append(f"{split} split has no selected windows for: {', '.join(missing)}")
    return assignment, warnings


def class_count_rows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase, selected_value in [("before_balancing", None), ("after_balancing", 1)]:
        phase_windows = windows if selected_value is None else [w for w in windows if int(w["selected_after_balancing"]) == selected_value]
        for split in ["all", "train", "val", "test"]:
            split_windows = phase_windows if split == "all" else [w for w in phase_windows if w.get("split") == split]
            counts = Counter(w["target_class"] for w in split_windows)
            for class_name in CLASSES:
                rows.append(
                    {
                        "phase": phase,
                        "split": split,
                        "target_class": class_name,
                        "window_count": counts.get(class_name, 0),
                    }
                )
    return rows


def count_rows_by_clip(
    windows: list[dict[str, Any]],
    classes: list[str],
    phase: str,
    selected_value: int | None = None,
) -> list[dict[str, Any]]:
    phase_windows = windows if selected_value is None else [
        window for window in windows if int(window["selected_after_balancing"]) == selected_value
    ]
    counts: dict[tuple[str, str], int] = Counter(
        (window["clip_id"], window["target_class"]) for window in phase_windows
    )
    clips = sorted({window["clip_id"] for window in windows})
    rows: list[dict[str, Any]] = []
    for clip in clips:
        for class_name in classes:
            rows.append(
                {
                    "phase": phase,
                    "clip_id": clip,
                    "target_class": class_name,
                    "window_count": counts.get((clip, class_name), 0),
                }
            )
    return rows


def class_totals(windows: list[dict[str, Any]], selected_value: int | None = None) -> dict[str, int]:
    selected_windows = windows if selected_value is None else [
        window for window in windows if int(window["selected_after_balancing"]) == selected_value
    ]
    counts = Counter(window["target_class"] for window in selected_windows)
    return {class_name: counts.get(class_name, 0) for class_name in CLASSES}


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": round(float(min(values)), 6),
        "median": round(float(np.median(values)), 6),
        "max": round(float(max(values)), 6),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    derived_root = Path(args.derived_root)
    output_dir = Path(args.output_dir)
    clips = clip_dirs(derived_root)
    frames_by_clip: dict[str, pd.DataFrame] = {}
    skipped: list[dict[str, str]] = []
    windows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    fps_by_clip: dict[str, float] = {}
    window_frames_by_clip: dict[str, int] = {}
    label_region_frames_by_clip: dict[str, int] = {}
    stride_frames_by_clip: dict[str, int] = {}
    background_center_margin_frames_by_clip: dict[str, int] = {}
    clip_frame_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for clip_dir in clips:
        clip_id = clip_dir.name
        frames_path = clip_dir / "temporal_frames.csv"
        events_path = clip_dir / "events" / "event_candidates_unified.csv"
        if not frames_path.exists() or not events_path.exists():
            skipped.append(
                {
                    "clip_id": clip_id,
                    "reason": "missing temporal_frames.csv or events/event_candidates_unified.csv",
                }
            )
            rejection_rows.append(
                {
                    "clip_id": clip_id,
                    "window_kind": "clip",
                    "candidate_start_frame": "",
                    "candidate_end_frame": "",
                    "reason": "missing_temporal_data",
                    "detail": "missing temporal_frames.csv or events/event_candidates_unified.csv",
                }
            )
            continue
        frames = read_csv(frames_path)
        if "frame" not in frames.columns or frames.empty:
            skipped.append({"clip_id": clip_id, "reason": "temporal_frames.csv missing frame column or empty"})
            rejection_rows.append(
                {
                    "clip_id": clip_id,
                    "window_kind": "clip",
                    "candidate_start_frame": "",
                    "candidate_end_frame": "",
                    "reason": "missing_temporal_data",
                    "detail": "temporal_frames.csv missing frame column or empty",
                }
            )
            continue
        fps = infer_fps(frames, args.default_fps)
        window_frames = max(1, int(round(args.window_seconds * fps)))
        label_region_frames = max(1, int(round(args.label_region_seconds * fps)))
        stride_frames = max(1, int(round(args.stride_seconds * fps)))
        background_center_margin_frames = max(1, int(round(args.background_center_margin_seconds * fps)))
        frame_values = pd.to_numeric(frames["frame"], errors="coerce").dropna().astype(int)
        min_frame = int(frame_values.min())
        max_frame = int(frame_values.max())
        frame_count = max_frame - min_frame + 1
        duration_seconds = frame_count / fps if fps > 0 else 0.0
        clip_frame_rows.append(
            {
                "clip_id": clip_id,
                "fps": round(fps, 6),
                "min_frame": min_frame,
                "max_frame": max_frame,
                "frame_count": frame_count,
                "duration_seconds": round(duration_seconds, 6),
                "requested_window_seconds": args.window_seconds,
                "window_frames": window_frames,
                "label_region_seconds": args.label_region_seconds,
                "label_region_frames": label_region_frames,
                "stride_seconds": args.stride_seconds,
                "stride_frames": stride_frames,
                "background_center_margin_seconds": args.background_center_margin_seconds,
                "background_center_margin_frames": background_center_margin_frames,
                "window_fits_without_padding": int(window_frames <= frame_count),
            }
        )
        events = load_events(clip_dir)
        clip_windows, clip_rejections = generate_windows_for_clip(
            clip_dir,
            frames,
            events,
            window_frames,
            label_region_frames,
            stride_frames,
            background_center_margin_frames,
        )
        rejection_rows.extend(clip_rejections)
        if not clip_windows:
            skipped.append({"clip_id": clip_id, "reason": "no weak event or background windows generated"})
            continue
        frames_by_clip[clip_id] = frames
        fps_by_clip[clip_id] = fps
        window_frames_by_clip[clip_id] = window_frames
        label_region_frames_by_clip[clip_id] = label_region_frames
        stride_frames_by_clip[clip_id] = stride_frames
        background_center_margin_frames_by_clip[clip_id] = background_center_margin_frames
        windows.extend(clip_windows)

    if not frames_by_clip:
        raise SystemExit("No processable clips found with temporal frames and unified event candidates.")

    usable_clip_frame_rows = [row for row in clip_frame_rows if row["clip_id"] in frames_by_clip]
    frame_counts = [int(row["frame_count"]) for row in usable_clip_frame_rows]
    durations = [float(row["duration_seconds"]) for row in usable_clip_frame_rows]
    if durations and args.window_seconds >= min(durations):
        shortest = min(durations)
        warning = (
            f"Requested window length {args.window_seconds:.3f}s is greater than or equal to "
            f"the shortest usable clip duration {shortest:.3f}s; background windows may be impossible "
            "because strict background windows must fit inside the clip and avoid event spans."
        )
        warnings.append(warning)
        print(f"Warning: {warning}")

    feature_columns, excluded_columns = numeric_feature_columns(frames_by_clip)
    candidate_clip_rows = count_rows_by_clip(windows, EVENT_CLASSES, "before_balancing")
    background_before_rows = count_rows_by_clip(windows, ["background"], "before_balancing")
    class_by_clip_rows = clip_class_report(sorted(frames_by_clip), windows)
    write_csv(
        output_dir / "weak_event_candidate_counts_by_clip.csv",
        candidate_clip_rows,
        ["phase", "clip_id", "target_class", "window_count"],
    )
    write_csv(
        output_dir / "weak_event_background_counts_by_clip_before_balancing.csv",
        background_before_rows,
        ["phase", "clip_id", "target_class", "window_count"],
    )
    write_csv(
        output_dir / "weak_event_window_rejections.csv",
        rejection_rows,
        ["clip_id", "window_kind", "candidate_start_frame", "candidate_end_frame", "reason", "detail"],
    )
    write_csv(
        output_dir / "weak_event_class_by_clip_report.csv",
        class_by_clip_rows,
        [
            "clip_id",
            "total_windows",
            "classes_present",
            "background_windows",
            "carry_windows",
            "pass_windows",
            "turnover_windows",
            "shot_windows",
        ],
    )
    write_csv(
        output_dir / "weak_event_clip_frame_diagnostics.csv",
        clip_frame_rows,
        [
            "clip_id",
            "fps",
            "min_frame",
            "max_frame",
            "frame_count",
            "duration_seconds",
            "requested_window_seconds",
            "window_frames",
            "label_region_seconds",
            "label_region_frames",
            "stride_seconds",
            "stride_frames",
            "background_center_margin_seconds",
            "background_center_margin_frames",
            "window_fits_without_padding",
        ],
    )
    assignments, split_warnings = split_clips(
        sorted(frames_by_clip),
        windows,
        args.seed,
        args.train_clips,
        args.val_clips,
        args.test_clips,
    )
    windows = balance_windows_by_split(windows, args.balance_strategy, args.seed)

    window_fields = [
        "clip_id",
        "split",
        "input_start_frame",
        "input_end_frame",
        "start_frame",
        "end_frame",
        "valid_start_frame",
        "valid_end_frame",
        "center_frame",
        "label_start_frame",
        "label_end_frame",
        "label_center_frame",
        "window_frames",
        "label_region_frames",
        "observed_frames",
        "padding_frames",
        "padding_before_frames",
        "padding_after_frames",
        "padding_percent",
        "context_contains_other_event",
        "target_class",
        "label_source",
        "candidate_id",
        "source_event_type",
        "is_eligible_for_weak_label",
        "overlap_resolution",
        "selected_after_balancing",
    ]
    write_csv(output_dir / "weak_event_windows.csv", windows, window_fields)
    count_rows = class_count_rows(windows)
    write_csv(output_dir / "weak_event_class_counts.csv", count_rows, ["phase", "split", "target_class", "window_count"])
    background_after_rows = count_rows_by_clip(windows, ["background"], "after_balancing", selected_value=1)
    write_csv(
        output_dir / "weak_event_background_counts_by_clip_after_balancing.csv",
        background_after_rows,
        ["phase", "clip_id", "target_class", "window_count"],
    )
    split_rows = [
        {
            "clip_id": clip,
            "split": split,
            "fps": round(fps_by_clip.get(clip, args.default_fps), 6),
            "selected_window_count": sum(
                1 for w in windows if w["clip_id"] == clip and int(w["selected_after_balancing"]) == 1
            ),
        }
        for clip, split in sorted(assignments.items())
    ]
    write_csv(output_dir / "clip_split_manifest.csv", split_rows, ["clip_id", "split", "fps", "selected_window_count"])

    schema = {
        "classes": CLASSES,
        "selected_feature_columns": feature_columns,
        "excluded_leakage_or_identifier_columns": excluded_columns,
        "missing_value_handling": "Per-feature train-set means are used by the trainer; residual missing values are filled with 0 after normalization.",
        "normalization_method": "Train split z-score normalization saved with the model artifacts.",
        "label_source": "Weak labels from events/event_candidates_unified.csv only; duplicate unified events are skipped; manual review labels and overrides are not used.",
        "overlap_precedence": "For multiple weak events with the same center frame: shot > turnover > pass > carry. Overlapping windows with different centers are retained as separate weak examples.",
        "event_window_rule": "Event windows are centered on weak event candidate midpoints. input_start_frame/input_end_frame and start_frame/end_frame define the fixed-length sequence and may extend beyond clip bounds for explicit boundary padding; valid_start_frame/valid_end_frame record the clipped in-clip frame span. The target label is determined by the centered label region.",
        "background_rule": "Background windows use stride-spaced fixed-length input windows without padding. They are rejected only when the centered label region overlaps a weak event span or its background_center_margin_seconds margin; event context outside the label region is allowed and recorded.",
    }
    write_json(output_dir / "weak_event_feature_schema.json", schema)

    selected_windows = [w for w in windows if int(w["selected_after_balancing"]) == 1]
    padded_event_windows = [
        window
        for window in windows
        if window["target_class"] in EVENT_CLASSES and int(window.get("padding_frames", 0)) > 0
    ]
    background_clips_before = {
        row["clip_id"]
        for row in background_before_rows
        if row["target_class"] == "background" and int(row["window_count"]) > 0
    }
    accepted_background_windows = [window for window in windows if window["target_class"] == "background"]
    accepted_background_with_context = [
        window
        for window in accepted_background_windows
        if int(window.get("context_contains_other_event", 0)) == 1
    ]
    rejection_counts_by_reason = Counter(
        row["reason"]
        for row in rejection_rows
        if row["reason"] != "accepted_with_noncentral_context_event"
    )
    all_warnings = warnings + split_warnings
    summary = {
        "created_at": utc_now(),
        "derived_root": str(derived_root),
        "total_clips_discovered": len(clips),
        "processable_clips_used": len(frames_by_clip),
        "incomplete_clips_skipped": len(skipped),
        "skipped_clips": skipped,
        "total_windows_before_balancing": len(windows),
        "selected_windows_after_balancing": len(selected_windows),
        "feature_count": len(feature_columns),
        "input_window_seconds": args.window_seconds,
        "requested_window_seconds": args.window_seconds,
        "label_region_seconds": args.label_region_seconds,
        "requested_stride_seconds": args.stride_seconds,
        "requested_background_min_distance_seconds": args.background_min_distance_seconds,
        "background_center_margin_seconds": args.background_center_margin_seconds,
        "resolved_fps_by_clip": {clip: round(fps, 6) for clip, fps in sorted(fps_by_clip.items())},
        "resolved_fps_summary": numeric_summary(list(fps_by_clip.values())),
        "clip_frame_count_summary": numeric_summary([float(value) for value in frame_counts]),
        "clip_duration_seconds_summary": numeric_summary(durations),
        "window_frames_by_clip": dict(sorted(window_frames_by_clip.items())),
        "label_region_frames_by_clip": dict(sorted(label_region_frames_by_clip.items())),
        "stride_frames_by_clip": dict(sorted(stride_frames_by_clip.items())),
        "background_center_margin_frames_by_clip": dict(sorted(background_center_margin_frames_by_clip.items())),
        "padded_event_windows": len(padded_event_windows),
        "padded_event_window_padding_frames_total": sum(int(window["padding_frames"]) for window in padded_event_windows),
        "padded_event_window_padding_percent_max": (
            max(float(window["padding_percent"]) for window in padded_event_windows)
            if padded_event_windows
            else 0.0
        ),
        "class_counts_before_balancing": class_totals(windows),
        "class_counts_after_balancing": class_totals(windows, selected_value=1),
        "accepted_background_count": len(accepted_background_windows),
        "accepted_background_with_noncentral_event_context_count": len(accepted_background_with_context),
        "clips_with_background_window_count": len(background_clips_before),
        "clips_with_background_window": sorted(background_clips_before),
        "balance_strategy": args.balance_strategy,
        "valid_5_1_2_split_exists": True,
        "clip_split_assignment": assignments,
        "class_counts_by_split": count_rows,
        "candidate_counts_by_clip": candidate_clip_rows,
        "background_counts_by_clip_before_balancing": background_before_rows,
        "background_counts_by_clip_after_balancing": background_after_rows,
        "rejection_counts_by_reason": rejection_counts_by_reason,
        "window_rejection_reason_counts": rejection_counts_by_reason,
        "accepted_background_context_event_diagnostic_count": Counter(row["reason"] for row in rejection_rows).get(
            "accepted_with_noncentral_context_event",
            0,
        ),
        "warnings": all_warnings,
        "split_warnings": split_warnings,
    }
    write_json(output_dir / "weak_event_dataset_summary.json", summary)
    print(f"Discovered clips: {len(clips)}")
    print(f"Processable clips used: {len(frames_by_clip)}")
    print(f"Incomplete clips skipped: {len(skipped)}")
    print(f"Selected windows: {len(selected_windows)}")
    print(f"Feature count: {len(feature_columns)}")
    for warning in split_warnings:
        print(f"Warning: {warning}")
    print(f"Wrote weak GRU dataset artifacts under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
