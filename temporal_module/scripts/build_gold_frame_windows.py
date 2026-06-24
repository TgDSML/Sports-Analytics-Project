"""Build gold-labelled visual frame windows for CNN-LSTM experiments."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


DEFAULT_CLASSES = ["background", "carry", "pass", "turnover", "shot"]


def main() -> int:
    args = parse_args()
    return build_frame_windows(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gold visual frame-window tensors")
    parser.add_argument("--gold-events", type=Path, default=Path("temporal_module/data/gold_event_project/annotations/gold_event_intervals.csv"))
    parser.add_argument("--video-root", type=Path, default=Path("temporal_module/data/gold_event_project/cvat_uploads"))
    parser.add_argument("--output-dir", type=Path, default=Path("temporal_module/data_gold/frame_windows"))
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--label-region-frames", type=int, default=16)
    parser.add_argument("--sample-frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=112)
    parser.add_argument("--width", type=int, default=112)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    return parser.parse_args()


def build_frame_windows(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.gold_events.exists():
        write_not_ready(args.output_dir, args.gold_events, "Gold event interval file does not exist.")
        print(f"Gold events not found: {args.gold_events}")
        return 0
    events = pd.read_csv(args.gold_events)
    required = {"clip_id", "event_type", "start_frame", "end_frame"}
    missing = sorted(required - set(events.columns))
    if missing:
        write_not_ready(args.output_dir, args.gold_events, f"Gold events missing columns: {', '.join(missing)}")
        print(f"Gold events missing columns: {missing}")
        return 0

    events = normalize_events(events, set(args.classes))
    if events.empty:
        write_not_ready(args.output_dir, args.gold_events, "No usable events after filtering classes.")
        print("No usable events.")
        return 0

    label_to_id = {label: idx for idx, label in enumerate(args.classes)}
    windows: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict] = []
    skipped: list[dict] = []

    clip_ids = sorted(events["clip_id"].unique())
    if args.max_clips is not None:
        clip_ids = clip_ids[: max(0, args.max_clips)]
    start_time = time.perf_counter()
    for clip_index, clip_id in enumerate(clip_ids, start=1):
        elapsed = time.perf_counter() - start_time
        print(
            f"Processing clip {clip_index}/{len(clip_ids)}: {clip_id} "
            f"(windows so far: {len(windows)}, elapsed: {elapsed:.1f}s)",
            flush=True,
        )
        video_path = args.video_root / f"{clip_id}.mp4"
        if not video_path.exists():
            skipped.append({"clip_id": clip_id, "reason": "missing_video", "path": str(video_path)})
            continue
        clip_events = events[events["clip_id"] == clip_id].copy()
        try:
            clip_windows = build_clip_windows(
                clip_id=clip_id,
                video_path=video_path,
                events=clip_events,
                label_to_id=label_to_id,
                classes=args.classes,
                window_size=args.window_size,
                stride=args.stride,
                label_region_frames=args.label_region_frames,
                sample_frames=args.sample_frames,
                height=args.height,
                width=args.width,
            )
        except RuntimeError as error:
            skipped.append({"clip_id": clip_id, "reason": str(error), "path": str(video_path)})
            continue
        for x, y, row in clip_windows:
            row["window_id"] = len(metadata)
            windows.append(x)
            labels.append(y)
            metadata.append(row)
        elapsed = time.perf_counter() - start_time
        average = elapsed / clip_index
        eta = average * (len(clip_ids) - clip_index)
        print(
            f"Completed clip {clip_index}/{len(clip_ids)}: generated {len(clip_windows)} windows "
            f"(total: {len(windows)}, elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s)",
            flush=True,
        )

    if not windows:
        write_not_ready(args.output_dir, args.gold_events, "No frame windows generated.")
        write_csv(args.output_dir / "gold_frame_windows_skipped.csv", skipped, ["clip_id", "reason", "path"])
        print("No frame windows generated.")
        return 0

    x_array = np.stack(windows).astype(np.float32)
    y_array = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(
        args.output_dir / "gold_frame_windows.npz",
        X=x_array,
        y=y_array,
        label_names=np.asarray(args.classes),
    )
    pd.DataFrame(metadata).to_csv(args.output_dir / "gold_frame_windows_metadata.csv", index=False)
    write_csv(args.output_dir / "gold_frame_windows_skipped.csv", skipped, ["clip_id", "reason", "path"])
    write_summary(args.output_dir, args, x_array, y_array, metadata, skipped)
    print(f"Wrote gold frame windows to {args.output_dir}")
    print(f"Shape: {tuple(x_array.shape)}")
    return 0


def normalize_events(events: pd.DataFrame, classes: set[str]) -> pd.DataFrame:
    out = events.copy()
    out["event_type"] = out["event_type"].astype(str).str.strip().str.lower()
    out = out[out["event_type"].isin(classes)].copy()
    out["start_frame"] = pd.to_numeric(out["start_frame"], errors="coerce")
    out["end_frame"] = pd.to_numeric(out["end_frame"], errors="coerce")
    out = out.dropna(subset=["clip_id", "event_type", "start_frame", "end_frame"]).copy()
    out["start_frame"] = out["start_frame"].astype(int)
    out["end_frame"] = out["end_frame"].astype(int)
    swap = out["end_frame"] < out["start_frame"]
    out.loc[swap, ["start_frame", "end_frame"]] = out.loc[swap, ["end_frame", "start_frame"]].to_numpy()
    return out


def build_clip_windows(
    clip_id: str,
    video_path: Path,
    events: pd.DataFrame,
    label_to_id: dict[str, int],
    classes: list[str],
    window_size: int,
    stride: int,
    label_region_frames: int,
    sample_frames: int,
    height: int,
    width: int,
) -> list[tuple[np.ndarray, int, dict]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("could_not_open_video")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        capture.release()
        raise RuntimeError("no_video_frames")

    frames = read_all_frames(capture, height, width)
    capture.release()
    rows: list[tuple[np.ndarray, int, dict]] = []
    event_records = events.to_dict("records")
    for start in range(0, max(frame_count - window_size + 1, 1), max(1, stride)):
        end = min(start + window_size - 1, frame_count - 1)
        if end - start + 1 < window_size:
            continue
        label_start, label_end = centered_region(start, end, label_region_frames)
        label, event_id, overlap = choose_label(event_records, label_start, label_end, classes)
        frame_indices = sample_indices(start, end, sample_frames)
        tensor = frames[frame_indices].copy()
        rows.append(
            (
                tensor,
                label_to_id[label],
                {
                    "clip_id": clip_id,
                    "start_frame": start,
                    "end_frame": end,
                    "label_start_frame": label_start,
                    "label_end_frame": label_end,
                    "target_class": label,
                    "event_id": event_id,
                    "event_overlap_frames": overlap,
                    "fps": f"{fps:.6f}" if fps > 0 else "",
                    "source_video": str(video_path),
                },
            )
        )
    return rows


def sample_indices(start: int, end: int, count: int) -> list[int]:
    if count <= 1:
        return [(start + end) // 2]
    return [int(round(value)) for value in np.linspace(start, end, count)]


def read_all_frames(capture: cv2.VideoCapture, height: int, width: int) -> np.ndarray:
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(frame)
    if not frames:
        raise RuntimeError("no_decoded_frames")
    return np.stack(frames).astype(np.float32)


def centered_region(start: int, end: int, label_region_frames: int) -> tuple[int, int]:
    center = (start + end) // 2
    half = max(1, label_region_frames) // 2
    label_start = center - half
    label_end = label_start + max(1, label_region_frames) - 1
    return max(start, label_start), min(end, label_end)


def choose_label(events: list[dict], start: int, end: int, classes: list[str]) -> tuple[str, str, int]:
    best_label = "background"
    best_event_id = ""
    best_overlap = 0
    priority = {name: idx for idx, name in enumerate(classes)}
    for row in events:
        overlap = max(0, min(end, int(row["end_frame"])) - max(start, int(row["start_frame"])) + 1)
        if overlap <= 0:
            continue
        label = str(row["event_type"])
        if overlap > best_overlap or (overlap == best_overlap and priority.get(label, 0) > priority.get(best_label, 0)):
            best_label = label
            best_event_id = str(row.get("event_id", ""))
            best_overlap = overlap
    return best_label, best_event_id, best_overlap


def write_summary(output_dir: Path, args: argparse.Namespace, x_array: np.ndarray, y_array: np.ndarray, metadata: list[dict], skipped: list[dict]) -> None:
    counts = Counter(int(item) for item in y_array.tolist())
    distribution = [{"class": label, "windows": counts.get(idx, 0)} for idx, label in enumerate(args.classes)]
    write_csv(output_dir / "gold_frame_label_distribution.csv", distribution, ["class", "windows"])
    summary = {
        "status": "ready",
        "shape": list(x_array.shape),
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "sample_frames": int(args.sample_frames),
        "height": int(args.height),
        "width": int(args.width),
        "classes": args.classes,
        "label_distribution": {row["class"]: row["windows"] for row in distribution},
        "clips_used": len({row["clip_id"] for row in metadata}),
        "clips_skipped": len(skipped),
    }
    (output_dir / "gold_frame_windows_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Gold Frame Window Dataset Summary",
        "",
        "This dataset uses manually verified CVAT event intervals and sampled RGB frame windows.",
        "",
        f"- Windows: {x_array.shape[0]}",
        f"- Shape: {tuple(x_array.shape)}",
        f"- Frame samples per window: {args.sample_frames}",
        f"- Frame size: {args.width}x{args.height}",
        f"- Label region frames: {args.label_region_frames}",
        f"- Clips used: {summary['clips_used']}",
        f"- Clips skipped: {summary['clips_skipped']}",
        "",
        "## Label Distribution",
        "",
    ]
    lines.extend(f"- {row['class']}: {row['windows']}" for row in distribution)
    (output_dir / "gold_frame_windows_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_not_ready(output_dir: Path, path: Path, reason: str) -> None:
    lines = [
        "# Gold Frame Window Dataset Summary",
        "",
        "Status: NOT READY",
        "",
        f"Input path: `{path}`",
        f"Reason: {reason}",
    ]
    (output_dir / "gold_frame_windows_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
