"""Build gold-labelled temporal event windows from CVAT-imported intervals."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CLASSES = ["background", "carry", "pass", "turnover", "shot"]
EXCLUDED_COLUMNS = {
    "clip_id",
    "frame",
    "timestamp",
    "event_id",
    "event_type",
    "target_class",
    "label",
    "split",
    "source",
}


def main() -> int:
    args = parse_args()
    return build_gold_windows(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gold supervised event windows")
    parser.add_argument("--gold-events", type=Path, default=Path("temporal_module/data/gold_event_project/annotations/gold_event_intervals.csv"))
    parser.add_argument("--derived-root", type=Path, default=Path("temporal_module/data/derived"))
    parser.add_argument("--output-dir", type=Path, default=Path("temporal_module/data_gold/event_windows"))
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--label-region-frames", type=int, default=16)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    return parser.parse_args()


def build_gold_windows(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.gold_events.exists():
        write_not_ready(args.output_dir, args.gold_events, "Gold event interval file does not exist yet.")
        print(f"Gold events not found: {args.gold_events}")
        return 0

    events = pd.read_csv(args.gold_events)
    required = {"clip_id", "event_type", "start_frame", "end_frame"}
    missing = sorted(required - set(events.columns))
    if missing:
        write_not_ready(args.output_dir, args.gold_events, f"Gold events missing required columns: {', '.join(missing)}")
        print(f"Gold events missing columns: {missing}")
        return 0

    events = normalize_events(events, set(args.classes))
    if events.empty:
        write_not_ready(args.output_dir, args.gold_events, "No usable gold event intervals after filtering classes.")
        print("No usable gold intervals.")
        return 0

    frame_tables = load_frame_tables(args.derived_root, sorted(events["clip_id"].unique()))
    if not frame_tables:
        write_not_ready(args.output_dir, args.gold_events, "No matching temporal_frames.csv files were found.")
        print("No matching temporal frame tables.")
        return 0

    feature_names = discover_feature_names(frame_tables)
    if not feature_names:
        write_not_ready(args.output_dir, args.gold_events, "No numeric feature columns were found.")
        print("No numeric feature columns found.")
        return 0

    windows, labels, metadata = [], [], []
    label_to_id = {label: idx for idx, label in enumerate(args.classes)}
    for clip_id, table in frame_tables.items():
        clip_events = events[events["clip_id"] == clip_id].copy()
        clip_windows = build_clip_windows(
            clip_id=clip_id,
            table=table,
            events=clip_events,
            feature_names=feature_names,
            classes=args.classes,
            label_to_id=label_to_id,
            window_size=args.window_size,
            stride=args.stride,
            label_region_frames=args.label_region_frames,
        )
        for x, y, row in clip_windows:
            windows.append(x)
            labels.append(y)
            row["window_id"] = len(metadata)
            metadata.append(row)

    if not windows:
        write_not_ready(args.output_dir, args.gold_events, "No windows were generated from the available gold events.")
        print("No windows generated.")
        return 0

    x_array = np.stack(windows).astype(np.float32)
    y_array = np.asarray(labels, dtype=np.int64)
    np.savez_compressed(
        args.output_dir / "gold_event_windows.npz",
        X=x_array,
        y=y_array,
        feature_names=np.asarray(feature_names),
        label_names=np.asarray(args.classes),
    )
    pd.DataFrame(metadata).to_csv(args.output_dir / "gold_event_windows_metadata.csv", index=False)
    write_summary(args.output_dir, args, x_array, y_array, metadata, feature_names)
    print(f"Wrote gold event windows to {args.output_dir}")
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


def load_frame_tables(derived_root: Path, clip_ids: list[str]) -> dict[str, pd.DataFrame]:
    tables = {}
    for clip_id in clip_ids:
        path = derived_root / clip_id / "temporal_frames.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        if "frame" not in table.columns:
            continue
        table["frame"] = pd.to_numeric(table["frame"], errors="coerce")
        table = table.dropna(subset=["frame"]).copy()
        table["frame"] = table["frame"].astype(int)
        tables[clip_id] = table.sort_values("frame")
    return tables


def discover_feature_names(frame_tables: dict[str, pd.DataFrame]) -> list[str]:
    names = []
    for table in frame_tables.values():
        for column in table.columns:
            if column in EXCLUDED_COLUMNS or column.endswith("_id"):
                continue
            if pd.api.types.is_numeric_dtype(table[column]) and column not in names:
                names.append(column)
    return names


def build_clip_windows(
    clip_id: str,
    table: pd.DataFrame,
    events: pd.DataFrame,
    feature_names: list[str],
    classes: list[str],
    label_to_id: dict[str, int],
    window_size: int,
    stride: int,
    label_region_frames: int,
) -> list[tuple[np.ndarray, int, dict]]:
    if table.empty:
        return []
    prepared = table[["frame", *feature_names]].copy()
    for feature in feature_names:
        prepared[feature] = pd.to_numeric(prepared[feature], errors="coerce")
    prepared = prepared.set_index("frame")
    min_frame = int(prepared.index.min())
    max_frame = int(prepared.index.max())
    rows = []
    for start in range(min_frame, max_frame - window_size + 2, max(1, stride)):
        end = start + window_size - 1
        label_start, label_end = centered_region(start, end, label_region_frames)
        label, event_id, overlap = choose_label(events, label_start, label_end, classes)
        window = prepared.reindex(range(start, end + 1)).ffill().bfill().fillna(0.0)
        rows.append(
            (
                window[feature_names].to_numpy(dtype=np.float32),
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
                },
            )
        )
    return rows


def centered_region(start: int, end: int, label_region_frames: int) -> tuple[int, int]:
    center = (start + end) // 2
    half = max(1, label_region_frames) // 2
    label_start = center - half
    label_end = label_start + max(1, label_region_frames) - 1
    return max(start, label_start), min(end, label_end)


def choose_label(events: pd.DataFrame, start: int, end: int, classes: list[str]) -> tuple[str, str, int]:
    best_label = "background"
    best_event_id = ""
    best_overlap = 0
    priority = {name: idx for idx, name in enumerate(classes)}
    for row in events.to_dict("records"):
        overlap = max(0, min(end, int(row["end_frame"])) - max(start, int(row["start_frame"])) + 1)
        if overlap <= 0:
            continue
        label = str(row["event_type"])
        if overlap > best_overlap or (overlap == best_overlap and priority.get(label, 0) > priority.get(best_label, 0)):
            best_label = label
            best_event_id = str(row.get("event_id", ""))
            best_overlap = overlap
    return best_label, best_event_id, best_overlap


def write_not_ready(output_dir: Path, gold_events: Path, reason: str) -> None:
    report = [
        "# Gold Event Window Builder Status",
        "",
        "Status: NOT READY",
        "",
        f"Gold events path: `{gold_events}`",
        f"Reason: {reason}",
        "",
        "Run the CVAT import step first, then rebuild this dataset.",
    ]
    (output_dir / "gold_event_windows_summary.md").write_text("\n".join(report), encoding="utf-8")


def write_summary(output_dir: Path, args: argparse.Namespace, x_array: np.ndarray, y_array: np.ndarray, metadata: list[dict], feature_names: list[str]) -> None:
    counts = Counter(int(item) for item in y_array.tolist())
    rows = []
    for idx, label in enumerate(args.classes):
        rows.append({"class": label, "windows": counts.get(idx, 0)})
    write_csv(output_dir / "gold_event_label_distribution.csv", rows, ["class", "windows"])
    summary = {
        "status": "ready",
        "shape": list(x_array.shape),
        "window_size": int(args.window_size),
        "stride": int(args.stride),
        "label_region_frames": int(args.label_region_frames),
        "classes": args.classes,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "label_distribution": {args.classes[idx]: counts.get(idx, 0) for idx in range(len(args.classes))},
        "window_count": len(metadata),
    }
    (output_dir / "gold_event_windows_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Gold Event Window Dataset Summary",
        "",
        "This dataset uses manually verified CVAT event intervals when available.",
        "",
        f"- Windows: {x_array.shape[0]}",
        f"- Window size: {x_array.shape[1]}",
        f"- Feature count: {x_array.shape[2]}",
        f"- Label region frames: {args.label_region_frames}",
        "",
        "## Label Distribution",
        "",
    ]
    lines.extend(f"- {row['class']}: {row['windows']}" for row in rows)
    (output_dir / "gold_event_windows_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
