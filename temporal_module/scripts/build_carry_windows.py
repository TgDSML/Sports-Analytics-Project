"""Build fixed-length carry/background prototype windows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import PROJECT_ROOT, ensure_output_parent, reject_outputs_path, utc_now_iso  # noqa: E402


EXCLUDED_FEATURE_COLUMNS = {
    "clip_id",
    "frame",
    "timestamp",
    "possessor_track_id",
    "possession_team",
    "possession_reason",
    "p1_team",
    "p2_team",
    "p3_team",
    "p4_team",
}

INDEX_COLUMNS = [
    "clip_id",
    "start_frame",
    "end_frame",
    "center_frame",
    "label",
    "label_id",
    "window_length",
    "contains_frame_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-length temporal windows for carry/background labels."
    )
    parser.add_argument("--derived-root", default=str(Path("temporal_module") / "data" / "derived"))
    parser.add_argument("--dataset-root", default=str(Path("temporal_module") / "data" / "datasets"))
    parser.add_argument("--window-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    return parser.parse_args()


def enforce_dataset_output_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = (PROJECT_ROOT / "temporal_module" / "data" / "datasets").resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"Dataset outputs must be under {allowed}: {resolved}") from error
    reject_outputs_path(resolved)
    return resolved


def eligible_clips(derived_root: Path) -> list[tuple[str, Path, Path]]:
    if not derived_root.exists():
        raise FileNotFoundError(f"Derived root not found: {derived_root}")
    clips = []
    for clip_dir in sorted(path for path in derived_root.iterdir() if path.is_dir()):
        frames_path = clip_dir / "temporal_frames.csv"
        labels_path = clip_dir / "temporal_labels.csv"
        if frames_path.exists() and labels_path.exists():
            clips.append((clip_dir.name, frames_path, labels_path))
    return clips


def read_clip_tables(clip_id: str, frames_path: Path, labels_path: Path) -> pd.DataFrame:
    frames = pd.read_csv(frames_path)
    labels = pd.read_csv(labels_path)
    _require_columns(frames, frames_path, {"frame"})
    _require_columns(labels, labels_path, {"frame", "label", "label_id"})
    if "timestamp" not in frames.columns and "timestamp" not in labels.columns:
        raise ValueError(f"{clip_id} has no timestamp column in frames or labels")

    frames = frames.copy()
    labels = labels[["frame", "label", "label_id"] + (["timestamp"] if "timestamp" in labels.columns else [])].copy()
    frames["frame"] = pd.to_numeric(frames["frame"], errors="coerce")
    labels["frame"] = pd.to_numeric(labels["frame"], errors="coerce")
    frames = frames.dropna(subset=["frame"]).copy()
    labels = labels.dropna(subset=["frame"]).copy()
    frames["frame"] = frames["frame"].astype(int)
    labels["frame"] = labels["frame"].astype(int)

    table = frames.merge(labels, on="frame", how="inner", suffixes=("", "_label"))
    if "timestamp" not in table.columns and "timestamp_label" in table.columns:
        table = table.rename(columns={"timestamp_label": "timestamp"})
    table["clip_id"] = clip_id
    table = table.sort_values("frame").reset_index(drop=True)
    return table


def _require_columns(df: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing column(s): {', '.join(sorted(missing))}")


def discover_feature_names(clip_tables: dict[str, pd.DataFrame]) -> tuple[list[str], list[str]]:
    candidate_columns = []
    for table in clip_tables.values():
        for column in table.columns:
            if column in EXCLUDED_FEATURE_COLUMNS or column in {"label", "label_id", "timestamp_label"}:
                continue
            if column not in candidate_columns:
                candidate_columns.append(column)

    numeric_features = []
    missing_mask_features = []
    for column in candidate_columns:
        converted_parts = []
        has_non_numeric_values = False
        for table in clip_tables.values():
            if column not in table.columns:
                converted_parts.append(pd.Series([np.nan] * len(table)))
                continue
            original = table[column]
            converted = pd.to_numeric(original, errors="coerce")
            non_empty = original.notna() & (original.astype(str).str.strip() != "")
            if converted[non_empty].isna().any():
                has_non_numeric_values = True
                break
            converted_parts.append(converted)
        if has_non_numeric_values:
            continue
        combined = pd.concat(converted_parts, ignore_index=True)
        if combined.notna().any():
            numeric_features.append(column)
            if combined.isna().any():
                missing_mask_features.append(f"{column}_missing_mask")
    return numeric_features, missing_mask_features


def add_missing_masks(table: pd.DataFrame, feature_names: list[str], mask_sources: list[str]) -> pd.DataFrame:
    result = table.copy()
    mask_feature_names = {name.removesuffix("_missing_mask") for name in mask_sources}
    for feature in feature_names:
        if feature not in result.columns:
            result[feature] = np.nan
        result[feature] = pd.to_numeric(result[feature], errors="coerce")
        if feature in mask_feature_names:
            result[f"{feature}_missing_mask"] = result[feature].isna().astype(float)
        result[feature] = result[feature].fillna(0.0)
    return result


def build_windows(
    clip_tables: dict[str, pd.DataFrame],
    feature_names: list[str],
    missing_mask_feature_names: list[str],
    window_length: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, int], int]:
    arrays = []
    labels = []
    clip_ids = []
    center_frames = []
    index_rows = []
    per_clip_counts = {}
    rejected_gap_windows = 0
    ordered_features = feature_names + missing_mask_feature_names
    center_offset = window_length // 2

    for clip_id, table in clip_tables.items():
        prepared = add_missing_masks(table, feature_names, missing_mask_feature_names)
        per_clip_counts[clip_id] = 0
        for start in range(0, max(len(prepared) - window_length + 1, 0), stride):
            end = start + window_length
            window = prepared.iloc[start:end].copy()
            frame_values = window["frame"].to_numpy(dtype=int)
            contains_gap = bool(len(frame_values) > 1 and np.any(np.diff(frame_values) != 1))
            if contains_gap:
                rejected_gap_windows += 1
                continue

            center_row = window.iloc[center_offset]
            center_label = str(center_row["label"])
            center_label_id = int(center_row["label_id"])
            if center_label == "carry":
                center_label_id = 1
            elif center_label == "background":
                center_label_id = 0
            else:
                raise ValueError(f"Unsupported center label for {clip_id} frame {center_row['frame']}: {center_label}")

            feature_window = window[ordered_features].to_numpy(dtype=np.float32)
            arrays.append(feature_window)
            labels.append(center_label_id)
            clip_ids.append(clip_id)
            center_frames.append(int(center_row["frame"]))
            per_clip_counts[clip_id] += 1
            index_rows.append(
                {
                    "clip_id": clip_id,
                    "start_frame": int(frame_values[0]),
                    "end_frame": int(frame_values[-1]),
                    "center_frame": int(center_row["frame"]),
                    "label": center_label,
                    "label_id": center_label_id,
                    "window_length": window_length,
                    "contains_frame_gap": int(contains_gap),
                }
            )

    if arrays:
        x = np.stack(arrays).astype(np.float32)
    else:
        x = np.empty((0, window_length, len(ordered_features)), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return (
        x,
        y,
        np.asarray(clip_ids, dtype=object),
        np.asarray(center_frames, dtype=np.int64),
        index_rows,
        per_clip_counts,
        rejected_gap_windows,
    )


def write_index_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    output = ensure_output_parent(path)
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    try:
        if args.window_length <= 0:
            raise ValueError("--window-length must be positive")
        if args.stride <= 0:
            raise ValueError("--stride must be positive")

        derived_root = Path(args.derived_root)
        dataset_root = enforce_dataset_output_root(Path(args.dataset_root))
        clips = eligible_clips(derived_root)
        clip_tables = {
            clip_id: read_clip_tables(clip_id, frames_path, labels_path)
            for clip_id, frames_path, labels_path in clips
        }
        feature_names, missing_mask_feature_names = discover_feature_names(clip_tables)
        x, y, clip_ids, center_frames, index_rows, per_clip_counts, rejected_gap_windows = build_windows(
            clip_tables=clip_tables,
            feature_names=feature_names,
            missing_mask_feature_names=missing_mask_feature_names,
            window_length=args.window_length,
            stride=args.stride,
        )

        dataset_root.mkdir(parents=True, exist_ok=True)
        npz_path = dataset_root / "carry_background_windows.npz"
        metadata_path = dataset_root / "carry_background_windows_metadata.json"
        index_path = dataset_root / "carry_background_windows_index.csv"
        reject_outputs_path(npz_path)
        reject_outputs_path(metadata_path)
        reject_outputs_path(index_path)

        np.savez_compressed(
            npz_path,
            X=x,
            y=y,
            clip_ids=clip_ids,
            center_frames=center_frames,
        )
        write_index_csv(index_path, index_rows)

        carry_count = int((y == 1).sum())
        background_count = int((y == 0).sum())
        metadata = {
            "build_timestamp": utc_now_iso(),
            "window_length": int(args.window_length),
            "stride": int(args.stride),
            "number_of_windows": int(len(y)),
            "number_of_features": int(x.shape[2]),
            "feature_names": feature_names + missing_mask_feature_names,
            "missing_mask_feature_names": missing_mask_feature_names,
            "carry_window_count": carry_count,
            "background_window_count": background_count,
            "rejected_gap_window_count": int(rejected_gap_windows),
            "per_clip_window_counts": per_clip_counts,
            "input_file_paths": {
                clip_id: {
                    "temporal_frames": str(frames_path),
                    "temporal_labels": str(labels_path),
                }
                for clip_id, frames_path, labels_path in clips
            },
            "dataset_creation_assumptions": [
                "Only clips with both temporal_frames.csv and temporal_labels.csv are eligible.",
                "Windows are built within a single clip and never cross clip boundaries.",
                "Candidate windows with any adjacent frame difference not equal to 1 are rejected.",
                "The window label is the center row label.",
                "Only numeric temporal frame columns are model inputs.",
                "Excluded identifier/categorical columns are not model inputs.",
                "Original numeric NaN values are filled with 0.0 in X.",
                "A missing-mask feature is added for each numeric feature with any missing value across the dataset.",
            ],
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

        print("Discovered eligible clips:")
        for clip_id, _, _ in clips:
            print(f"- {clip_id}")
        print("Windows per clip:")
        for clip_id, count in per_clip_counts.items():
            print(f"- {clip_id}: {count}")
        print(f"Total windows: {len(y)}")
        print(f"Carry windows: {carry_count}")
        print(f"Background windows: {background_count}")
        print(f"Rejected gap-window count: {rejected_gap_windows}")
        print(f"X shape: {x.shape}")
        print(f"y shape: {y.shape}")
        print(f"NPZ output: {npz_path}")
        print(f"Metadata output: {metadata_path}")
        print(f"Index CSV output: {index_path}")
        return 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

