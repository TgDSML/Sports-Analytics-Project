"""Build weak frame-level carry/background labels from carry event CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import ensure_output_parent, read_csv_with_header, reject_outputs_path, utc_now_iso  # noqa: E402


LABEL_COLUMNS = [
    "clip_id",
    "frame",
    "timestamp",
    "label",
    "label_id",
    "label_source",
    "label_confidence",
    "event_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build weak carry/background frame labels for derived temporal clips."
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--derived-root", default=str(Path("temporal_module") / "data" / "derived"))
    return parser.parse_args()


def candidate_clip_ids(derived_root: Path) -> list[str]:
    if not derived_root.exists():
        raise FileNotFoundError(f"Derived root not found: {derived_root}")
    return [
        path.name
        for path in sorted(derived_root.iterdir())
        if path.is_dir() and (path / "temporal_frames.csv").exists()
    ]


def _require_columns(df: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing column(s): {', '.join(sorted(missing))}")


def build_labels_for_clip(
    clip_id: str,
    temporal_frames_path: Path,
    carries_path: Path,
    output_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    temporal_frames, temporal_header = read_csv_with_header(temporal_frames_path)
    carries, carries_header = read_csv_with_header(carries_path)
    _require_columns(temporal_frames, temporal_frames_path, {"frame", "timestamp"})
    _require_columns(carries, carries_path, {"carry_id", "start_frame", "end_frame", "carry_quality_flag"})

    labels = temporal_frames[["frame", "timestamp"]].copy()
    labels["clip_id"] = clip_id
    labels["frame"] = pd.to_numeric(labels["frame"], errors="coerce").astype("Int64")
    labels = labels.dropna(subset=["frame"]).copy()
    labels["frame"] = labels["frame"].astype(int)
    labels["label"] = "background"
    labels["label_id"] = 0
    labels["label_source"] = "default_background"
    labels["label_confidence"] = 1.0
    labels["event_id"] = pd.NA

    accepted = carries[carries["carry_quality_flag"].astype(str) == "ok"].copy()
    ignored_low_quality = int(len(carries) - len(accepted))
    frame_to_index = {int(frame): index for index, frame in labels["frame"].items()}
    duplicate_label_frame_count = 0
    zero_overlap_events: list[dict[str, Any]] = []

    for event in accepted.itertuples(index=False):
        carry_id = getattr(event, "carry_id")
        start_frame = int(float(getattr(event, "start_frame")))
        end_frame = int(float(getattr(event, "end_frame")))
        matching_frames = [
            frame
            for frame in frame_to_index
            if start_frame <= int(frame) <= end_frame
        ]
        if not matching_frames:
            zero_overlap_events.append(
                {
                    "carry_id": carry_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                }
            )
            continue
        for frame in matching_frames:
            row_index = frame_to_index[frame]
            if labels.at[row_index, "label"] == "carry":
                duplicate_label_frame_count += 1
                continue
            labels.at[row_index, "label"] = "carry"
            labels.at[row_index, "label_id"] = 1
            labels.at[row_index, "label_source"] = "carry_csv_ok"
            labels.at[row_index, "label_confidence"] = 1.0
            labels.at[row_index, "event_id"] = carry_id

    labels = labels[LABEL_COLUMNS].sort_values("frame").reset_index(drop=True)
    output = ensure_output_parent(output_path)
    labels.to_csv(output, index=False)

    carry_frames = int((labels["label"] == "carry").sum())
    background_frames = int((labels["label"] == "background").sum())
    metadata = {
        "clip_id": clip_id,
        "build_timestamp": utc_now_iso(),
        "input_paths": {
            "temporal_frames": str(temporal_frames_path),
            "carries": str(carries_path),
        },
        "detected_source_headers": {
            "temporal_frames": temporal_header,
            "carries": carries_header,
        },
        "output_path": str(output),
        "output_columns": LABEL_COLUMNS,
        "total_temporal_frames": int(len(labels)),
        "carry_labeled_frames": carry_frames,
        "background_labeled_frames": background_frames,
        "accepted_carry_event_count": int(len(accepted)),
        "ignored_low_quality_carry_event_count": ignored_low_quality,
        "duplicate_label_frame_count": int(duplicate_label_frame_count),
        "zero_overlap_accepted_events": zero_overlap_events,
        "label_rules": [
            "Only carry_quality_flag == 'ok' is accepted.",
            "Accepted carry ranges are inclusive over start_frame and end_frame.",
            "Frames outside accepted carry ranges are default background.",
            "No passes, interceptions, possession labels, duel labels, or low-quality carries are used.",
        ],
    }
    schema_path = output.with_name("temporal_labels_schema.json")
    schema_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return labels, metadata


def main() -> int:
    args = parse_args()
    try:
        outputs_root = Path(args.outputs_root)
        derived_root = Path(args.derived_root)
        reject_outputs_path(derived_root)

        summary_rows = []
        total_carry_frames = 0
        total_background_frames = 0
        total_accepted_events = 0
        total_ignored_events = 0

        for clip_id in candidate_clip_ids(derived_root):
            temporal_frames_path = derived_root / clip_id / "temporal_frames.csv"
            carries_path = outputs_root / clip_id / "carries" / "carries.csv"
            output_path = derived_root / clip_id / "temporal_labels.csv"
            row = {
                "clip_id": clip_id,
                "status": "failed",
                "total_frames": 0,
                "carry_frames": 0,
                "background_frames": 0,
                "accepted_carry_events": 0,
                "ignored_low_quality_carry_events": 0,
                "zero_overlap_accepted_events": 0,
                "output_path": "",
                "error_message": "",
            }
            try:
                if not carries_path.exists():
                    raise FileNotFoundError(f"Missing carries CSV: {carries_path}")
                labels, metadata = build_labels_for_clip(
                    clip_id=clip_id,
                    temporal_frames_path=temporal_frames_path,
                    carries_path=carries_path,
                    output_path=output_path,
                )
                carry_frames = int(metadata["carry_labeled_frames"])
                background_frames = int(metadata["background_labeled_frames"])
                accepted_events = int(metadata["accepted_carry_event_count"])
                ignored_events = int(metadata["ignored_low_quality_carry_event_count"])
                total_carry_frames += carry_frames
                total_background_frames += background_frames
                total_accepted_events += accepted_events
                total_ignored_events += ignored_events
                row.update(
                    {
                        "status": "success",
                        "total_frames": int(len(labels)),
                        "carry_frames": carry_frames,
                        "background_frames": background_frames,
                        "accepted_carry_events": accepted_events,
                        "ignored_low_quality_carry_events": ignored_events,
                        "zero_overlap_accepted_events": int(len(metadata["zero_overlap_accepted_events"])),
                        "output_path": str(output_path),
                    }
                )
            except Exception as error:
                row["error_message"] = str(error)
            summary_rows.append(row)

        summary_path = ensure_output_parent(derived_root / "build_carry_labels_summary.csv")
        with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "clip_id",
                "status",
                "total_frames",
                "carry_frames",
                "background_frames",
                "accepted_carry_events",
                "ignored_low_quality_carry_events",
                "zero_overlap_accepted_events",
                "output_path",
                "error_message",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        print("Per-clip carry label results:")
        for row in summary_rows:
            print(
                f"{row['clip_id']}: {row['status']} "
                f"frames={row['total_frames']} carry={row['carry_frames']} "
                f"background={row['background_frames']} events={row['accepted_carry_events']} "
                f"ignored={row['ignored_low_quality_carry_events']} error={row['error_message']}"
            )
        print(f"Total carry frames: {total_carry_frames}")
        print(f"Total background frames: {total_background_frames}")
        print(f"Total accepted carry events: {total_accepted_events}")
        print(f"Total ignored low-quality events: {total_ignored_events}")
        print(f"Summary CSV: {summary_path}")
        return 0 if all(row["status"] == "success" for row in summary_rows) else 1
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

