"""Batch-export weak pass events for clips with temporal frame tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_pass_events import build_pass_events  # noqa: E402
from src.io_utils import ensure_output_parent, read_csv_with_header, reject_outputs_path, utc_now_iso  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-export weak pass events for derived temporal clips."
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--derived-root", default=str(Path("temporal_module") / "data" / "derived"))
    return parser.parse_args()


def candidate_clip_ids(derived_root: Path) -> list[str]:
    if not derived_root.exists():
        raise FileNotFoundError(f"Derived root not found: {derived_root}")
    clip_ids = []
    for path in sorted(derived_root.iterdir()):
        if path.is_dir() and (path / "temporal_frames.csv").exists():
            clip_ids.append(path.name)
    return clip_ids


def write_pass_schema(
    schema_path: Path,
    clip_id: str,
    possession_path: Path,
    possession_debug_path: Path,
    possession_header: list[str],
    possession_debug_header: list[str],
    event_count: int,
    output_columns: list[str],
) -> None:
    schema = {
        "clip_id": clip_id,
        "build_timestamp": utc_now_iso(),
        "input_paths": {
            "possession": str(possession_path),
            "possession_debug": str(possession_debug_path),
        },
        "detected_source_headers": {
            "possession": possession_header,
            "possession_debug": possession_debug_header,
        },
        "event_count": int(event_count),
        "output_columns": output_columns,
        "label_source": "possession_transition_heuristic",
        "assumptions": [
            "Weak pass events are same-team stable possession transitions from player A to player B.",
            "Weak pass events are not ground truth.",
            "This batch script reuses temporal_module/scripts/export_pass_events.py build_pass_events logic.",
        ],
        "settings": {
            "min_possession_frames": 6,
            "min_possession_duration": 0.20,
            "max_transfer_gap": 2.0,
        },
    }
    schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        outputs_root = Path(args.outputs_root)
        derived_root = Path(args.derived_root)
        reject_outputs_path(derived_root)

        clip_ids = candidate_clip_ids(derived_root)
        summary_rows = []
        total_events = 0

        for clip_id in clip_ids:
            possession_path = outputs_root / clip_id / "possession" / "possession.csv"
            possession_debug_path = outputs_root / clip_id / "possession" / "possession_debug.csv"
            output_path = derived_root / clip_id / "passes_weak.csv"
            row = {
                "clip_id": clip_id,
                "status": "failed",
                "pass_event_count": 0,
                "input_possession_rows": 0,
                "input_possession_debug_rows": 0,
                "output_path": "",
                "error_message": "",
            }

            try:
                if not possession_path.exists():
                    raise FileNotFoundError(f"Missing possession CSV: {possession_path}")
                if not possession_debug_path.exists():
                    raise FileNotFoundError(f"Missing possession debug CSV: {possession_debug_path}")

                possession_df, possession_header = read_csv_with_header(possession_path)
                possession_debug_df, possession_debug_header = read_csv_with_header(possession_debug_path)
                events = build_pass_events(
                    clip_id=clip_id,
                    possession_df=possession_df,
                    min_frames=6,
                    min_duration=0.20,
                    max_transfer_gap=2.0,
                )

                output = ensure_output_parent(output_path)
                events.to_csv(output, index=False)
                schema_path = output.with_name("passes_weak_schema.json")
                write_pass_schema(
                    schema_path=schema_path,
                    clip_id=clip_id,
                    possession_path=possession_path,
                    possession_debug_path=possession_debug_path,
                    possession_header=possession_header,
                    possession_debug_header=possession_debug_header,
                    event_count=len(events),
                    output_columns=list(events.columns),
                )

                event_count = int(len(events))
                total_events += event_count
                row.update(
                    {
                        "status": "success",
                        "pass_event_count": event_count,
                        "input_possession_rows": int(len(possession_df)),
                        "input_possession_debug_rows": int(len(possession_debug_df)),
                        "output_path": str(output),
                    }
                )
            except Exception as error:
                row["error_message"] = str(error)
            summary_rows.append(row)

        summary_path = ensure_output_parent(derived_root / "build_all_passes_summary.csv")
        with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "clip_id",
                "status",
                "pass_event_count",
                "input_possession_rows",
                "input_possession_debug_rows",
                "output_path",
                "error_message",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        successful = sum(row["status"] == "success" for row in summary_rows)
        failed = sum(row["status"] == "failed" for row in summary_rows)
        print("Per-clip pass export results:")
        for row in summary_rows:
            print(
                f"{row['clip_id']}: {row['status']} "
                f"passes={row['pass_event_count']} error={row['error_message']}"
            )
        print(f"Candidate clips: {len(clip_ids)}")
        print(f"Successful clips: {successful}")
        print(f"Failed clips: {failed}")
        print(f"Total weak pass events: {total_events}")
        print(f"Summary CSV: {summary_path}")
        return 0 if failed == 0 else 1
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

