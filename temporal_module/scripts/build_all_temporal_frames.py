"""Batch builder for canonical temporal frame tables."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_builder import build_temporal_frames  # noqa: E402
from src.io_utils import ensure_output_parent, format_missing_inputs, reject_outputs_path  # noqa: E402


REQUIRED_RELATIVE_PATHS = {
    "tracks": Path("tracks") / "tracks.csv",
    "ball_tracks": Path("tracks") / "ball_tracks.csv",
    "player_teams": Path("teams") / "player_teams.csv",
    "possession": Path("possession") / "possession.csv",
    "possession_debug": Path("possession") / "possession_debug.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build temporal frame tables for every available clip.")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--derived-root", default=str(Path("temporal_module") / "data" / "derived"))
    parser.add_argument("--k-nearest", type=int, default=4)
    parser.add_argument("--defender-radius-px", type=float, default=100.0)
    return parser.parse_args()


def _candidate_clip_dirs(outputs_root: Path) -> list[Path]:
    if not outputs_root.exists():
        raise FileNotFoundError(f"Outputs root not found: {outputs_root}")
    return sorted(path for path in outputs_root.iterdir() if path.is_dir())


def main() -> int:
    args = parse_args()
    try:
        reject_outputs_path(args.derived_root)
        outputs_root = Path(args.outputs_root)
        derived_root = Path(args.derived_root)
        rows = []
        total_output_rows = 0
        candidates = _candidate_clip_dirs(outputs_root)
        for clip_dir in candidates:
            clip_id = clip_dir.name
            inputs = {name: clip_dir / relative for name, relative in REQUIRED_RELATIVE_PATHS.items()}
            missing = format_missing_inputs(inputs)
            if missing:
                rows.append(
                    {
                        "clip_id": clip_id,
                        "status": "skipped",
                        "missing_inputs": ";".join(missing),
                        "input_row_counts": "",
                        "output_path": "",
                        "output_rows": 0,
                        "error_message": "",
                    }
                )
                continue
            output_path = derived_root / clip_id / "temporal_frames.csv"
            try:
                result = build_temporal_frames(
                    clip_id=clip_id,
                    tracks_path=inputs["tracks"],
                    ball_tracks_path=inputs["ball_tracks"],
                    player_teams_path=inputs["player_teams"],
                    possession_path=inputs["possession"],
                    possession_debug_path=inputs["possession_debug"],
                    output_path=output_path,
                    k_nearest=args.k_nearest,
                    defender_radius_px=args.defender_radius_px,
                )
                output_rows = len(result.frame_table)
                total_output_rows += output_rows
                rows.append(
                    {
                        "clip_id": clip_id,
                        "status": "success",
                        "missing_inputs": "",
                        "input_row_counts": ";".join(
                            f"{key}={value}" for key, value in result.metadata["input_row_counts"].items()
                        ),
                        "output_path": str(result.output_csv),
                        "output_rows": output_rows,
                        "error_message": "",
                    }
                )
            except Exception as error:
                rows.append(
                    {
                        "clip_id": clip_id,
                        "status": "failed",
                        "missing_inputs": "",
                        "input_row_counts": "",
                        "output_path": "",
                        "output_rows": 0,
                        "error_message": str(error),
                    }
                )

        summary_path = ensure_output_parent(derived_root / "build_all_summary.csv")
        with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
            fieldnames = [
                "clip_id",
                "status",
                "missing_inputs",
                "input_row_counts",
                "output_path",
                "output_rows",
                "error_message",
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        success = sum(row["status"] == "success" for row in rows)
        skipped = sum(row["status"] == "skipped" for row in rows)
        failed = sum(row["status"] == "failed" for row in rows)
        print("Per-clip result table:")
        for row in rows:
            print(f"{row['clip_id']}: {row['status']} rows={row['output_rows']} missing={row['missing_inputs']} error={row['error_message']}")
        print(f"Candidate clips: {len(candidates)}")
        print(f"Successful clips: {success}")
        print(f"Skipped clips: {skipped}")
        print(f"Failed clips: {failed}")
        print(f"Total output rows: {total_output_rows}")
        print(f"Summary CSV: {summary_path}")
        return 0 if success > 0 else 1
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

