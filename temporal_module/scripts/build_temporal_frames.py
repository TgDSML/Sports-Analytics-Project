"""CLI for building one canonical temporal frame table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_builder import build_temporal_frames  # noqa: E402
from src.io_utils import print_header_report, print_row_counts, reject_outputs_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical temporal frame features for one clip.")
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--ball-tracks", required=True)
    parser.add_argument("--player-teams", required=True)
    parser.add_argument("--possession", required=True)
    parser.add_argument("--possession-debug", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k-nearest", type=int, default=4)
    parser.add_argument("--defender-radius-px", type=float, default=100.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        reject_outputs_path(args.output)
        result = build_temporal_frames(
            clip_id=args.clip_id,
            tracks_path=args.tracks,
            ball_tracks_path=args.ball_tracks,
            player_teams_path=args.player_teams,
            possession_path=args.possession,
            possession_debug_path=args.possession_debug,
            output_path=args.output,
            k_nearest=args.k_nearest,
            defender_radius_px=args.defender_radius_px,
        )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print_header_report(result.metadata["detected_source_headers"])
    print_row_counts(result.metadata["input_row_counts"])
    print(f"Output path: {result.output_csv}")
    print(f"Schema path: {result.output_schema}")
    print(f"Output row count: {len(result.frame_table)}")
    print("Output columns:")
    print(", ".join(result.frame_table.columns))
    print("Missing-value summary:")
    for column, count in result.metadata["missing_value_count"].items():
        print(f"- {column}: {count}")
    print("First 5 output rows:")
    print(result.frame_table.head(5).to_string(index=False))
    if result.metadata["assumptions"]:
        print("Assumptions:")
        for assumption in result.metadata["assumptions"]:
            print(f"- {assumption}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

