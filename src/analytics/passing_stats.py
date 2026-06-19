"""Estimate pass counts from possession output."""

import argparse
import csv
from pathlib import Path

import pandas as pd


def write_summary(
    possession_csv: Path,
    output_csv: Path,
    output_md: Path,
) -> dict[str, float]:
    """Write pass summary files from a possession CSV."""
    possession_df = pd.read_csv(possession_csv)
    active_possession = possession_df[
        possession_df["team"].isin(["Team A", "Team B"])
    ].copy()

    if active_possession.empty:
        summary = {
            "successful_passes": 0,
            "failed_passes": 0,
            "total_passes": 0,
            "pass_accuracy": 0.0,
        }
    else:
        player_changed = (
            active_possession["nearest_player_id"]
            != active_possession["nearest_player_id"].shift()
        )
        team_changed = active_possession["team"] != active_possession["team"].shift()
        ball_transfers = active_possession[player_changed].dropna(
            subset=["nearest_player_id"]
        )

        success_passes = ball_transfers[~team_changed.reindex(ball_transfers.index)]
        failed_passes = ball_transfers[team_changed.reindex(ball_transfers.index)]
        success = len(success_passes)
        failed = len(failed_passes)
        total = success + failed
        summary = {
            "successful_passes": success,
            "failed_passes": failed,
            "total_passes": total,
            "pass_accuracy": (success / total) * 100 if total else 0.0,
        }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric, value in summary.items():
            writer.writerow({"metric": metric, "value": value})

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(
        "\n".join(
            [
                "# Passing Summary",
                "",
                f"- Successful passes: {summary['successful_passes']}",
                f"- Failed passes: {summary['failed_passes']}",
                f"- Total passes: {summary['total_passes']}",
                f"- Pass accuracy: {summary['pass_accuracy']:.2f}%",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate pass counts from possession output")
    parser.add_argument(
        "--possession-csv",
        type=Path,
        default=Path("outputs/possession_30s_720p.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/passing_summary_30s_720p.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/passing_summary_30s_720p.md"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = write_summary(args.possession_csv, args.output_csv, args.output_md)
    print(f"Passing summary saved to: {args.output_csv}")
    print(f"Passing report saved to: {args.output_md}")
    print(f"Successful passes: {summary['successful_passes']}")
    print(f"Failed passes: {summary['failed_passes']}")
    print(f"Pass accuracy: {summary['pass_accuracy']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
