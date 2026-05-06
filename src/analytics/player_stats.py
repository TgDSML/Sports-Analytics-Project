"""Generate basic player movement statistics from tracking CSV files."""

import argparse
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import Font
import pandas as pd


def generate_player_stats(
    tracks_csv_path,
    output_csv_path,
    readable_output_path=None,
    markdown_output_path=None,
    excel_output_path=None,
):
    """Generate raw and optional readable movement summaries for each player."""
    tracks_path = Path(tracks_csv_path)
    if not tracks_path.exists():
        raise FileNotFoundError(f"Tracks CSV not found: {tracks_path}")

    tracks = pd.read_csv(tracks_path)
    required_columns = {"track_id", "confidence", "center_x", "center_y", "timestamp"}
    missing_columns = required_columns - set(tracks.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Tracks CSV missing required column(s): {columns}")

    sort_columns = [column for column in ("track_id", "frame", "timestamp") if column in tracks.columns]
    if sort_columns:
        tracks = tracks.sort_values(sort_columns)

    rows = []
    for track_id, group in tracks.groupby("track_id"):
        clean_group = group.dropna(subset=["center_x", "center_y", "timestamp"])
        frames_seen = int(len(clean_group))
        avg_confidence = float(clean_group["confidence"].mean()) if frames_seen else 0.0
        total_distance = _total_pixel_distance(clean_group)
        total_time = _total_time(clean_group)
        avg_speed = total_distance / total_time if total_time > 0 else 0.0

        rows.append(
            {
                "track_id": int(track_id),
                "frames_seen": frames_seen,
                "avg_confidence": avg_confidence,
                "total_pixel_distance": total_distance,
                "avg_speed_pixels_per_sec": avg_speed,
                "time_visible_sec": total_time,
            }
        )

    stats = pd.DataFrame(
        rows,
        columns=[
            "track_id",
            "frames_seen",
            "avg_confidence",
            "total_pixel_distance",
            "avg_speed_pixels_per_sec",
            "time_visible_sec",
        ],
    )

    output = Path(output_csv_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_columns = [
        "track_id",
        "frames_seen",
        "avg_confidence",
        "total_pixel_distance",
        "avg_speed_pixels_per_sec",
    ]
    stats[raw_columns].to_csv(output, index=False)

    readable_stats = _build_readable_stats(stats)
    if readable_output_path is not None:
        _write_readable_csv(readable_stats, readable_output_path)
    if markdown_output_path is not None:
        _write_markdown_table(readable_stats, markdown_output_path)
    if excel_output_path is not None:
        _write_excel(readable_stats, excel_output_path)


def _total_pixel_distance(group) -> float:
    points = group[["center_x", "center_y"]].to_numpy(dtype=float)
    if len(points) < 2:
        return 0.0

    deltas = np.diff(points, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    return float(distances.sum())


def _total_time(group) -> float:
    timestamps = group["timestamp"].to_numpy(dtype=float)
    if len(timestamps) < 2:
        return 0.0
    return float(timestamps[-1] - timestamps[0])


def _build_readable_stats(stats: pd.DataFrame) -> pd.DataFrame:
    readable = stats.rename(
        columns={
            "track_id": "Player ID",
            "frames_seen": "Frames Seen",
            "avg_confidence": "Average Confidence",
            "total_pixel_distance": "Total Distance (pixels)",
            "avg_speed_pixels_per_sec": "Average Speed (pixels/sec)",
            "time_visible_sec": "Time Visible (sec)",
        }
    )

    readable = readable[
        [
            "Player ID",
            "Frames Seen",
            "Average Confidence",
            "Total Distance (pixels)",
            "Average Speed (pixels/sec)",
            "Time Visible (sec)",
        ]
    ]

    if readable.empty:
        return readable

    readable = readable.sort_values(
        ["Frames Seen", "Total Distance (pixels)"],
        ascending=[False, False],
    )
    readable["Average Confidence"] = readable["Average Confidence"].round(3)
    readable["Total Distance (pixels)"] = readable["Total Distance (pixels)"].round(2)
    readable["Average Speed (pixels/sec)"] = readable[
        "Average Speed (pixels/sec)"
    ].round(2)
    readable["Time Visible (sec)"] = readable["Time Visible (sec)"].round(2)
    return readable


def _write_readable_csv(readable_stats: pd.DataFrame, output_path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    readable_stats.to_csv(output, index=False)


def _write_markdown_table(readable_stats: pd.DataFrame, output_path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    columns = list(readable_stats.columns)
    lines = [
        "# Player Movement Statistics",
        "",
        _markdown_row(columns),
        _markdown_row(["---"] * len(columns)),
    ]
    for row in readable_stats.itertuples(index=False, name=None):
        lines.append(_markdown_row(row))

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_row(values) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def _write_excel(readable_stats: pd.DataFrame, output_path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    readable_stats.to_excel(output, index=False)

    workbook = load_workbook(output)
    worksheet = workbook.active
    worksheet.freeze_panes = "A2"

    for cell in worksheet[1]:
        cell.font = Font(bold=True)

    for column_cells in worksheet.columns:
        max_length = max(len(str(cell.value or "")) for cell in column_cells)
        column_letter = column_cells[0].column_letter
        worksheet.column_dimensions[column_letter].width = max_length + 2

    workbook.save(output)


def main() -> None:
    """Parse CLI arguments and generate player stats from an existing tracks CSV."""
    parser = argparse.ArgumentParser(
        description="Generate basic player movement statistics from a tracks CSV"
    )
    parser.add_argument("--tracks-csv", required=True, help="Path to the tracking CSV file")
    parser.add_argument("--output", required=True, help="Path to save the raw stats CSV")
    parser.add_argument(
        "--readable-output",
        help="Optional path to save a human-friendly stats CSV",
    )
    parser.add_argument(
        "--markdown-output",
        help="Optional path to save a Markdown stats table",
    )
    parser.add_argument(
        "--excel-output",
        help="Optional path to save a formatted Excel stats workbook",
    )
    args = parser.parse_args()

    try:
        generate_player_stats(
            tracks_csv_path=args.tracks_csv,
            output_csv_path=args.output,
            readable_output_path=args.readable_output,
            markdown_output_path=args.markdown_output,
            excel_output_path=args.excel_output,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}")
        return

    print(f"Player stats saved to: {args.output}")
    if args.readable_output:
        print(f"Readable player stats saved to: {args.readable_output}")
    if args.markdown_output:
        print(f"Markdown player stats saved to: {args.markdown_output}")
    if args.excel_output:
        print(f"Excel player stats saved to: {args.excel_output}")


if __name__ == "__main__":
    main()
