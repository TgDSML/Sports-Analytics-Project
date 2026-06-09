"""Compare generic and football-specific ball detection models."""

import argparse
import csv
from pathlib import Path

from src.detection.ball_detector import BallFilterConfig, process_ball_detection_video


GENERIC_MODEL_NAME = "yolov8n.pt"


def parse_summary(summary_csv_path: Path) -> dict[tuple[str, str], str]:
    """Read diagnostics summary rows into a dictionary."""
    with summary_csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return {
            (row["section"], row["metric"]): row["value"]
            for row in reader
            if row.get("section") and row.get("metric")
        }


def run_model(
    label: str,
    model_path: str,
    video_path: Path,
    output_dir: Path,
    conf: float,
    imgsz: int,
    filter_config: BallFilterConfig,
) -> dict:
    """Run one model and return comparison metadata."""
    model_dir = output_dir / label
    raw_csv = model_dir / "ball_detections_raw.csv"
    filtered_csv = model_dir / "ball_detections_filtered.csv"
    video = model_dir / "ball_detected_filtered.mp4"
    summary_csv = model_dir / "ball_detection_summary.csv"
    summary_md = model_dir / "ball_detection_summary.md"
    raw_count, filtered_count = process_ball_detection_video(
        video_path=video_path,
        raw_output_csv_path=raw_csv,
        filtered_output_csv_path=filtered_csv,
        output_video_path=video,
        summary_csv_path=summary_csv,
        summary_md_path=summary_md,
        model_path=model_path,
        conf=conf,
        imgsz=imgsz,
        filter_config=filter_config,
        debug_dir=model_dir,
        debug_frame_stride=0,
    )
    summary = parse_summary(summary_csv)
    return {
        "label": label,
        "model": model_path,
        "raw_count": raw_count,
        "filtered_count": filtered_count,
        "raw_csv": raw_csv,
        "filtered_csv": filtered_csv,
        "video": video,
        "summary_csv": summary_csv,
        "summary": summary,
    }


def build_comparison_markdown(generic: dict, candidate: dict) -> str:
    """Build a concise model comparison report."""
    rows = []
    for result in (generic, candidate):
        summary = result["summary"]
        frames = float(summary.get(("counts", "frames"), 0) or 0)
        filtered = float(result["filtered_count"])
        raw = float(result["raw_count"])
        rows.append(
            {
                "label": result["label"],
                "model": result["model"],
                "raw": int(raw),
                "filtered": int(filtered),
                "raw_per_frame": raw / frames if frames else 0,
                "filtered_per_frame": filtered / frames if frames else 0,
                "filtered_zero": summary.get(("counts", "filtered_frames_zero"), "n/a"),
                "filtered_one": summary.get(("counts", "filtered_frames_one"), "n/a"),
                "filtered_multiple": summary.get(("counts", "filtered_frames_multiple"), "n/a"),
                "confidence": summary.get(("filtered_distribution", "confidence"), "n/a"),
                "width": summary.get(("filtered_distribution", "width"), "n/a"),
                "height": summary.get(("filtered_distribution", "height"), "n/a"),
                "area": summary.get(("filtered_distribution", "area"), "n/a"),
            }
        )

    lines = [
        "# Ball Model Comparison",
        "",
        "## Models",
        "",
    ]
    for row in rows:
        lines.append(f"- {row['label']}: `{row['model']}`")

    lines.extend(
        [
            "",
            "## Detection Counts",
            "",
            "| Model | Raw detections | Filtered detections | Raw/frame | Filtered/frame | Filtered frames 0/1/multiple |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['raw']} | {row['filtered']} | "
            f"{row['raw_per_frame']:.3f} | {row['filtered_per_frame']:.3f} | "
            f"{row['filtered_zero']} / {row['filtered_one']} / {row['filtered_multiple']} |"
        )

    lines.extend(["", "## Filtered Confidence And Box Distributions", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['label']}",
                "",
                f"- confidence: {row['confidence']}",
                f"- width: {row['width']}",
                f"- height: {row['height']}",
                f"- area: {row['area']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Visual QA",
            "",
            f"- Generic filtered video: `{generic['video']}`",
            f"- Candidate filtered video: `{candidate['video']}`",
            "",
            "Inspect both videos directly. Counts alone do not prove ball quality because false positives can still pass size and top-region filters.",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compare two ball detection models")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--generic-model", default=GENERIC_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ball_debug/model_comparison"))
    parser.add_argument("--report-output", type=Path, default=Path("outputs/ball_debug/model_comparison.md"))
    parser.add_argument("--generic-conf", type=float, default=0.10)
    parser.add_argument("--candidate-conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--ball-min-area", type=int, default=20)
    parser.add_argument("--ball-max-area", type=int, default=500)
    parser.add_argument("--ball-min-width", type=int, default=4)
    parser.add_argument("--ball-max-width", type=int, default=30)
    parser.add_argument("--ball-min-height", type=int, default=4)
    parser.add_argument("--ball-max-height", type=int, default=30)
    parser.add_argument("--ball-max-detections-per-frame", type=int, default=1)
    parser.add_argument("--ball-exclude-top-ratio", type=float, default=0.08)
    parser.add_argument("--ball-exclude-bottom-ratio", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    """Run the comparison."""
    args = parse_args()
    filter_config = BallFilterConfig(
        min_area=args.ball_min_area,
        max_area=args.ball_max_area,
        min_width=args.ball_min_width,
        max_width=args.ball_max_width,
        min_height=args.ball_min_height,
        max_height=args.ball_max_height,
        max_detections_per_frame=args.ball_max_detections_per_frame,
        exclude_top_ratio=args.ball_exclude_top_ratio,
        exclude_bottom_ratio=args.ball_exclude_bottom_ratio,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generic = run_model(
        label="generic_yolov8n",
        model_path=args.generic_model,
        video_path=args.video,
        output_dir=args.output_dir,
        conf=args.generic_conf,
        imgsz=args.imgsz,
        filter_config=filter_config,
    )
    candidate = run_model(
        label="candidate",
        model_path=args.candidate_model,
        video_path=args.video,
        output_dir=args.output_dir,
        conf=args.candidate_conf,
        imgsz=args.imgsz,
        filter_config=filter_config,
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        build_comparison_markdown(generic=generic, candidate=candidate),
        encoding="utf-8",
    )
    print(f"Ball model comparison saved to: {args.report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
