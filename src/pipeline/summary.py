"""Dataset-level summary outputs."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from src.pipeline.run_clip import ClipRunResult


def write_dataset_summary(
    results: list[ClipRunResult],
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write dataset summary CSV and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dataset_summary.csv"
    md_path = output_dir / "dataset_summary.md"

    stage_counts: Counter[str] = Counter()
    stage_failures: Counter[str] = Counter()
    for result in results:
        for stage in result.stages:
            stage_counts[stage.stage] += 1
            if not stage.success:
                stage_failures[stage.stage] += 1

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "clip_id",
                "status",
                "output_root",
                "failed_stage",
                "stage_statuses",
            ],
        )
        writer.writeheader()
        for result in results:
            failed_stage = next(
                (stage.stage for stage in result.stages if not stage.success),
                "",
            )
            writer.writerow(
                {
                    "clip_id": result.clip_id,
                    "status": "succeeded" if result.success else "failed",
                    "output_root": result.output_root,
                    "failed_stage": failed_stage,
                    "stage_statuses": "; ".join(
                        f"{stage.stage}={'ok' if stage.success else 'failed'}"
                        for stage in result.stages
                    ),
                }
            )

    total = len(results)
    succeeded = sum(result.success for result in results)
    failed = total - succeeded
    lines = [
        "# Dataset Pipeline Summary",
        "",
        f"- Total clips attempted: {total}",
        f"- Total clips succeeded: {succeeded}",
        f"- Total clips failed: {failed}",
        "",
        "## Stage Counts",
        "",
    ]
    for stage in sorted(stage_counts):
        success_count = stage_counts[stage] - stage_failures[stage]
        lines.append(
            f"- {stage}: {success_count} succeeded, {stage_failures[stage]} failed"
        )

    lines.extend(["", "## Clip Outputs", ""])
    for result in results:
        status = "succeeded" if result.success else "failed"
        lines.append(f"- {result.clip_id}: {status} -> {result.output_root}")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path
