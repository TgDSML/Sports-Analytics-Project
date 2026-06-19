"""Run enabled manifest clips through the batch analytics pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline.config import DEFAULT_CONFIG_PATH, get_config_value, load_config
from src.pipeline.logging_utils import configure_logger
from src.pipeline.manifest import read_manifest
from src.pipeline.run_clip import ClipRunResult, process_clip
from src.pipeline.summary import write_dataset_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dataset batch analytics pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--clip-id", action="append", dest="clip_ids")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    manifest_path = args.manifest or Path(get_config_value(config, "data.manifest_path"))
    outputs_root = Path(get_config_value(config, "runtime.outputs_dir", "outputs"))
    continue_on_error = bool(get_config_value(config, "runtime.continue_on_error", True))
    logger = configure_logger("dataset", outputs_root / "dataset_pipeline.log")

    rows = read_manifest(manifest_path, enabled_only=True)
    if args.clip_ids:
        requested = set(args.clip_ids)
        rows = [row for row in rows if row.clip_id in requested]

    if not rows:
        logger.error("No enabled manifest rows to process.")
        return 1

    results: list[ClipRunResult] = []
    logger.info("Starting dataset run for %s clip(s)", len(rows))
    for index, row in enumerate(rows, start=1):
        logger.info("[%s/%s] %s", index, len(rows), row.clip_id)
        result = process_clip(row, config)
        results.append(result)
        if not result.success and not continue_on_error:
            logger.error("Stopping after failed clip: %s", row.clip_id)
            break

    summary_csv, summary_md = write_dataset_summary(results, outputs_root)
    succeeded = sum(result.success for result in results)
    failed = len(results) - succeeded
    logger.info("Dataset run complete: %s succeeded, %s failed", succeeded, failed)
    logger.info("Summary CSV: %s", summary_csv)
    logger.info("Summary report: %s", summary_md)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
