"""Create a fixed clip-level split for carry/background windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import PROJECT_ROOT, ensure_output_parent, reject_outputs_path, utc_now_iso  # noqa: E402


SPLIT_CLIPS = {
    "train": [
        "england_epl__2015_2016__2015_09_26___17_00_Manchester_United_3___0_Sunderland__h1_720p",
        "england_epl__2015_2016__2015_09_26___17_00_Manchester_United_3___0_Sunderland__h2_720p",
        "england_epl__2015_2016__2016_03_05___18_00_Manchester_City_4___0_Aston_Villa__h2_720p",
        "england_epl__2015_2016__2016_03_20___19_00_Manchester_City_0___1_Manchester_United__h2_720p",
        "england_epl__2016_2017__2016_10_02___18_30_Burnley_0___1_Arsenal__h1_720p",
    ],
    "validation": [
        "england_epl__2014_2015__2015_04_11___19_30_Burnley_0___1_Arsenal__h1_720p",
    ],
    "test": [
        "england_epl__2015_2016__2016_01_24___19_00_Arsenal_0___1_Chelsea__h1_720p",
    ],
    "excluded": [
        "england_epl__2015_2016__2015_10_03___17_00_Manchester_City_6___1_Newcastle_Utd__h1_720p",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create fixed clip-level train/validation/test split for carry/background windows."
    )
    parser.add_argument("--dataset-root", default=str(Path("temporal_module") / "data" / "datasets"))
    return parser.parse_args()


def enforce_dataset_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = (PROJECT_ROOT / "temporal_module" / "data" / "datasets").resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"Dataset outputs must be under {allowed}: {resolved}") from error
    reject_outputs_path(resolved)
    return resolved


def assignment_map() -> dict[str, str]:
    result = {}
    duplicates = []
    for split, clip_ids in SPLIT_CLIPS.items():
        for clip_id in clip_ids:
            if clip_id in result:
                duplicates.append(clip_id)
            result[clip_id] = split
    if duplicates:
        raise ValueError(f"Clip appears in more than one split: {', '.join(sorted(set(duplicates)))}")
    return result


def require_columns(index: pd.DataFrame, path: Path) -> None:
    required = {"clip_id", "label", "label_id"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"{path} missing column(s): {', '.join(sorted(missing))}")


def summarize_split(split_index: pd.DataFrame) -> dict[str, Any]:
    summary = {}
    for split in ["train", "validation", "test", "excluded"]:
        rows = split_index[split_index["split"] == split]
        summary[split] = {
            "clips": int(rows["clip_id"].nunique()),
            "total_windows": int(len(rows)),
            "carry_windows": int((rows["label_id"].astype(int) == 1).sum()),
            "background_windows": int((rows["label_id"].astype(int) == 0).sum()),
        }
    return summary


def main() -> int:
    args = parse_args()
    try:
        dataset_root = enforce_dataset_root(Path(args.dataset_root))
        index_path = dataset_root / "carry_background_windows_index.csv"
        if not index_path.exists():
            raise FileNotFoundError(f"Window index CSV not found: {index_path}")

        index = pd.read_csv(index_path)
        require_columns(index, index_path)
        mapping = assignment_map()
        index_clips = set(index["clip_id"].astype(str).unique())
        assigned_clips = set(mapping)
        missing_assignments = sorted(index_clips - assigned_clips)
        if missing_assignments:
            raise ValueError(
                "Clip(s) in window index are not assigned to any split: "
                + ", ".join(missing_assignments)
            )

        split_index = index.copy()
        split_index["split"] = split_index["clip_id"].map(mapping)
        if split_index["split"].isna().any():
            missing = sorted(split_index.loc[split_index["split"].isna(), "clip_id"].astype(str).unique())
            raise ValueError(f"Unassigned row(s) found after split join: {', '.join(missing)}")

        output_csv = ensure_output_parent(dataset_root / "carry_background_split.csv")
        output_json = ensure_output_parent(dataset_root / "carry_background_split.json")
        split_index.to_csv(output_csv, index=False)

        split_summary = summarize_split(split_index)
        payload = {
            "build_timestamp": utc_now_iso(),
            "input_index_path": str(index_path),
            "output_csv_path": str(output_csv),
            "clip_assignments": SPLIT_CLIPS,
            "per_split_counts": split_summary,
            "validation": {
                "no_clip_in_multiple_splits": True,
                "every_index_clip_assigned_once": True,
                "index_clip_count": int(len(index_clips)),
                "assigned_clip_count": int(len(index_clips & assigned_clips)),
            },
        }
        output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        print("Carry/background split summary:")
        for split in ["train", "validation", "test", "excluded"]:
            counts = split_summary[split]
            print(
                f"{split}: clips={counts['clips']} windows={counts['total_windows']} "
                f"carry={counts['carry_windows']} background={counts['background_windows']}"
            )
        print(f"Output CSV: {output_csv}")
        print(f"Output JSON: {output_json}")
        return 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

