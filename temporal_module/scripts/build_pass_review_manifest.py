"""Build human-review manifests from scored pass candidates."""

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

from src.io_utils import PROJECT_ROOT, ensure_output_parent, reject_outputs_path, utc_now_iso  # noqa: E402


PRESERVED_COLUMNS = [
    "clip_id",
    "refined_event_id",
    "start_frame",
    "end_frame",
    "center_frame",
    "team",
    "pass_plausibility_score",
    "pass_score_tier",
    "pass_score_status",
    "is_eligible_for_weak_label",
    "is_eligible_for_pass_score_weak_label",
    "pass_score_reasons",
]

REVIEW_COLUMNS = [
    "review_context_start_frame",
    "review_context_end_frame",
    "review_priority",
    "manual_label",
    "manual_confidence",
    "manual_release_frame",
    "manual_receiver_control_frame",
    "manual_reviewer",
    "manual_review_notes",
    "manual_reviewed_at",
    "manual_override_weak_label",
]

OUTPUT_COLUMNS = PRESERVED_COLUMNS + REVIEW_COLUMNS

SUMMARY_COLUMNS = [
    "clip_id",
    "status",
    "scored_pass_rows",
    "manifest_rows_written",
    "high_priority_rows",
    "medium_priority_rows",
    "low_priority_rows",
    "score_weak_label_candidate_rows",
    "output_path",
    "error_message",
]

ALLOWED_MANUAL_LABELS = ["true_pass", "not_pass", "uncertain", "not_reviewed"]
ALLOWED_MANUAL_CONFIDENCE = ["high", "medium", "low"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pass-review manifests from pass_candidates_scored.csv files.")
    parser.add_argument("--derived-root", default=str(Path("temporal_module") / "data" / "derived"))
    parser.add_argument("--review-context-before-frames", type=int, default=20)
    parser.add_argument("--review-context-after-frames", type=int, default=20)
    return parser.parse_args()


def enforce_derived_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = (PROJECT_ROOT / "temporal_module" / "data" / "derived").resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"Review manifest outputs must be under {allowed}: {resolved}") from error
    reject_outputs_path(resolved)
    return resolved


def candidate_clip_ids(derived_root: Path) -> list[str]:
    if not derived_root.exists():
        raise FileNotFoundError(f"Derived root not found: {derived_root}")
    return sorted(
        path.name
        for path in derived_root.iterdir()
        if path.is_dir() and (path / "events" / "pass_candidates_scored.csv").exists()
    )


def get_value(row: pd.Series, column: str, default: Any = "") -> Any:
    if column not in row.index:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return value


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value) or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value) or str(value).strip() == "":
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def review_priority(row: pd.Series) -> str:
    score_weak = to_int(get_value(row, "is_eligible_for_pass_score_weak_label", 0), 0) == 1
    tier = str(get_value(row, "pass_score_tier", "")).strip().lower()
    if score_weak or tier == "high":
        return "high_priority"
    if tier == "medium":
        return "medium_priority"
    return "low_priority"


def build_manifest(scored: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, source in scored.iterrows():
        row = {column: get_value(source, column, "") for column in PRESERVED_COLUMNS}
        start_frame = to_int(get_value(source, "start_frame", 0), 0)
        end_frame = to_int(get_value(source, "end_frame", start_frame), start_frame)
        row.update(
            {
                "review_context_start_frame": max(0, start_frame - args.review_context_before_frames),
                "review_context_end_frame": max(0, end_frame + args.review_context_after_frames),
                "review_priority": review_priority(source),
                "manual_label": "",
                "manual_confidence": "",
                "manual_release_frame": "",
                "manual_receiver_control_frame": "",
                "manual_reviewer": "",
                "manual_review_notes": "",
                "manual_reviewed_at": "",
                "manual_override_weak_label": "",
            }
        )
        rows.append(row)

    manifest = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if manifest.empty:
        return manifest

    manifest["_score_weak_sort"] = pd.to_numeric(
        manifest["is_eligible_for_pass_score_weak_label"],
        errors="coerce",
    ).fillna(0)
    manifest["_score_sort"] = pd.to_numeric(manifest["pass_plausibility_score"], errors="coerce").fillna(-1)
    manifest["_center_sort"] = pd.to_numeric(manifest["center_frame"], errors="coerce").fillna(10**12)
    manifest = manifest.sort_values(
        ["_score_weak_sort", "_score_sort", "clip_id", "_center_sort"],
        ascending=[False, False, True, True],
    ).drop(columns=["_score_weak_sort", "_score_sort", "_center_sort"])
    return manifest.reset_index(drop=True)


def write_schema(path: Path, input_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    payload = {
        "build_timestamp": utc_now_iso(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "output_columns": OUTPUT_COLUMNS,
        "allowed_manual_label_values": ALLOWED_MANUAL_LABELS,
        "allowed_manual_confidence_values": ALLOWED_MANUAL_CONFIDENCE,
        "manual_override_weak_label_rules": [
            "blank when not reviewed",
            "1 only when manual_label == true_pass and manual_confidence is high or medium",
            "0 when manual_label == not_pass",
            "blank when manual_label == uncertain",
        ],
        "ordering": [
            "is_eligible_for_pass_score_weak_label descending",
            "pass_plausibility_score descending",
            "clip_id ascending",
            "center_frame ascending",
        ],
        "review_priority_rules": {
            "high_priority": "score weak-label candidate or high score tier",
            "medium_priority": "medium score tier",
            "low_priority": "low score tier",
        },
        "settings": {
            "review_context_before_frames": args.review_context_before_frames,
            "review_context_after_frames": args.review_context_after_frames,
        },
        "warnings": [
            "This manifest is for human review and calibration.",
            "It does not modify scored candidates, refined candidates, raw candidates, unified catalogs, or model-label files.",
        ],
    }
    output = ensure_output_parent(path)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def process_clip(clip_id: str, derived_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    scored_path = derived_root / clip_id / "events" / "pass_candidates_scored.csv"
    output_path = derived_root / clip_id / "events" / "pass_review_manifest.csv"
    schema_path = derived_root / clip_id / "events" / "pass_review_manifest_schema.json"
    scored = pd.read_csv(scored_path)
    manifest = build_manifest(scored, args)
    output = ensure_output_parent(output_path)
    manifest.to_csv(output, index=False)
    write_schema(schema_path, scored_path, output_path, args)
    return {
        "clip_id": clip_id,
        "status": "success",
        "scored_pass_rows": int(len(scored)),
        "manifest_rows_written": int(len(manifest)),
        "high_priority_rows": int((manifest["review_priority"] == "high_priority").sum()) if not manifest.empty else 0,
        "medium_priority_rows": int((manifest["review_priority"] == "medium_priority").sum()) if not manifest.empty else 0,
        "low_priority_rows": int((manifest["review_priority"] == "low_priority").sum()) if not manifest.empty else 0,
        "score_weak_label_candidate_rows": int(pd.to_numeric(manifest.get("is_eligible_for_pass_score_weak_label", 0), errors="coerce").fillna(0).sum()) if not manifest.empty else 0,
        "output_path": str(output),
        "error_message": "",
    }


def failed_row(clip_id: str, error: Exception) -> dict[str, Any]:
    row: dict[str, Any] = {column: 0 for column in SUMMARY_COLUMNS}
    row["clip_id"] = clip_id
    row["status"] = "failed"
    row["output_path"] = ""
    row["error_message"] = str(error)
    return row


def main() -> int:
    args = parse_args()
    try:
        derived_root = enforce_derived_root(Path(args.derived_root))
        rows: list[dict[str, Any]] = []
        for clip_id in candidate_clip_ids(derived_root):
            try:
                row = process_clip(clip_id, derived_root, args)
            except Exception as error:
                row = failed_row(clip_id, error)
            rows.append(row)
            print(
                f"{clip_id}: {row['status']} manifest_rows={row['manifest_rows_written']} "
                f"high={row['high_priority_rows']} medium={row['medium_priority_rows']} "
                f"low={row['low_priority_rows']} error={row['error_message']}"
            )

        summary_path = ensure_output_parent(derived_root / "pass_review_manifest_build_summary.csv")
        with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Summary CSV: {summary_path}")
        return 0 if rows and all(row["status"] == "success" for row in rows) else 1
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
