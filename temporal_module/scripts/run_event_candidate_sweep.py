"""Run the normal temporal event-candidate sweep end to end."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import ensure_output_parent, reject_outputs_path  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

STAGE_SUMMARIES = {
    "temporal_frames": "build_all_summary.csv",
    "weak_pass_events": "build_all_passes_summary.csv",
    "carries": "build_carry_labels_summary.csv",
    "pass_candidates": "pass_candidates_build_summary.csv",
    "turnover_candidates": "turnover_candidates_build_summary.csv",
    "shot_candidates": "shot_candidates_build_summary.csv",
    "refinement": "refined_event_candidates_build_summary.csv",
    "unified_events": "unified_event_candidates_build_summary.csv",
    "pass_scoring": "pass_candidate_scoring_build_summary.csv",
}

REQUIRED_FINAL_STAGES = [
    "temporal_frames",
    "weak_pass_events",
    "carries",
    "pass_candidates",
    "turnover_candidates",
    "shot_candidates",
    "refinement",
    "unified_events",
    "pass_scoring",
]


@dataclass(frozen=True)
class Stage:
    key: str
    label: str
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full per-clip temporal event-candidate sweep."
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--derived-root", default=str(Path("temporal_module") / "data" / "derived"))
    parser.add_argument("--k-nearest", type=int, default=4)
    parser.add_argument("--defender-radius-px", type=float, default=100.0)
    parser.add_argument("--max-merged-pass-span-frames", type=int, default=24)
    parser.add_argument("--interception-duplicate-frame-tolerance", type=int, default=2)
    parser.add_argument("--clip-id", default="", help="Exact derived clip directory name to process.")
    args = parser.parse_args()
    apply_stage_arg_defaults(args)
    return args


def apply_stage_arg_defaults(args: argparse.Namespace) -> None:
    defaults = {
        "max_transition_seconds": 2.0,
        "min_previous_owner_frames": 3,
        "min_new_owner_frames": 2,
        "motion_lookaround_frames": 12,
        "min_winner_frames": 2,
        "interception_match_frame_tolerance": 2,
        "speed_quantile": 0.95,
        "acceleration_quantile": 0.95,
        "goal_region_width_fraction": 0.20,
        "merge_gap_frames": 12,
        "pass_merge_gap_frames": 12,
        "min_pass_duration_seconds": 0.08,
        "min_pass_frame_span": 2,
        "turnover_context_before_frames": 8,
        "turnover_context_after_frames": 8,
        "min_turnover_previous_stable_frames": 3,
        "min_turnover_winner_stable_frames": 2,
        "shot_local_peak_window_frames": 6,
        "min_end_region_fraction": 0.20,
        "max_ball_interpolated_fraction_for_review": 0.25,
        "max_carry_overlap_fraction_for_review": 0.20,
        "interception_context_frames": 8,
        "sender_context_frames": 10,
        "receiver_context_frames": 10,
        "minimum_stable_frames": 4,
        "free_ball_distance_threshold_px": 35.0,
        "maximum_ball_missing_fraction": 0.25,
        "maximum_ball_interpolated_fraction": 0.25,
        "max_plausible_pass_duration_frames": 24,
        "post_turnover_context_frames": 5,
        "recovery_to_pass_overlap_tolerance_frames": 3,
        "recovery_to_pass_min_receiver_stable_frames": 2,
        "recovery_to_pass_min_frames_after_turnover": 2,
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)


def script_command(script_name: str, *args: str) -> list[str]:
    return [sys.executable, str(SCRIPT_DIR / script_name), *args]


def build_stages(args: argparse.Namespace) -> list[Stage]:
    outputs_root = str(args.outputs_root)
    derived_root = str(args.derived_root)
    return [
        Stage(
            "temporal_frames",
            "build temporal frames",
            script_command(
                "build_all_temporal_frames.py",
                "--outputs-root",
                outputs_root,
                "--derived-root",
                derived_root,
                "--k-nearest",
                str(args.k_nearest),
                "--defender-radius-px",
                str(args.defender_radius_px),
            ),
        ),
        Stage(
            "weak_pass_events",
            "export weak pass events",
            script_command(
                "build_all_pass_events.py",
                "--outputs-root",
                outputs_root,
                "--derived-root",
                derived_root,
            ),
        ),
        Stage(
            "carries",
            "build carries",
            script_command(
                "build_carry_labels.py",
                "--outputs-root",
                outputs_root,
                "--derived-root",
                derived_root,
            ),
        ),
        Stage(
            "pass_candidates",
            "build pass candidates",
            script_command(
                "build_pass_candidates.py",
                "--outputs-root",
                outputs_root,
                "--derived-root",
                derived_root,
            ),
        ),
        Stage(
            "turnover_candidates",
            "build turnover candidates",
            script_command(
                "build_turnover_candidates.py",
                "--outputs-root",
                outputs_root,
                "--derived-root",
                derived_root,
            ),
        ),
        Stage(
            "shot_candidates",
            "build shot candidates",
            script_command(
                "build_shot_candidates.py",
                "--outputs-root",
                outputs_root,
                "--derived-root",
                derived_root,
            ),
        ),
        Stage(
            "refinement",
            "refine event candidates",
            script_command(
                "refine_event_candidates.py",
                "--derived-root",
                derived_root,
                "--max-merged-pass-span-frames",
                str(args.max_merged_pass_span_frames),
            ),
        ),
        Stage(
            "unified_events",
            "build unified event candidates",
            script_command(
                "build_unified_event_candidates.py",
                "--outputs-root",
                outputs_root,
                "--derived-root",
                derived_root,
                "--interception-duplicate-frame-tolerance",
                str(args.interception_duplicate_frame_tolerance),
            ),
        ),
    ]


def pass_scoring_stage(args: argparse.Namespace) -> Stage:
    return Stage(
        "pass_scoring",
        "score pass candidates",
        script_command(
            "score_pass_candidates.py",
            "--derived-root",
            str(args.derived_root),
            "--post-turnover-context-frames",
            "5",
            "--recovery-to-pass-overlap-tolerance-frames",
            "3",
            "--recovery-to-pass-min-receiver-stable-frames",
            "2",
            "--recovery-to-pass-min-frames-after-turnover",
            "2",
        ),
    )


def run_stage(stage: Stage) -> int:
    print(f"\n=== {stage.label} ===")
    print(" ".join(stage.command))
    completed = subprocess.run(stage.command, check=False)
    if completed.returncode != 0:
        print(f"Stage failed: {stage.label} exit_code={completed.returncode}", file=sys.stderr)
    return int(completed.returncode)


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return {str(row.get("clip_id", "")): dict(row) for row in reader if row.get("clip_id")}


def collect_stage_summaries(derived_root: Path) -> dict[str, dict[str, dict[str, str]]]:
    return {
        stage: read_summary(derived_root / filename)
        for stage, filename in STAGE_SUMMARIES.items()
    }


def candidate_clip_ids(derived_root: Path, summaries: dict[str, dict[str, dict[str, str]]]) -> list[str]:
    clip_ids: set[str] = set()
    if derived_root.exists():
        clip_ids.update(path.name for path in derived_root.iterdir() if path.is_dir())
    for rows in summaries.values():
        clip_ids.update(rows)
    return sorted(clip_ids)


def available_derived_clip_ids(derived_root: Path) -> list[str]:
    if not derived_root.exists():
        return []
    return sorted(path.name for path in derived_root.iterdir() if path.is_dir())


def validate_selected_clip(derived_root: Path, clip_id: str) -> None:
    available = available_derived_clip_ids(derived_root)
    if clip_id in available:
        return
    lines = [
        f"Requested clip ID not found under {derived_root}: {clip_id}",
        "Available derived clip IDs:",
    ]
    lines.extend(f"- {available_clip_id}" for available_clip_id in available)
    if not available:
        lines.append("- <none>")
    raise ValueError("\n".join(lines))


def missing_pass_scoring_prerequisites(derived_root: Path) -> dict[str, str]:
    missing: dict[str, str] = {}
    if not derived_root.exists():
        return missing
    for clip_dir in sorted(path for path in derived_root.iterdir() if path.is_dir()):
        events_dir = clip_dir / "events"
        required = [
            events_dir / "pass_candidates_refined.csv",
            events_dir / "event_candidates_unified.csv",
        ]
        missing_files = [str(path) for path in required if not path.exists()]
        if missing_files:
            missing[clip_dir.name] = ";".join(missing_files)
    return missing


def missing_pass_scoring_prerequisites_for_clip(derived_root: Path, clip_id: str) -> dict[str, str]:
    events_dir = derived_root / clip_id / "events"
    required = [
        events_dir / "pass_candidates_refined.csv",
        events_dir / "event_candidates_unified.csv",
    ]
    missing_files = [str(path) for path in required if not path.exists()]
    return {clip_id: ";".join(missing_files)} if missing_files else {}


def write_combined_summary(
    derived_root: Path,
    stage_return_codes: dict[str, int],
    pass_scoring_prereq_failures: dict[str, str] | None = None,
    selected_clip_id: str = "",
    selected_stage_rows: dict[str, dict[str, str]] | None = None,
) -> tuple[Path, list[dict[str, str]]]:
    summaries = collect_stage_summaries(derived_root) if selected_stage_rows is None else {}
    pass_scoring_prereq_failures = pass_scoring_prereq_failures or {}
    selected_stage_rows = selected_stage_rows or {}
    rows: list[dict[str, str]] = []
    clip_ids = [selected_clip_id] if selected_clip_id else candidate_clip_ids(derived_root, summaries)
    for clip_id in clip_ids:
        row: dict[str, str] = {"clip_id": clip_id, "selected_clip_id": selected_clip_id}
        errors: list[str] = []
        for stage in REQUIRED_FINAL_STAGES:
            source = selected_stage_rows.get(stage, {}) if selected_clip_id else summaries.get(stage, {}).get(clip_id, {})
            status = str(source.get("status", "not_run"))
            error = str(source.get("error_message", ""))
            if stage == "pass_scoring" and clip_id in pass_scoring_prereq_failures:
                status = "failed"
                error = f"Missing pass scoring prerequisite(s): {pass_scoring_prereq_failures[clip_id]}"
            row[f"{stage}_status"] = status
            row[f"{stage}_error"] = error
            if status != "success":
                errors.append(f"{stage}={status}{': ' + error if error else ''}")
        for stage, return_code in stage_return_codes.items():
            row[f"{stage}_exit_code"] = str(return_code)
        row["final_status"] = "success" if not errors else "failed"
        row["error_message"] = "; ".join(errors)
        rows.append(row)

    fieldnames = ["clip_id", "selected_clip_id"]
    for stage in REQUIRED_FINAL_STAGES:
        fieldnames.extend([f"{stage}_status", f"{stage}_error"])
    for stage in REQUIRED_FINAL_STAGES:
        field = f"{stage}_exit_code"
        if any(field in row for row in rows):
            fieldnames.append(field)
    fieldnames.extend(["final_status", "error_message"])

    summary_path = ensure_output_parent(derived_root / "event_candidate_sweep_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return summary_path, rows


def normalize_stage_row(row: dict[str, Any]) -> dict[str, str]:
    return {str(key): "" if value is None else str(value) for key, value in row.items()}


def selected_failed_row(clip_id: str, error: Exception) -> dict[str, str]:
    return {"clip_id": clip_id, "status": "failed", "error_message": str(error)}


def temporal_frames_for_clip(clip_id: str, outputs_root: Path, derived_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    from build_all_temporal_frames import REQUIRED_RELATIVE_PATHS  # noqa: WPS433
    from src.feature_builder import build_temporal_frames  # noqa: WPS433
    from src.io_utils import format_missing_inputs  # noqa: WPS433

    clip_dir = outputs_root / clip_id
    inputs = {name: clip_dir / relative for name, relative in REQUIRED_RELATIVE_PATHS.items()}
    missing = format_missing_inputs(inputs)
    if missing:
        raise FileNotFoundError(f"Missing input(s): {';'.join(missing)}")
    result = build_temporal_frames(
        clip_id=clip_id,
        tracks_path=inputs["tracks"],
        ball_tracks_path=inputs["ball_tracks"],
        player_teams_path=inputs["player_teams"],
        possession_path=inputs["possession"],
        possession_debug_path=inputs["possession_debug"],
        output_path=derived_root / clip_id / "temporal_frames.csv",
        k_nearest=args.k_nearest,
        defender_radius_px=args.defender_radius_px,
    )
    return {
        "clip_id": clip_id,
        "status": "success",
        "output_rows": int(len(result.frame_table)),
        "output_path": str(result.output_csv),
        "error_message": "",
    }


def weak_pass_events_for_clip(clip_id: str, outputs_root: Path, derived_root: Path) -> dict[str, Any]:
    from build_all_pass_events import build_pass_events, write_pass_schema  # noqa: WPS433
    from src.io_utils import read_csv_with_header  # noqa: WPS433

    possession_path = outputs_root / clip_id / "possession" / "possession.csv"
    possession_debug_path = outputs_root / clip_id / "possession" / "possession_debug.csv"
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
    output_path = ensure_output_parent(derived_root / clip_id / "passes_weak.csv")
    events.to_csv(output_path, index=False)
    write_pass_schema(
        schema_path=output_path.with_name("passes_weak_schema.json"),
        clip_id=clip_id,
        possession_path=possession_path,
        possession_debug_path=possession_debug_path,
        possession_header=possession_header,
        possession_debug_header=possession_debug_header,
        event_count=len(events),
        output_columns=list(events.columns),
    )
    return {
        "clip_id": clip_id,
        "status": "success",
        "pass_event_count": int(len(events)),
        "input_possession_rows": int(len(possession_df)),
        "input_possession_debug_rows": int(len(possession_debug_df)),
        "output_path": str(output_path),
        "error_message": "",
    }


def carry_labels_for_clip(clip_id: str, outputs_root: Path, derived_root: Path) -> dict[str, Any]:
    from build_carry_labels import build_labels_for_clip  # noqa: WPS433

    temporal_frames_path = derived_root / clip_id / "temporal_frames.csv"
    carries_path = outputs_root / clip_id / "carries" / "carries.csv"
    if not carries_path.exists():
        raise FileNotFoundError(f"Missing carries CSV: {carries_path}")
    labels, metadata = build_labels_for_clip(
        clip_id=clip_id,
        temporal_frames_path=temporal_frames_path,
        carries_path=carries_path,
        output_path=derived_root / clip_id / "temporal_labels.csv",
    )
    return {
        "clip_id": clip_id,
        "status": "success",
        "total_frames": int(len(labels)),
        "carry_frames": int(metadata["carry_labeled_frames"]),
        "background_frames": int(metadata["background_labeled_frames"]),
        "accepted_carry_events": int(metadata["accepted_carry_event_count"]),
        "ignored_low_quality_carry_events": int(metadata["ignored_low_quality_carry_event_count"]),
        "output_path": str(derived_root / clip_id / "temporal_labels.csv"),
        "error_message": "",
    }


def run_selected_clip_stage(stage: Stage, clip_id: str, outputs_root: Path, derived_root: Path, args: argparse.Namespace) -> dict[str, str]:
    print(f"\n=== {stage.label} ===")
    try:
        if stage.key == "temporal_frames":
            row = temporal_frames_for_clip(clip_id, outputs_root, derived_root, args)
        elif stage.key == "weak_pass_events":
            row = weak_pass_events_for_clip(clip_id, outputs_root, derived_root)
        elif stage.key == "carries":
            row = carry_labels_for_clip(clip_id, outputs_root, derived_root)
        elif stage.key == "pass_candidates":
            from build_pass_candidates import process_clip  # noqa: WPS433

            row = process_clip(clip_id, outputs_root, derived_root, args)
        elif stage.key == "turnover_candidates":
            from build_turnover_candidates import process_clip  # noqa: WPS433

            row = process_clip(clip_id, outputs_root, derived_root, args)
        elif stage.key == "shot_candidates":
            from build_shot_candidates import process_clip  # noqa: WPS433

            row = process_clip(clip_id, outputs_root, derived_root, args)
        elif stage.key == "refinement":
            from refine_event_candidates import process_clip  # noqa: WPS433

            row = process_clip(clip_id, derived_root, args)
        elif stage.key == "unified_events":
            from build_unified_event_candidates import process_clip  # noqa: WPS433

            row = process_clip(clip_id, outputs_root, derived_root, args)
        elif stage.key == "pass_scoring":
            from score_pass_candidates import process_clip  # noqa: WPS433

            row = process_clip(clip_id, derived_root, args)
        else:
            raise ValueError(f"Unknown stage: {stage.key}")
    except Exception as error:
        row = selected_failed_row(clip_id, error)
    row = normalize_stage_row(row)
    print(f"{clip_id}: {row.get('status', 'not_run')} error={row.get('error_message', '')}")
    return row


def run_selected_clip_sweep(args: argparse.Namespace, derived_root: Path) -> tuple[Path, list[dict[str, str]], dict[str, int]]:
    clip_id = str(args.clip_id)
    validate_selected_clip(derived_root, clip_id)
    outputs_root = Path(args.outputs_root)
    stage_rows: dict[str, dict[str, str]] = {}
    stage_return_codes: dict[str, int] = {}

    for stage in build_stages(args):
        row = run_selected_clip_stage(stage, clip_id, outputs_root, derived_root, args)
        stage_rows[stage.key] = row
        stage_return_codes[stage.key] = 0 if row.get("status") == "success" else 1

    prereq_failures = missing_pass_scoring_prerequisites_for_clip(derived_root, clip_id)
    if prereq_failures:
        error = f"Missing pass scoring prerequisite(s): {prereq_failures[clip_id]}"
        print("\n=== score pass candidates ===", file=sys.stderr)
        print(f"{clip_id}: failed {error}", file=sys.stderr)
        stage_rows["pass_scoring"] = {"clip_id": clip_id, "status": "failed", "error_message": error}
        stage_return_codes["pass_scoring"] = 1
    else:
        scoring_stage = pass_scoring_stage(args)
        row = run_selected_clip_stage(scoring_stage, clip_id, outputs_root, derived_root, args)
        stage_rows[scoring_stage.key] = row
        stage_return_codes[scoring_stage.key] = 0 if row.get("status") == "success" else 1

    summary_path, rows = write_combined_summary(
        derived_root=derived_root,
        stage_return_codes=stage_return_codes,
        pass_scoring_prereq_failures=prereq_failures,
        selected_clip_id=clip_id,
        selected_stage_rows=stage_rows,
    )
    return summary_path, rows, stage_return_codes


def main() -> int:
    args = parse_args()
    try:
        derived_root = Path(args.derived_root)
        reject_outputs_path(derived_root)
        if args.clip_id:
            summary_path, rows, stage_return_codes = run_selected_clip_sweep(args, derived_root)
            failed = sum(row["final_status"] == "failed" for row in rows)
            print("\nPer-clip event-candidate sweep summary:")
            for row in rows:
                print(f"{row['clip_id']}: {row['final_status']} error={row['error_message']}")
            print(f"Successful clips: {sum(row['final_status'] == 'success' for row in rows)}")
            print(f"Failed clips: {failed}")
            print(f"Summary CSV: {summary_path}")
            return 0 if rows and failed == 0 and all(code == 0 for code in stage_return_codes.values()) else 1

        stage_return_codes: dict[str, int] = {}

        for stage in build_stages(args):
            stage_return_codes[stage.key] = run_stage(stage)

        prereq_failures = missing_pass_scoring_prerequisites(derived_root)
        if prereq_failures:
            print("\n=== score pass candidates ===", file=sys.stderr)
            for clip_id, missing in prereq_failures.items():
                print(f"{clip_id}: failed Missing pass scoring prerequisite(s): {missing}", file=sys.stderr)
            stage_return_codes["pass_scoring"] = 1
        else:
            scoring_stage = pass_scoring_stage(args)
            stage_return_codes[scoring_stage.key] = run_stage(scoring_stage)

        summary_path, rows = write_combined_summary(
            derived_root=derived_root,
            stage_return_codes=stage_return_codes,
            pass_scoring_prereq_failures=prereq_failures,
        )
        successful = sum(row["final_status"] == "success" for row in rows)
        failed = sum(row["final_status"] == "failed" for row in rows)
        print("\nPer-clip event-candidate sweep summary:")
        for row in rows:
            print(f"{row['clip_id']}: {row['final_status']} error={row['error_message']}")
        print(f"Successful clips: {successful}")
        print(f"Failed clips: {failed}")
        print(f"Summary CSV: {summary_path}")
        return 0 if rows and failed == 0 and all(code == 0 for code in stage_return_codes.values()) else 1
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
