"""Build a read-only SoccerNet download plan without fetching files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANIFEST_COLUMNS = [
    "local_clip_id",
    "exists_in_outputs",
    "exists_in_derived",
    "local_status",
    "has_player_teams",
    "has_possession",
    "has_temporal_frames",
    "has_unified_events",
    "has_pass_scoring",
    "notes",
]

ADDED_PLAN_COLUMNS = [
    "is_exact_local_duplicate",
    "matching_local_clip_id",
    "download_recommendation",
    "selection_rank",
    "selection_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a duplicate-aware, read-only plan for new SoccerNet clips."
    )
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--derived-root", default=str(Path("temporal_module") / "data" / "derived"))
    parser.add_argument("--soccernet-inventory", default="")
    parser.add_argument("--target-new-clips", type=int, default=25)
    parser.add_argument("--output-dir", default=str(Path("temporal_module") / "data" / "soccernet_inventory"))
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def immediate_dir_names(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {path.name for path in root.iterdir() if path.is_dir()}


def as_int(value: bool) -> int:
    return 1 if value else 0


def local_status(
    exists_in_outputs: bool,
    exists_in_derived: bool,
    has_player_teams: bool,
    has_possession: bool,
    has_temporal_frames: bool,
    has_unified_events: bool,
    has_pass_scoring: bool,
) -> str:
    if exists_in_derived and not exists_in_outputs:
        return "derived_only"
    if exists_in_outputs and not (has_player_teams and has_possession):
        return "incomplete_source_inputs"
    if exists_in_outputs and not exists_in_derived:
        return "outputs_only"
    if (
        exists_in_outputs
        and exists_in_derived
        and has_player_teams
        and has_possession
        and has_temporal_frames
        and has_unified_events
        and has_pass_scoring
    ):
        return "fully_processed"
    return "unknown"


def build_existing_manifest(outputs_root: Path, derived_root: Path) -> list[dict[str, Any]]:
    output_ids = immediate_dir_names(outputs_root)
    derived_ids = immediate_dir_names(derived_root)
    rows: list[dict[str, Any]] = []
    for clip_id in sorted(output_ids | derived_ids):
        output_dir = outputs_root / clip_id
        derived_dir = derived_root / clip_id
        exists_in_outputs = clip_id in output_ids
        exists_in_derived = clip_id in derived_ids
        has_player_teams = (output_dir / "teams" / "player_teams.csv").exists()
        has_possession = (output_dir / "possession" / "possession.csv").exists()
        has_temporal_frames = (derived_dir / "temporal_frames.csv").exists()
        has_unified_events = (derived_dir / "events" / "event_candidates_unified.csv").exists()
        has_pass_scoring = (derived_dir / "events" / "pass_candidates_scored.csv").exists()
        status = local_status(
            exists_in_outputs=exists_in_outputs,
            exists_in_derived=exists_in_derived,
            has_player_teams=has_player_teams,
            has_possession=has_possession,
            has_temporal_frames=has_temporal_frames,
            has_unified_events=has_unified_events,
            has_pass_scoring=has_pass_scoring,
        )
        notes = []
        if exists_in_outputs and not has_player_teams:
            notes.append("missing teams/player_teams.csv")
        if exists_in_outputs and not has_possession:
            notes.append("missing possession/possession.csv")
        rows.append(
            {
                "local_clip_id": clip_id,
                "exists_in_outputs": as_int(exists_in_outputs),
                "exists_in_derived": as_int(exists_in_derived),
                "local_status": status,
                "has_player_teams": as_int(has_player_teams),
                "has_possession": as_int(has_possession),
                "has_temporal_frames": as_int(has_temporal_frames),
                "has_unified_events": as_int(has_unified_events),
                "has_pass_scoring": as_int(has_pass_scoring),
                "notes": "; ".join(notes),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_inventory(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def normalize_identity(value: Any) -> str:
    text = str(value or "").strip().strip('"').strip("'")
    text = text.replace("\\", "/")
    text = re.sub(r"/+", "/", text)
    return text.strip("/").casefold()


def normalized_basename(value: Any) -> str:
    normalized = normalize_identity(value)
    if not normalized:
        return ""
    return normalized.rsplit("/", 1)[-1]


def local_identity_map(local_clip_ids: set[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for clip_id in sorted(local_clip_ids):
        normalized = normalize_identity(clip_id)
        if normalized:
            mapping[normalized] = clip_id
    return mapping


def candidate_identity_values(row: dict[str, str]) -> list[str]:
    values: list[str] = []
    for key, value in row.items():
        key_lower = key.casefold()
        if not value:
            continue
        if any(token in key_lower for token in ["clip", "video", "path", "file", "name", "slug", "id"]):
            values.append(value)
    return values


def exact_duplicate(row: dict[str, str], local_map: dict[str, str]) -> str:
    for value in candidate_identity_values(row):
        normalized = normalize_identity(value)
        basename = normalized_basename(value)
        for candidate in [normalized, basename]:
            if candidate in local_map:
                return local_map[candidate]
    return ""


def truthy_text(value: Any) -> bool:
    text = str(value or "").strip().casefold()
    return text in {"1", "true", "yes", "y", "ok", "available", "readable", "exists", "present"}


def field_contains(row: dict[str, str], key_tokens: list[str], value_tokens: list[str]) -> bool:
    for key, value in row.items():
        key_lower = key.casefold()
        if not any(token in key_lower for token in key_tokens):
            continue
        value_lower = str(value or "").casefold()
        if truthy_text(value_lower):
            return True
        if any(token in value_lower for token in value_tokens):
            return True
    return False


def is_readable_video(row: dict[str, str]) -> bool:
    return field_contains(row, ["readable", "video"], ["readable", "ok", "available", "exists"])


def is_ready_for_video_pilot(row: dict[str, str]) -> bool:
    return field_contains(row, ["status", "pilot"], ["ready_for_video_pilot"])


def has_nearby_annotations(row: dict[str, str]) -> bool:
    return field_contains(row, ["annotation", "label"], ["nearby", "available", "exists", ".json", ".xml", ".csv"])


def has_tracking_files(row: dict[str, str]) -> bool:
    return field_contains(row, ["tracking", "track"], ["available", "exists", ".json", ".csv", ".pkl"])


def parent_group(row: dict[str, str]) -> str:
    for key, value in row.items():
        if not value:
            continue
        key_lower = key.casefold()
        if any(token in key_lower for token in ["parent", "folder", "directory", "path"]):
            normalized = normalize_identity(value)
            if "/" in normalized:
                return normalized.rsplit("/", 1)[0]
            return normalized
    values = candidate_identity_values(row)
    if values:
        normalized = normalize_identity(values[0])
        return normalized.rsplit("/", 1)[0] if "/" in normalized else normalized
    return "unknown_parent"


def duration_bucket(row: dict[str, str]) -> str:
    for key, value in row.items():
        key_lower = key.casefold()
        if "duration" not in key_lower or not value:
            continue
        try:
            duration = float(str(value).strip())
        except ValueError:
            return normalize_identity(value)
        if duration < 300:
            return "short"
        if duration < 1800:
            return "medium"
        return "long"
    return "unknown_duration"


def half_bucket(row: dict[str, str]) -> str:
    for key, value in row.items():
        key_lower = key.casefold()
        if any(token in key_lower for token in ["half", "period"]):
            normalized = normalize_identity(value)
            if normalized:
                return normalized
    for value in candidate_identity_values(row):
        normalized = normalize_identity(value)
        if "h1" in normalized:
            return "h1"
        if "h2" in normalized:
            return "h2"
    return "unknown_half"


def priority_score(row: dict[str, str]) -> int:
    score = 0
    score += 100 if is_readable_video(row) else 0
    score += 50 if is_ready_for_video_pilot(row) else 0
    score += 20 if has_nearby_annotations(row) else 0
    score += 10 if has_tracking_files(row) else 0
    return score


def diversity_group(row: dict[str, str]) -> str:
    return "|".join([parent_group(row), duration_bucket(row), half_bucket(row)])


def selection_reason(row: dict[str, str]) -> str:
    reasons = []
    if is_readable_video(row):
        reasons.append("readable_video")
    if is_ready_for_video_pilot(row):
        reasons.append("ready_for_video_pilot")
    if has_nearby_annotations(row):
        reasons.append("nearby_annotation_files")
    if has_tracking_files(row):
        reasons.append("tracking_files_available")
    reasons.append(f"parent_group={parent_group(row)}")
    reasons.append(f"duration_bucket={duration_bucket(row)}")
    reasons.append(f"half={half_bucket(row)}")
    return ";".join(reasons)


def add_duplicate_fields(rows: list[dict[str, str]], local_clip_ids: set[str]) -> list[dict[str, str]]:
    local_map = local_identity_map(local_clip_ids)
    planned: list[dict[str, str]] = []
    for row in rows:
        output = dict(row)
        duplicate = exact_duplicate(row, local_map)
        has_identity = bool(candidate_identity_values(row))
        output["is_exact_local_duplicate"] = as_int(bool(duplicate))
        output["matching_local_clip_id"] = duplicate
        if duplicate:
            recommendation = "exclude_exact_duplicate"
        elif has_identity:
            recommendation = "candidate_for_new_download"
        else:
            recommendation = "manual_review_required"
        output["download_recommendation"] = recommendation
        output["selection_rank"] = ""
        output["selection_reason"] = ""
        planned.append(output)
    return planned


def ranked_new_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    candidates = [
        row
        for row in rows
        if row.get("download_recommendation") == "candidate_for_new_download"
    ]
    sorted_candidates = sorted(
        candidates,
        key=lambda row: (
            -priority_score(row),
            parent_group(row),
            duration_bucket(row),
            half_bucket(row),
            normalize_identity(candidate_identity_values(row)[0] if candidate_identity_values(row) else ""),
        ),
    )
    groups: dict[str, deque[dict[str, str]]] = defaultdict(deque)
    for row in sorted_candidates:
        groups[diversity_group(row)].append(row)
    ordered: list[dict[str, str]] = []
    while groups:
        for group in sorted(list(groups)):
            if not groups[group]:
                del groups[group]
                continue
            ordered.append(groups[group].popleft())
            if not groups.get(group):
                groups.pop(group, None)
    return ordered


def apply_selection(rows: list[dict[str, str]], target: int) -> None:
    selected = ranked_new_candidates(rows)[: max(0, target)]
    selected_ids = {id(row): index + 1 for index, row in enumerate(selected)}
    for row in rows:
        if id(row) in selected_ids:
            row["selection_rank"] = selected_ids[id(row)]
            row["selection_reason"] = selection_reason(row)


def summary_payload(
    manifest_rows: list[dict[str, Any]],
    inventory_rows: list[dict[str, str]] | None,
    target: int,
    no_inventory: bool,
) -> dict[str, Any]:
    local_count = len(manifest_rows)
    fully_processed = sum(row["local_status"] == "fully_processed" for row in manifest_rows)
    incomplete = sum(row["local_status"] != "fully_processed" for row in manifest_rows)
    inventory_rows = inventory_rows or []
    duplicates = sum(row.get("download_recommendation") == "exclude_exact_duplicate" for row in inventory_rows)
    manual_review = sum(row.get("download_recommendation") == "manual_review_required" for row in inventory_rows)
    selected = sum(str(row.get("selection_rank", "")).strip() != "" for row in inventory_rows)
    payload = {
        "build_timestamp": utc_now_iso(),
        "existing_local_clip_count": local_count,
        "fully_processed_local_clip_count": fully_processed,
        "incomplete_local_clip_count": incomplete,
        "soccernet_inventory_candidates_considered": len(inventory_rows),
        "exact_duplicates_excluded": duplicates,
        "manual_review_candidates": manual_review,
        "new_download_candidates_selected": selected,
        "requested_target": int(target),
        "final_selected_count": selected,
    }
    if no_inventory:
        payload["note"] = "No SoccerNet candidate inventory was supplied; no download candidates were guessed or created."
    return payload


def build_plan(args: argparse.Namespace) -> int:
    outputs_root = Path(args.outputs_root)
    derived_root = Path(args.derived_root)
    output_dir = Path(args.output_dir)
    manifest_rows = build_existing_manifest(outputs_root, derived_root)
    manifest_path = output_dir / "existing_local_clip_manifest.csv"
    write_csv(manifest_path, manifest_rows, MANIFEST_COLUMNS)

    summary_path = output_dir / "soccernet_new_clip_download_plan_summary.json"
    if not args.soccernet_inventory:
        write_json(
            summary_path,
            summary_payload(
                manifest_rows=manifest_rows,
                inventory_rows=None,
                target=args.target_new_clips,
                no_inventory=True,
            ),
        )
        print(f"Existing local clip manifest: {manifest_path}")
        print(f"Summary JSON: {summary_path}")
        print("No SoccerNet inventory supplied; no download candidates were created.")
        return 0

    inventory_path = Path(args.soccernet_inventory)
    if not inventory_path.exists():
        raise FileNotFoundError(f"SoccerNet inventory CSV not found: {inventory_path}")
    inventory_rows, inventory_columns = read_inventory(inventory_path)
    local_clip_ids = {str(row["local_clip_id"]) for row in manifest_rows}
    plan_rows = add_duplicate_fields(inventory_rows, local_clip_ids)
    apply_selection(plan_rows, args.target_new_clips)

    plan_columns = inventory_columns + [column for column in ADDED_PLAN_COLUMNS if column not in inventory_columns]
    plan_path = output_dir / "soccernet_new_clip_download_plan.csv"
    write_csv(plan_path, plan_rows, plan_columns)
    write_json(
        summary_path,
        summary_payload(
            manifest_rows=manifest_rows,
            inventory_rows=plan_rows,
            target=args.target_new_clips,
            no_inventory=False,
        ),
    )
    print(f"Existing local clip manifest: {manifest_path}")
    print(f"Download plan CSV: {plan_path}")
    print(f"Summary JSON: {summary_path}")
    print("No downloads were performed.")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return build_plan(args)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
