"""Download only reviewed SoccerNet selections from a manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "inventory_clip_id",
    "video_path",
    "relative_video_path",
    "download_recommendation",
    "manual_include",
    "manual_exclude_reason",
}

LOG_COLUMNS = [
    "inventory_clip_id",
    "dataset_mode",
    "relative_video_path",
    "target_path",
    "status",
    "api_call",
    "message",
]

ERROR_COLUMNS = [
    "inventory_clip_id",
    "dataset_mode",
    "relative_video_path",
    "error_stage",
    "error_message",
]

VALID_MANUAL_INCLUDE = {"1", "true", "yes"}
VALID_TRACKING_SPLITS = {"train", "valid", "test", "challenge"}
INVENTORY_OUTPUT_DIR = Path("temporal_module") / "data" / "soccernet_inventory"
PASSWORD_VALUES: set[str] = set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download reviewed SoccerNet selections only.")
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--soccernet-root", required=True)
    parser.add_argument(
        "--dataset-mode",
        required=True,
        choices=["tracking", "broadcast_224p", "broadcast_720p"],
    )
    parser.add_argument("--video-password-env", default="SOCCERNET_VIDEO_PASSWORD")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-downloads", type=int, default=None)
    parser.add_argument("--allow-existing", action="store_true")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Selection manifest not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = list(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - set(fieldnames)
        if missing:
            raise ValueError(f"Selection manifest missing required column(s): {', '.join(sorted(missing))}")
        return [dict(row) for row in reader], fieldnames


def approved_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    selected: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for row in rows:
        include_value = str(row.get("manual_include", "")).strip().casefold()
        if include_value and include_value not in VALID_MANUAL_INCLUDE:
            errors.append(
                error_row(
                    row,
                    "manifest_filter",
                    "manual_include must be blank or one of: 1, true, yes",
                )
            )
            continue
        if str(row.get("download_recommendation", "")).strip() != "candidate_for_new_download":
            continue
        if str(row.get("manual_exclude_reason", "")).strip():
            continue
        selected.append(row)
    return selected, errors


def apply_max_downloads(rows: list[dict[str, str]], max_downloads: int | None) -> list[dict[str, str]]:
    if max_downloads is None:
        return rows
    return rows[: max(0, max_downloads)]


def safe_relative_path(value: str) -> Path:
    relative = Path(str(value).replace("\\", "/"))
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise ValueError(f"relative_video_path must be a safe relative path: {value}")
    return relative


def target_path(row: dict[str, str], soccernet_root: Path) -> Path:
    return soccernet_root / safe_relative_path(row.get("relative_video_path", ""))


def broadcast_filename(mode: str, relative_path: Path) -> str:
    filename = relative_path.name
    expected = "224p" if mode == "broadcast_224p" else "720p"
    if expected not in filename:
        half = infer_half(relative_path)
        filename = f"{half}_{expected}.mkv"
    return filename


def infer_half(path: Path) -> int:
    stem = path.stem.casefold()
    if stem.startswith("2") or "h2" in stem:
        return 2
    return 1


def game_id_from_relative_path(relative_path: Path) -> str:
    if len(relative_path.parts) < 2:
        raise ValueError(f"relative_video_path must include game folders and filename: {relative_path}")
    return Path(*relative_path.parts[:-1]).as_posix()


def require_video_password(env_name: str) -> str:
    password = os.getenv(env_name)
    if not password:
        raise RuntimeError(f"Missing required SoccerNet video password environment variable: {env_name}")
    PASSWORD_VALUES.add(password)
    return password


def import_downloader():
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError as error:
        raise RuntimeError("SoccerNet is not installed. Run: python -m pip install SoccerNet") from error
    return SoccerNetDownloader


def dry_run_print(rows: list[dict[str, str]], calls: list[str]) -> None:
    print("Selected rows:")
    for row in rows:
        print(f"- {row.get('inventory_clip_id', '')}: {row.get('relative_video_path', '')}")
    print("Intended SoccerNet API calls:")
    for call in calls:
        print(f"- {call}")


def api_call_for_row(args: argparse.Namespace, row: dict[str, str]) -> tuple[str, dict[str, str] | None]:
    if args.dataset_mode == "tracking":
        split = tracking_split(row)
        if not split:
            return "", error_row(
                row,
                "metadata",
                "tracking mode requires an exact tracking_split, split, or dataset_split column",
            )
        return f"SoccerNetDownloader.downloadDataTask(task='tracking', split={[split]!r})", None
    try:
        game_id, filename, _relative = api_call_for_broadcast(args.dataset_mode, row)
    except Exception as error:
        return "", error_row(row, "metadata", str(error))
    return f"SoccerNetDownloader.downloadGame(game={game_id!r}, files=[{filename!r}])", None


def api_call_for_broadcast(mode: str, row: dict[str, str]) -> tuple[str, str, Path]:
    relative = safe_relative_path(row.get("relative_video_path", ""))
    game_id = game_id_from_relative_path(relative)
    filename = broadcast_filename(mode, relative)
    call = f"SoccerNetDownloader.downloadGame(game={game_id!r}, files=[{filename!r}])"
    return game_id, filename, relative


def tracking_split(row: dict[str, str]) -> str:
    for key in ["tracking_split", "split", "dataset_split"]:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    try:
        first_part = safe_relative_path(row.get("relative_video_path", "")).parts[0]
    except Exception:
        first_part = ""
    if first_part in VALID_TRACKING_SPLITS:
        return first_part
    return ""


def tracking_target_path(row: dict[str, str], soccernet_root: Path) -> Path:
    split = tracking_split(row)
    return soccernet_root / "tracking" / f"{split}.zip" if split else soccernet_root / "tracking"


def api_calls_for_tracking(rows: list[dict[str, str]]) -> tuple[list[str], list[dict[str, str]]]:
    errors: list[dict[str, str]] = []
    splits: list[str] = []
    for row in rows:
        split = tracking_split(row)
        if not split:
            errors.append(
                error_row(
                    row,
                    "metadata",
                    "tracking mode requires an exact tracking_split, split, or dataset_split column",
                )
            )
            continue
        if split not in splits:
            splits.append(split)
    calls = [
        f"SoccerNetDownloader.downloadDataTask(task='tracking', split={splits!r})"
    ] if splits else []
    return calls, errors


def download_broadcast(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    soccernet_root: Path,
    password: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    SoccerNetDownloader = import_downloader()
    downloader = SoccerNetDownloader(LocalDirectory=str(soccernet_root))
    downloader.password = password
    logs: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for row in rows:
        try:
            game_id, filename, relative = api_call_for_broadcast(args.dataset_mode, row)
            intended_path = soccernet_root / game_id / filename
            if intended_path.exists() and not args.allow_existing:
                logs.append(log_row(row, args.dataset_mode, intended_path, "skipped_existing", f"downloadGame game={game_id!r} files=[{filename!r}]", "Target exists"))
                continue
            downloader.downloadGame(game=game_id, files=[filename], verbose=True)
            logs.append(log_row(row, args.dataset_mode, intended_path, "completed", f"downloadGame game={game_id!r} files=[{filename!r}]", ""))
        except Exception as error:
            errors.append(error_row(row, "download", str(error)))
    return logs, errors


def download_tracking(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    soccernet_root: Path,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    logs: list[dict[str, str]] = []
    downloadable_rows: list[dict[str, str]] = []
    for row in rows:
        try:
            exact_target = target_path(row, soccernet_root)
        except Exception:
            exact_target = tracking_target_path(row, soccernet_root)
        if exact_target.exists() and not args.allow_existing:
            logs.append(
                log_row(
                    row,
                    args.dataset_mode,
                    exact_target,
                    "skipped_existing",
                    "downloadDataTask task='tracking'",
                    "Manifest target exists",
                )
            )
            continue
        downloadable_rows.append(row)

    calls, errors = api_calls_for_tracking(downloadable_rows)
    if errors:
        return logs, errors
    splits = []
    for row in downloadable_rows:
        split = tracking_split(row)
        if split not in splits:
            splits.append(split)
    skipped_existing = []
    for split in splits:
        path = soccernet_root / "tracking" / f"{split}.zip"
        if path.exists() and not args.allow_existing:
            skipped_existing.append(split)
    if skipped_existing:
        for row in downloadable_rows:
            split = tracking_split(row)
            if split in skipped_existing:
                logs.append(log_row(row, args.dataset_mode, tracking_target_path(row, soccernet_root), "skipped_existing", calls[0], "Tracking split zip exists"))
        splits = [split for split in splits if split not in skipped_existing]
    if not splits:
        return logs, errors

    SoccerNetDownloader = import_downloader()
    downloader = SoccerNetDownloader(LocalDirectory=str(soccernet_root))
    try:
        downloader.downloadDataTask(task="tracking", split=splits, verbose=True)
        for row in downloadable_rows:
            if tracking_split(row) in splits:
                logs.append(log_row(row, args.dataset_mode, tracking_target_path(row, soccernet_root), "completed", f"downloadDataTask task='tracking' split={splits!r}", ""))
    except Exception as error:
        for row in downloadable_rows:
            if tracking_split(row) in splits:
                errors.append(error_row(row, "download", str(error)))
    return logs, errors


def log_row(
    row: dict[str, str],
    dataset_mode: str,
    path: Path,
    status: str,
    api_call: str,
    message: str,
) -> dict[str, str]:
    return {
        "inventory_clip_id": row.get("inventory_clip_id", ""),
        "dataset_mode": dataset_mode,
        "relative_video_path": row.get("relative_video_path", ""),
        "target_path": str(path),
        "status": status,
        "api_call": api_call,
        "message": message,
    }


def error_row(row: dict[str, str], stage: str, message: str) -> dict[str, str]:
    return {
        "inventory_clip_id": row.get("inventory_clip_id", ""),
        "dataset_mode": "",
        "relative_video_path": row.get("relative_video_path", ""),
        "error_stage": stage,
        "error_message": sanitize(message),
    }


def sanitize(message: str) -> str:
    text = str(message)
    for value in PASSWORD_VALUES:
        if value:
            text = text.replace(value, "[redacted]")
    default_password = os.getenv("SOCCERNET_VIDEO_PASSWORD", "")
    if default_password:
        text = text.replace(default_password, "[redacted]")
    return text


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def selected_api_calls(args: argparse.Namespace, rows: list[dict[str, str]]) -> tuple[list[str], list[dict[str, str]]]:
    if args.dataset_mode == "tracking":
        return api_calls_for_tracking(rows)
    calls = []
    errors = []
    for row in rows:
        try:
            game_id, filename, _relative = api_call_for_broadcast(args.dataset_mode, row)
            calls.append(f"SoccerNetDownloader.downloadGame(game={game_id!r}, files=[{filename!r}])")
        except Exception as error:
            errors.append(error_row(row, "metadata", str(error)))
    return calls, errors


def summary_payload(
    args: argparse.Namespace,
    requested_rows: int,
    selected_rows: int,
    logs: list[dict[str, str]],
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    skipped_existing = sum(row.get("status") == "skipped_existing" for row in logs)
    completed = sum(row.get("status") == "completed" for row in logs)
    return {
        "manifest_path": str(Path(args.selection_manifest)),
        "dataset_mode": args.dataset_mode,
        "dry_run": bool(args.dry_run),
        "requested_rows": int(requested_rows),
        "selected_rows": int(selected_rows),
        "skipped_existing_rows": int(skipped_existing),
        "completed_downloads": int(completed),
        "failed_downloads": int(len(errors)),
        "timestamp": utc_now_iso(),
    }


def run(args: argparse.Namespace) -> int:
    manifest_path = Path(args.selection_manifest)
    soccernet_root = Path(args.soccernet_root)
    rows, _fieldnames = read_manifest(manifest_path)
    approved, filter_errors = approved_rows(rows)
    selected = apply_max_downloads(approved, args.max_downloads)
    logs: list[dict[str, str]] = []
    errors: list[dict[str, str]] = list(filter_errors)

    calls, call_errors = selected_api_calls(args, selected)
    errors.extend(call_errors)

    if args.dry_run:
        dry_run_print(selected, calls)
        for row in selected:
            try:
                path = target_path(row, soccernet_root)
            except Exception:
                path = soccernet_root
            api_call, row_error = api_call_for_row(args, row)
            if row_error is not None:
                errors.append(row_error)
                continue
            logs.append(log_row(row, args.dataset_mode, path, "dry_run_selected", api_call, "No download performed"))
    elif not errors:
        soccernet_root.mkdir(parents=True, exist_ok=True)
        if args.dataset_mode == "tracking":
            stage_logs, stage_errors = download_tracking(selected, args, soccernet_root)
        else:
            try:
                password = require_video_password(args.video_password_env)
            except Exception as error:
                errors.append(error_row({}, "password", str(error)))
                password = ""
            if errors:
                stage_logs, stage_errors = [], []
            else:
                stage_logs, stage_errors = download_broadcast(selected, args, soccernet_root, password)
        logs.extend(stage_logs)
        errors.extend(stage_errors)

    output_dir = INVENTORY_OUTPUT_DIR
    errors_path = output_dir / "soccernet_selected_download_errors.csv"
    write_csv(output_dir / "soccernet_selected_download_log.csv", logs, LOG_COLUMNS)
    if errors:
        write_csv(errors_path, errors, ERROR_COLUMNS)
    elif errors_path.exists():
        errors_path.unlink()
    write_json(
        output_dir / "soccernet_selected_download_summary.json",
        summary_payload(args, requested_rows=len(rows), selected_rows=len(selected), logs=logs, errors=errors),
    )

    print(f"Selected rows: {len(selected)}")
    print(f"Completed downloads: {sum(row.get('status') == 'completed' for row in logs)}")
    print(f"Failed downloads: {len(errors)}")
    print(f"Logs written under: {output_dir}")
    return 0 if not errors else 1


def main() -> int:
    args = parse_args()
    try:
        return run(args)
    except Exception as error:
        print(f"Error: {sanitize(str(error))}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
