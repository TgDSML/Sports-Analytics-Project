"""Build a reproducible SoccerNet Premier League new-clip selection manifest."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.pipeline.config import DEFAULT_CONFIG_PATH, get_config_value, load_config
from src.pipeline.manifest import make_clip_id


VALID_RESOLUTIONS = {"224p", "720p"}
MANIFEST_COLUMNS = [
    "inventory_clip_id",
    "video_path",
    "relative_video_path",
    "download_recommendation",
    "manual_include",
    "manual_exclude_reason",
    "clip_id",
    "split",
    "competition",
    "source",
    "game_id",
    "half",
    "resolution",
    "enabled",
    "selection_seed",
    "selection_reason",
    "is_new_selection",
    "excluded_existing_count",
    "notes",
]
AUDIT_COLUMNS = [
    "game_id",
    "half",
    "resolution",
    "candidate_status",
    "exclusion_source",
    "expected_video_path",
    "is_selected",
    "reason",
]


@dataclass(frozen=True)
class HalfCandidate:
    split: str
    game_id: str
    half: int
    resolution: str
    expected_video_path: Path

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.game_id, self.half, self.resolution)

    @property
    def filename(self) -> str:
        return build_video_filename(self.half, self.resolution)

    @property
    def relative_video_path(self) -> str:
        return Path(self.game_id, self.filename).as_posix()

    @property
    def clip_id(self) -> str:
        return make_clip_id(self.game_id, self.half, self.resolution)


def build_video_filename(half: int, resolution: str) -> str:
    """Return the SoccerNet video filename for a half and resolution."""
    if half not in {1, 2}:
        raise ValueError("half must be 1 or 2")
    if resolution not in VALID_RESOLUTIONS:
        raise ValueError(f"resolution must be one of: {', '.join(sorted(VALID_RESOLUTIONS))}")
    return f"{half}_{resolution}.mkv"


def get_epl_games(splits: list[str], competition_prefix: str) -> list[tuple[str, str]]:
    try:
        from SoccerNet.utils import getListGames
    except ImportError as error:
        raise SystemExit("SoccerNet is not installed. Run: pip install SoccerNet") from error

    games: list[tuple[str, str]] = []
    for split in splits:
        split_games = getListGames(split=split)
        print(f"\nSplit: {split} -> {len(split_games)} games")
        for game_id in split_games[:10]:
            print("  SAMPLE:", repr(game_id))
        for game_id in split_games:
            normalized = str(game_id).replace("\\", "/").strip()
            if normalized.startswith(competition_prefix.rstrip("/")):
                games.append((split, normalized))
    return sorted(games, key=lambda item: (item[0], item[1]))


def epl_half_candidates(
    games: list[tuple[str, str]],
    soccernet_dir: Path,
    resolution: str,
) -> list[HalfCandidate]:
    return [
        HalfCandidate(
            split=split,
            game_id=game_id,
            half=half,
            resolution=resolution,
            expected_video_path=soccernet_dir / game_id / build_video_filename(half, resolution),
        )
        for split, game_id in games
        for half in (1, 2)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a duplicate-aware SoccerNet Premier League selection manifest."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--soccernet-dir", type=Path)
    parser.add_argument("--resolution", choices=sorted(VALID_RESOLUTIONS))
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--exclude-manifest", type=Path)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--derived-root", type=Path, default=Path("temporal_module") / "data" / "derived")
    parser.add_argument("--exclude-existing-local", dest="exclude_existing_local", action="store_true", default=True)
    parser.add_argument("--no-exclude-existing-local", dest="exclude_existing_local", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def normalize_path(value: Any) -> str:
    return str(value or "").strip().strip('"').strip("'").replace("\\", "/").strip("/")


def parse_half(value: Any) -> int | None:
    try:
        half = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return half if half in {1, 2} else None


def parse_resolution(value: Any) -> str:
    text = str(value or "").strip()
    return text if text in VALID_RESOLUTIONS else ""


def parse_key_from_video_path(video_path: Any, resolution_hint: str = "") -> tuple[str, int, str] | None:
    normalized = normalize_path(video_path)
    if not normalized:
        return None
    path = Path(normalized)
    filename = path.name
    match = re.fullmatch(r"([12])_(224p|720p)\.mkv", filename)
    if not match:
        return None
    half = int(match.group(1))
    resolution = match.group(2) or resolution_hint
    if resolution_hint and resolution != resolution_hint:
        return None
    parent = path.parent.as_posix()
    if parent in {"", "."}:
        return None
    return (parent, half, resolution)


def read_exclude_manifest(path: Path, resolution: str) -> tuple[set[tuple[str, int, str]], int]:
    if not path.exists():
        raise FileNotFoundError(f"Exclude manifest not found: {path}")
    excluded: set[tuple[str, int, str]] = set()
    unresolved = 0
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            game_id = normalize_path(row.get("game_id", ""))
            half = parse_half(row.get("half", ""))
            row_resolution = parse_resolution(row.get("resolution", "")) or resolution
            if game_id and half is not None and row_resolution in VALID_RESOLUTIONS:
                excluded.add((game_id, half, row_resolution))
                continue
            parsed = parse_key_from_video_path(row.get("video_path", ""), resolution)
            if parsed is not None:
                excluded.add(parsed)
            else:
                unresolved += 1
    return excluded, unresolved


def existing_destination_keys(candidates: list[HalfCandidate]) -> set[tuple[str, int, str]]:
    return {candidate.key for candidate in candidates if candidate.expected_video_path.exists()}


def immediate_dir_names(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def local_identity_keys(root: Path, resolution: str) -> tuple[set[tuple[str, int, str]], list[str]]:
    keys: set[tuple[str, int, str]] = set()
    unresolved: list[str] = []
    for name in immediate_dir_names(root):
        parsed = parse_local_clip_id(name, resolution)
        if parsed is None:
            unresolved.append(name)
        else:
            keys.add(parsed)
    return keys, unresolved


def parse_local_clip_id(clip_id: str, resolution_hint: str) -> tuple[str, int, str] | None:
    _ = resolution_hint
    # Existing local clip folders are filesystem-safe IDs. They collapse spaces
    # and hyphens to underscores, so they are not reversible to exact SoccerNet
    # game IDs without an external manifest.
    return None


def sample_epl_videos(
    candidates: list[HalfCandidate],
    sample_size: int,
    seed: int,
) -> list[HalfCandidate]:
    """Randomly select half-videos from eligible new candidates."""
    if len(candidates) < sample_size:
        raise RuntimeError(
            f"Only {len(candidates)} eligible new EPL half-videos found; "
            f"cannot sample {sample_size}."
        )
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, sample_size), key=lambda item: (item.game_id, item.half))


def manifest_rows_for_selection(
    selection: list[HalfCandidate],
    soccernet_dir: Path,
    seed: int,
    excluded_existing_count: int,
) -> list[dict[str, Any]]:
    rows = []
    for candidate in selection:
        rows.append(
            {
                "inventory_clip_id": candidate.clip_id,
                "video_path": str(candidate.expected_video_path),
                "relative_video_path": candidate.relative_video_path,
                "download_recommendation": "candidate_for_new_download",
                "manual_include": "",
                "manual_exclude_reason": "",
                "clip_id": candidate.clip_id,
                "split": candidate.split,
                "competition": "england_epl",
                "source": "SoccerNet",
                "game_id": candidate.game_id,
                "half": candidate.half,
                "resolution": candidate.resolution,
                "enabled": "true",
                "selection_seed": seed,
                "selection_reason": "random_epl_half_excluding_exact_known_existing",
                "is_new_selection": 1,
                "excluded_existing_count": excluded_existing_count,
                "notes": f"manifest_for_selected_downloader;root={soccernet_dir}",
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


def audit_row(
    candidate: HalfCandidate,
    status: str,
    source: str,
    selected: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "game_id": candidate.game_id,
        "half": candidate.half,
        "resolution": candidate.resolution,
        "candidate_status": status,
        "exclusion_source": source,
        "expected_video_path": str(candidate.expected_video_path),
        "is_selected": 1 if selected else 0,
        "reason": reason,
    }


def unresolved_audit_row(name: str, source: str, resolution: str) -> dict[str, Any]:
    return {
        "game_id": "",
        "half": "",
        "resolution": resolution,
        "candidate_status": "unresolved_local_identity",
        "exclusion_source": source,
        "expected_video_path": "",
        "is_selected": 0,
        "reason": name,
    }


def build_selection(
    candidates: list[HalfCandidate],
    selected: list[HalfCandidate],
    manifest_excluded: set[tuple[str, int, str]],
    destination_excluded: set[tuple[str, int, str]],
    local_excluded: set[tuple[str, int, str]],
    unresolved_local: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    selected_keys = {candidate.key for candidate in selected}
    rows = []
    for candidate in candidates:
        if candidate.key in selected_keys:
            rows.append(audit_row(candidate, "selected_new", "", True, "selected by seeded random sample"))
        elif candidate.key in manifest_excluded:
            rows.append(audit_row(candidate, "excluded_manifest", "exclude_manifest", False, "exact game_id+half+resolution or exact parsed video_path"))
        elif candidate.key in destination_excluded:
            rows.append(audit_row(candidate, "excluded_existing_destination", "soccernet_dir", False, "expected destination file exists"))
        elif candidate.key in local_excluded:
            rows.append(audit_row(candidate, "excluded_existing_destination", "local_clip_directory", False, "local clip directory safely mapped to exact game_id+half+resolution"))
        else:
            rows.append(audit_row(candidate, "eligible_not_selected", "", False, "eligible but not selected by seeded sample"))
    for source, name in unresolved_local:
        rows.append(unresolved_audit_row(name, source, candidates[0].resolution if candidates else ""))
    return rows


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    soccernet_dir = args.soccernet_dir or Path(get_config_value(config, "data.soccernet_dir"))
    manifest_path = args.manifest or Path(get_config_value(config, "data.manifest_path"))
    resolution = args.resolution or str(get_config_value(config, "video.resolution", "720p"))
    sample_size = args.sample_size or int(get_config_value(config, "data.sample_size", 10))
    seed = args.seed if args.seed is not None else int(get_config_value(config, "data.random_seed", 42))
    splits = args.splits or list(get_config_value(config, "data.splits", ["valid"]))
    competition_prefix = str(get_config_value(config, "data.competition_prefix", "england_epl/"))

    games = get_epl_games(splits=splits, competition_prefix=competition_prefix)
    if not games:
        print(f"Error: no Premier League SoccerNet games found in splits: {splits}")
        return 1

    candidates = epl_half_candidates(games=games, soccernet_dir=soccernet_dir, resolution=resolution)
    manifest_excluded: set[tuple[str, int, str]] = set()
    unresolved_manifest_rows = 0
    if args.exclude_manifest:
        manifest_excluded, unresolved_manifest_rows = read_exclude_manifest(args.exclude_manifest, resolution)

    destination_excluded = existing_destination_keys(candidates)
    local_excluded: set[tuple[str, int, str]] = set()
    unresolved_local: list[tuple[str, str]] = []
    if args.exclude_existing_local:
        output_keys, output_unresolved = local_identity_keys(args.outputs_root, resolution)
        derived_keys, derived_unresolved = local_identity_keys(args.derived_root, resolution)
        local_excluded = output_keys | derived_keys
        unresolved_local.extend(("outputs_root", name) for name in output_unresolved)
        unresolved_local.extend(("derived_root", name) for name in derived_unresolved)

    excluded_keys = manifest_excluded | destination_excluded | local_excluded
    eligible = [candidate for candidate in candidates if candidate.key not in excluded_keys]
    selected = sample_epl_videos(eligible, sample_size, seed)
    excluded_existing_count = len({key for key in excluded_keys if key in {candidate.key for candidate in candidates}})
    manifest_rows = manifest_rows_for_selection(selected, soccernet_dir, seed, excluded_existing_count)
    audit_rows = build_selection(
        candidates=candidates,
        selected=selected,
        manifest_excluded=manifest_excluded,
        destination_excluded=destination_excluded,
        local_excluded=local_excluded,
        unresolved_local=unresolved_local,
    )

    audit_path = manifest_path.with_name(f"{manifest_path.stem}_selection_audit.csv")
    summary_path = manifest_path.with_name(f"{manifest_path.stem}_selection_summary.json")
    write_csv(manifest_path, manifest_rows, MANIFEST_COLUMNS)
    write_csv(audit_path, audit_rows, AUDIT_COLUMNS)
    write_json(
        summary_path,
        {
            "requested_sample_size": sample_size,
            "actual_selected_count": len(selected),
            "seed": seed,
            "epl_candidate_pool_size": len(candidates),
            "excluded_by_manifest_count": len(manifest_excluded),
            "unresolved_manifest_row_count": unresolved_manifest_rows,
            "excluded_by_existing_destination_count": len(destination_excluded),
            "unresolved_local_identity_count": len(unresolved_local),
            "eligible_remaining_count": len(eligible),
            "selected_count": len(selected),
            "dry_run": bool(args.dry_run),
            "timestamp": utc_now_iso(),
            "downloads_performed": 0,
        },
    )

    print(f"Wrote new-selection manifest: {manifest_path}")
    print(f"Wrote selection audit: {audit_path}")
    print(f"Wrote selection summary: {summary_path}")
    print(f"Selected {len(selected)} new EPL half-video(s).")
    print("No downloads were performed. Use temporal_module/scripts/download_soccernet_selected_clips.py after review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
