"""Download a reproducible SoccerNet Premier League pilot sample."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from src.pipeline.config import DEFAULT_CONFIG_PATH, get_config_value, load_config
from src.pipeline.manifest import ManifestRow, make_clip_id, write_manifest


VALID_RESOLUTIONS = {"224p", "720p"}
DEFAULT_PASSWORD = os.getenv("SOCCERNET_PASSWORD", "s0cc3rn3t")


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
        split_matches: list[str] = []
        for game_id in split_games:
            normalized = str(game_id).replace("\\", "/").strip()
            if normalized.startswith(competition_prefix.rstrip("/")):
                split_matches.append(normalized)
        print(f"\nSplit: {split} -> {len(split_matches)} available {competition_prefix.rstrip('/')} games")
        for game_id in split_matches[:10]:
            print("  AVAILABLE:", repr(game_id))
        games.extend((split, game_id) for game_id in split_matches)
    return sorted(games, key=lambda item: (item[0], item[1]))


def sample_epl_videos(
    games: list[tuple[str, str]],
    sample_size: int,
    seed: int,
) -> list[tuple[str, str, int]]:
    """Randomly select half-videos from eligible games."""
    candidates = [
        (split, game_id, half)
        for split, game_id in games
        for half in (1, 2)
    ]
    if len(candidates) < sample_size:
        raise RuntimeError(
            f"Only {len(candidates)} eligible EPL half-videos found; "
            f"cannot sample {sample_size}."
        )
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, sample_size), key=lambda item: (item[1], item[2]))


def manifest_rows_for_selection(
    selection: list[tuple[str, str, int]],
    soccernet_dir: Path,
    resolution: str,
) -> list[ManifestRow]:
    """Create manifest rows for sampled SoccerNet half-videos."""
    rows = []
    for split, game_id, half in selection:
        filename = build_video_filename(half, resolution)
        rows.append(
            ManifestRow(
                clip_id=make_clip_id(game_id, half, resolution),
                split=split,
                competition="england_epl",
                source="SoccerNet",
                game_id=game_id,
                half=half,
                video_path=soccernet_dir / game_id / filename,
                enabled=True,
                notes="pilot_random_sample",
            )
        )
    return rows


def download_selection(
    rows: list[ManifestRow],
    soccernet_dir: Path,
    resolution: str,
    password: str,
) -> None:
    """Download missing SoccerNet files for the selected rows."""
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError as error:
        raise SystemExit("SoccerNet is not installed. Run: pip install SoccerNet") from error

    downloader = SoccerNetDownloader(LocalDirectory=str(soccernet_dir))
    downloader.password = password

    for row in rows:
        filename = build_video_filename(row.half, resolution)
        if row.video_path.exists():
            print(f"Already exists: {row.video_path}")
            continue
        print(f"Downloading {row.game_id} half {row.half} ({resolution})")
        downloader.downloadGame(files=[filename], game=row.game_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download 25 random SoccerNet Premier League half-videos."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--soccernet-dir", type=Path)
    parser.add_argument("--resolution", choices=sorted(VALID_RESOLUTIONS))
    parser.add_argument("--sample-size", type=int, default=25)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", action="append", dest="splits")
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    soccernet_dir = args.soccernet_dir or Path(get_config_value(config, "data.soccernet_dir"))
    manifest_path = args.manifest or Path(get_config_value(config, "data.manifest_path"))
    resolution = args.resolution or str(get_config_value(config, "video.resolution", "720p"))
    sample_size = args.sample_size or int(get_config_value(config, "data.sample_size", 25))
    seed = args.seed if args.seed is not None else int(get_config_value(config, "data.random_seed", 42))
    splits = args.splits or list(get_config_value(config, "data.splits", ["valid"]))
    competition_prefix = str(get_config_value(config, "data.competition_prefix", "england_epl/"))

    soccernet_dir.mkdir(parents=True, exist_ok=True)
    games = get_epl_games(splits=splits, competition_prefix=competition_prefix)
    if not games:
        print(f"Error: no Premier League SoccerNet games found in splits: {splits}")
        return 1

    selection = sample_epl_videos(games=games, sample_size=sample_size, seed=seed)
    rows = manifest_rows_for_selection(selection, soccernet_dir, resolution)
    if not args.no_download:
        download_selection(rows, soccernet_dir, resolution, DEFAULT_PASSWORD)

    write_manifest(manifest_path, rows)
    print(f"Wrote manifest with {len(rows)} clip(s): {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
