"""Prepare a manifest from SoccerNet EPL videos already present on disk."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.data_tools.download_soccernet_epl import VALID_RESOLUTIONS, build_video_filename
from src.pipeline.config import DEFAULT_CONFIG_PATH, get_config_value, load_config
from src.pipeline.manifest import ManifestRow, make_clip_id, write_manifest


def discover_local_epl_videos(
    soccernet_dir: Path,
    competition_prefix: str,
    resolution: str,
    split: str,
) -> list[ManifestRow]:
    """Return manifest rows for local EPL half-videos."""
    rows = []
    for half in (1, 2):
        filename = build_video_filename(half, resolution)
        for video_path in sorted(soccernet_dir.rglob(filename)):
            if not video_path.is_file():
                continue
            try:
                game_id = video_path.relative_to(soccernet_dir).parent.as_posix()
            except ValueError:
                continue
            if not game_id.startswith(competition_prefix):
                continue
            rows.append(
                ManifestRow(
                    clip_id=make_clip_id(game_id, half, resolution),
                    split=split,
                    competition="england_epl",
                    source="SoccerNet",
                    game_id=game_id,
                    half=half,
                    video_path=video_path,
                    enabled=True,
                    notes="local_manifest",
                )
            )
    return sorted(rows, key=lambda row: row.clip_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create data/manifests/dataset_manifest.csv from local SoccerNet EPL files."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--soccernet-dir", type=Path)
    parser.add_argument("--resolution", choices=sorted(VALID_RESOLUTIONS))
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", default="valid")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    soccernet_dir = args.soccernet_dir or Path(get_config_value(config, "data.soccernet_dir"))
    manifest_path = args.manifest or Path(get_config_value(config, "data.manifest_path"))
    resolution = args.resolution or str(get_config_value(config, "video.resolution", "720p"))
    sample_size = args.sample_size or int(get_config_value(config, "data.sample_size", 10))
    seed = args.seed if args.seed is not None else int(get_config_value(config, "data.random_seed", 42))
    competition_prefix = str(get_config_value(config, "data.competition_prefix", "england_epl/"))

    rows = discover_local_epl_videos(
        soccernet_dir=soccernet_dir,
        competition_prefix=competition_prefix,
        resolution=resolution,
        split=args.split,
    )
    if len(rows) < sample_size:
        print(
            f"Error: found {len(rows)} local EPL {resolution} video(s), "
            f"need {sample_size}."
        )
        return 1

    rng = random.Random(seed)
    rows = sorted(rng.sample(rows, sample_size), key=lambda row: row.clip_id)
    write_manifest(manifest_path, rows)
    print(f"Wrote manifest with {len(rows)} local clip(s): {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
