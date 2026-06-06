"""Download a small SoccerNet validation sample."""

import argparse
import os
from pathlib import Path

try:
    from SoccerNet.Downloader import SoccerNetDownloader
    from SoccerNet.utils import getListGames
except ImportError as error:
    raise SystemExit("SoccerNet is not installed. Run: pip install SoccerNet") from error


LOCAL_DIRECTORY = Path("data/SoccerNet")
VALID_RESOLUTIONS = {"224p", "720p"}
SOCCERNET_PASSWORD = os.getenv("SOCCERNET_PASSWORD", "s0cc3rn3t")
SPLIT = ["valid"]
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}


def find_downloaded_videos(root_dir: Path) -> list[Path]:
    """Return downloaded videos under the SoccerNet sample directory."""
    return sorted(
        path
        for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def build_file_list(resolution: str) -> list[str]:
    """Return first-half and second-half SoccerNet files for a resolution."""
    if resolution not in VALID_RESOLUTIONS:
        valid_values = ", ".join(sorted(VALID_RESOLUTIONS))
        raise ValueError(f"Resolution must be one of: {valid_values}")
    return [f"1_{resolution}.mkv", f"2_{resolution}.mkv"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download one SoccerNet validation game sample."
    )
    parser.add_argument(
        "--resolution",
        choices=sorted(VALID_RESOLUTIONS),
        default="224p",
        help="Video resolution to download. Defaults to 224p.",
    )
    return parser.parse_args()


def main() -> None:
    """Download one validation game at the requested resolution if needed."""
    args = parse_args()
    files = build_file_list(args.resolution)
    LOCAL_DIRECTORY.mkdir(parents=True, exist_ok=True)

    downloader = SoccerNetDownloader(LocalDirectory=str(LOCAL_DIRECTORY))
    downloader.password = SOCCERNET_PASSWORD

    games = getListGames(split=SPLIT[0])
    if not games:
        print(f"Error: no SoccerNet games found for split: {SPLIT}")
        return

    game = games[0]
    expected_files = [LOCAL_DIRECTORY / game / filename for filename in files]

    print(f"Starting {args.resolution} download...")
    if all(path.exists() for path in expected_files):
        print("Sample videos already exist, skipping download.")
    else:
        downloader.downloadGame(files=files, game=game)
    print("Download complete")

    videos = find_downloaded_videos(LOCAL_DIRECTORY)
    if not videos:
        print(f"Error: no videos found in {LOCAL_DIRECTORY} after download.")
        return

    print(f"Found {len(videos)} video(s).")
    print("First video paths:")
    for video_path in videos[:5]:
        print(video_path)


if __name__ == "__main__":
    main()
