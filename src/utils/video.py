"""Video file helpers."""

import random
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}


def validate_video_path(video_path: str | Path) -> bool:
    """Return True when the video path exists and is a file."""
    return Path(video_path).is_file()


def find_random_video(root_dir: str | Path) -> Path:
    """Find one random supported video file under a directory."""
    root_path = Path(root_dir)

    if not root_path.exists():
        raise FileNotFoundError(f"SoccerNet directory not found: {root_path}")

    if not root_path.is_dir():
        raise NotADirectoryError(f"SoccerNet path is not a directory: {root_path}")

    video_files = sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        extensions = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise FileNotFoundError(
            f"No video files found in {root_path}. Expected extensions: {extensions}"
        )

    return random.choice(video_files)
