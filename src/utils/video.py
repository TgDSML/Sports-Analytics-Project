from pathlib import Path


def validate_video_path(video_path):
    """Return True when the video path exists and is a file."""
    return Path(video_path).is_file()
