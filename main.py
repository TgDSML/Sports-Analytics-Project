import argparse
from pathlib import Path

import cv2


def load_first_frame(video_path: str):
    """Load the first frame from a video file."""
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    success, frame = capture.read()
    capture.release()

    if not success:
        raise ValueError(f"Could not read first frame from: {video_path}")

    return frame


def main():
    parser = argparse.ArgumentParser(description="Sports analytics video starter")
    parser.add_argument(
        "--video",
        default="data/sample.mp4",
        help="Path to the input video file",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        print("Add a video file or run: python main.py --video path/to/video.mp4")
        return

    frame = load_first_frame(str(video_path))
    cv2.imshow("First Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
