# sports-analytics-project

## Project Description

This project is a simple computer vision baseline for sports analytics. It runs a pretrained Ultralytics YOLO model on local sports video, keeps player detections, and writes an annotated output video with resolution-aware bounding boxes and labels.

The current implementation is intentionally small and reproducible:

- YOLO player detection
- Optional random video selection from a local SoccerNet sample
- Annotated MP4 output
- Placeholder modules for future tracking and analytics work

## Setup Instructions

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

You can also run the setup script for your platform:

Windows:

```bat
setup.bat
```

Linux/macOS:

```bash
chmod +x setup.sh
./setup.sh
```

## Download SoccerNet Sample

Download a small low-resolution SoccerNet sample:

```bash
python scripts/download_soccernet_video.py
```

The script downloads `1_224p.mkv` and `2_224p.mkv` for one validation game into `data/SoccerNet/`. It uses the official SoccerNet API and does not download the full dataset.

If you need to override the default SoccerNet password, set `SOCCERNET_PASSWORD` before running the script.

Windows PowerShell:

```powershell
$env:SOCCERNET_PASSWORD="your-password"
python scripts/download_soccernet_video.py
```

Linux/macOS:

```bash
SOCCERNET_PASSWORD="your-password" python scripts/download_soccernet_video.py
```

## Run YOLO Baseline

Run YOLO on a sample video:

```bash
python main.py --video data/sample_30s.mp4 --output outputs/yolov8_baseline.mp4 --model yolov8n.pt
```

Run YOLO on one random local SoccerNet video:

```bash
python main.py --random-soccernet --soccernet-dir data/SoccerNet --output outputs/yolo_random_soccernet_baseline.mp4 --model yolov8n.pt
```

Optionally display annotated frames while processing:

```bash
python main.py --random-soccernet --soccernet-dir data/SoccerNet --output outputs/yolo_random_soccernet_baseline.mp4 --model yolov8n.pt --show
```

## Example Commands

Compile-check the project:

```bash
python -m compileall main.py src
```

Run with a higher confidence threshold:

```bash
python main.py --video data/sample_30s.mp4 --output outputs/yolov8_conf_025.mp4 --model yolov8n.pt --conf 0.25
```

Run with a smaller inference size for faster CPU processing:

```bash
python main.py --video data/sample_30s.mp4 --output outputs/yolov8_fast.mp4 --model yolov8n.pt --imgsz 416
```

## Project Structure

```text
sports-analytics-project/
|-- data/              # Local input videos and datasets, ignored by Git
|-- notebooks/         # Experiments and exploratory analysis
|-- outputs/           # Generated videos and reports, ignored by Git
|-- scripts/           # Utility scripts
|-- src/
|   |-- analytics/     # Sports metrics and analysis code
|   |-- detection/     # YOLO detection and visualization code
|   |-- tracking/      # Tracking integration placeholder
|   `-- utils/         # Shared helper functions
|-- main.py            # YOLO video baseline entry point
|-- requirements.txt   # Python dependencies
|-- setup.bat          # Windows setup script
|-- setup.sh           # Linux/macOS setup script
`-- README.md
```

## Repository Hygiene

- `data/` and `outputs/` are ignored except for `.gitkeep` placeholders.
- `.venv/`, Python caches, generated model weights, and local FFmpeg artifacts are ignored.
- Keep videos, datasets, generated outputs, and model weights out of Git.
