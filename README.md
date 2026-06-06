# sports-analytics-project

## Project Description

This project is a simple computer vision baseline for sports analytics. It runs a pretrained Ultralytics YOLO model on local sports video, keeps player detections, and writes an annotated output video with resolution-aware bounding boxes and labels.

The current implementation is intentionally small and reproducible:

- YOLO player detection
- Optional random video selection from a local SoccerNet sample
- Annotated MP4 output
- ByteTrack tracking by default, with centroid tracking available as a fallback
- Tracks CSV export
- Player movement heatmap generation
- Player trajectory visualization and movement statistics
- Improved jersey-color team classification
- Role-aware goalkeeper/referee heuristics

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

Download a small 224p SoccerNet sample:

```bash
python scripts/download_soccernet_video.py
```

The default is equivalent to:

```bash
python scripts/download_soccernet_video.py --resolution 224p
```

Download the 720p version of the same small validation sample:

```bash
python scripts/download_soccernet_video.py --resolution 720p
```

For `224p`, the script downloads `1_224p.mkv` and `2_224p.mkv`. For `720p`, it downloads `1_720p.mkv` and `2_720p.mkv`. It downloads only one `valid` split game into `data/SoccerNet/` and does not download the full dataset.

If you need to override the default SoccerNet password, set `SOCCERNET_PASSWORD` before running the script.

Windows PowerShell:

```powershell
$env:SOCCERNET_PASSWORD="your-password"
python scripts/download_soccernet_video.py --resolution 224p
```

Linux/macOS:

```bash
SOCCERNET_PASSWORD="your-password" python scripts/download_soccernet_video.py --resolution 224p
```

## Run YOLO Baseline

Run YOLO on a sample video:

```bash
python main.py --video data/sample_30s.mp4 --output outputs/yolov8_baseline.mp4 --model yolov8n.pt
```

Run YOLO and export detections to CSV:

```bash
python main.py --video data/sample_30s.mp4 --output outputs/yolo_30s_baseline.mp4 --model yolov8n.pt --conf 0.15 --imgsz 640 --csv-output outputs/detections_30s.csv
```

Run YOLO with centroid tracking and export detections plus tracks:

```bash
python main.py --video data/sample_30s.mp4 --output outputs/tracked_30s.mp4 --model yolov8n.pt --conf 0.15 --imgsz 640 --csv-output outputs/detections_30s.csv --enable-tracking --tracker-type centroid --tracks-csv outputs/tracks_30s.csv
```

Run YOLO with the improved centroid tracker:

```bash
python main.py --video data/sample_30s.mp4 --output outputs/tracked_30s_improved.mp4 --model yolov8n.pt --conf 0.2 --imgsz 640 --enable-tracking --tracker-type centroid --tracks-csv outputs/tracks_30s_improved.csv --max-distance 120 --max-missing 30 --smoothing 0.7 --min-box-area 100
```

## Tracking Backends

Tracking is enabled with `--enable-tracking`. The default backend is `bytetrack`.

- `bytetrack`: preferred backend using Ultralytics ByteTrack for more stable player IDs.
- `centroid`: simple baseline tracker kept as a fallback and for comparison.

Example ByteTrack run:

```bash
python main.py --video data/sample_30s.mp4 --output outputs/tracked_30s.mp4 --model yolov8n.pt --conf 0.2 --imgsz 640 --enable-tracking --tracker-type bytetrack --tracks-csv outputs/tracks_30s.csv
```

Run YOLO with tracking and generate a single-player heatmap:

```bash
python main.py --video data/sample_30s.mp4 --output outputs/tracked_30s.mp4 --model yolov8n.pt --enable-tracking --tracks-csv outputs/tracks_30s.csv --generate-heatmap --heatmap-track-id 3 --heatmap-output outputs/heatmap_player3.png
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

## Analytics Outputs

Run YOLO baseline and export detections to CSV:

```bash
python main.py --video data/sample_30s.mp4 --output outputs/yolo_30s_baseline.mp4 --model yolov8n.pt --conf 0.15 --imgsz 640 --csv-output outputs/detections_30s.csv
```

Run tracking and export tracks to CSV:

```bash
python main.py --video data/sample_30s.mp4 --output outputs/tracked_30s.mp4 --model yolov8n.pt --enable-tracking --tracks-csv outputs/tracks_30s.csv
```

Generate a heatmap during the tracking pipeline:

```bash
python main.py --video data/sample_30s.mp4 --output outputs/tracked_30s.mp4 --model yolov8n.pt --enable-tracking --tracks-csv outputs/tracks_30s.csv --generate-heatmap --heatmap-output outputs/heatmap_all.png
```

Generate an all-player heatmap from an existing tracks CSV without rerunning YOLO:

```bash
python -m src.analytics.heatmap --tracks-csv outputs/tracks_30s.csv --output outputs/heatmap_all_from_csv.png
```

Generate a single-player heatmap from an existing tracks CSV:

```bash
python -m src.analytics.heatmap --tracks-csv outputs/tracks_30s.csv --track-id 3 --output outputs/heatmap_player3_from_csv.png
```

Generate all-player trajectories from an existing tracks CSV:

```bash
python -m src.analytics.trajectories --tracks-csv outputs/tracks_30s.csv --output outputs/trajectories_all.png
```

Generate a single-player trajectory from an existing tracks CSV:

```bash
python -m src.analytics.trajectories --tracks-csv outputs/tracks_30s.csv --track-id 3 --output outputs/trajectory_player3.png
```

Generate player movement statistics from an existing tracks CSV:

```bash
python -m src.analytics.player_stats --tracks-csv outputs/tracks_30s.csv --output outputs/player_stats_30s.csv
```

Generate raw, readable, Markdown, and Excel player statistics:

```bash
python -m src.analytics.player_stats --tracks-csv outputs/tracks_30s.csv --output outputs/player_stats_30s.csv --readable-output outputs/player_stats_30s_readable.csv --markdown-output outputs/player_stats_30s.md --excel-output outputs/player_stats_30s.xlsx
```

Generate trajectories and player statistics during the tracking pipeline:

```bash
python main.py --video data/sample_30s.mp4 --output outputs/tracked_30s.mp4 --enable-tracking --tracks-csv outputs/tracks_30s.csv --generate-trajectories --trajectory-output outputs/trajectories_all.png --generate-player-stats --player-stats-output outputs/player_stats_30s.csv
```

## Team Classification

The team classifier assigns tracked players to `Team A`, `Team B`, or `Unknown` using an explainable jersey-color workflow. It samples only the central torso area of each tracked box, rejects grass-heavy or low-quality crops, aggregates valid jersey samples per `track_id`, and clusters robust track-level LAB colors.

Role-aware classification can also separate outfield players, goalkeeper candidates, referee candidates, and unknown tracks. It uses extra color clusters plus track position and movement summaries. By default, team heatmaps include only outfield players so goalkeeper/referee movement does not distort team movement maps.

This is a baseline only. It can be wrong when detections are noisy, players are occluded, lighting changes, kits have similar colors, or tracks are fragmented. Goalkeeper/referee detection is heuristic and deliberately avoids forcing labels when evidence is weak. It does not use homography, possession, event detection, roster identity, or a learned re-identification model.

Run color-based team classification, draw a team-colored tracking video, and generate team heatmaps:

```bash
python -m src.analytics.team_classifier --video data/sample_30s.mp4 --tracks-csv outputs/tracks_30s.csv --output outputs/player_teams_30s.csv --team-video-output outputs/team_tracked_30s.mp4 --team-a-heatmap outputs/heatmap_team_a.png --team-b-heatmap outputs/heatmap_team_b.png --detect-roles --role-clusters 5
```

Generated outputs:

- `outputs/player_teams_30s.csv`
- `outputs/team_tracked_30s.mp4`
- `outputs/heatmap_team_a.png`
- `outputs/heatmap_team_b.png`
- `outputs/team_debug/team_assignments.csv`
- `outputs/team_debug/role_assignments.csv`
- `outputs/team_debug/movement_summary.csv`
- `outputs/team_debug/color_clusters.csv`
- `outputs/team_debug/color_clusters_palette.png`
- `outputs/team_debug/role_crops/`

## SoccerNet 720p Clip Pipeline

If `1_720p.mkv` exists under `data/SoccerNet/`, cut a 30-second 720p clip and rerun detection, tracking, movement analytics, player stats, and team classification with `_720p` output names:

```bash
python scripts/run_720p_clip_pipeline.py
```

Generated outputs include:

- `data/sample_30s_720p.mp4`
- `outputs/tracked_30s_720p.mp4`
- `outputs/detections_30s_720p.csv`
- `outputs/tracks_30s_720p.csv`
- `outputs/heatmap_all_720p.png`
- `outputs/trajectories_all_720p.png`
- `outputs/player_stats_30s_720p.csv`
- `outputs/player_teams_30s_720p.csv`
- `outputs/team_tracked_30s_720p.mp4`
- `outputs/heatmap_team_a_720p.png`
- `outputs/heatmap_team_b_720p.png`
- `outputs/team_debug/team_assignments.csv`
- `outputs/team_debug/role_assignments.csv`
- `outputs/team_debug/movement_summary.csv`
- `outputs/team_debug/color_clusters.csv`
- `outputs/team_debug/color_clusters_palette.png`
- `outputs/team_debug/team_palette.png`
- `outputs/team_debug/crops/`
- `outputs/team_debug/role_crops/`

Team classification is a non-training heuristic. It samples only the central
upper-body region of each tracked box, rejects grass-heavy or low-quality crops,
aggregates jersey colors per track, then clusters eligible tracks into two team
colors. Role detection then checks additional color clusters, track position,
and movement to mark goalkeeper/referee candidates when the evidence is strong
enough. Tracks without enough clean torso samples are labeled `Unknown`.

Current 720p validation counts:

- Team assignments: 59 Team A, 89 Team B, 57 Unknown.
- Role assignments: 59 `team_a_player`, 89 `team_b_player`, 1 `goalkeeper_right`, 56 `unknown`, 0 `referee`.
- Track 49 is the current `goalkeeper_right` candidate.
- Team heatmaps are outfield-only by default.

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
|   |-- tracking/      # Centroid tracking code
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
