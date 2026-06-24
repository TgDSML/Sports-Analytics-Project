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
- Optional ball detection baseline with separate CSV and annotated video outputs
- Ball tracking from filtered ball detections
- Baseline possession estimation from nearest player-to-ball distance

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

## Local SoccerNet Data

Place local SoccerNet videos under `data/SoccerNet/`. The orchestrated pipeline expects a real video path via `--video` and writes all generated artifacts under a per-clip folder in `outputs/`.

Example source path:

```text
data/SoccerNet/england_epl/2015-2016/2015-09-26 - 17-00 Manchester United 3 - 0 Sunderland/1_720p.mkv
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

## Ball Detection Baseline

Ball detection is implemented as a separate opt-in baseline so player detection, player tracking, and team overlays stay unchanged. It runs YOLO on the input video, keeps model classes named `ball`, `football`, `soccer ball`, or `sports ball`, exports raw and filtered CSVs, and writes a separate filtered annotated video with compact `ball 0.72` labels.

Run ball detection with the default generic YOLO model:

```bash
python main.py --video data/sample_30s.mp4 --output outputs/tracked_30s.mp4 --detect-ball --ball-raw-output-csv outputs/ball_detections_raw.csv --ball-output-csv outputs/ball_detections_filtered.csv --ball-video-output outputs/ball_detected_filtered.mp4
```

Run ball detection with a soccer-specific or fine-tuned model:

```bash
python main.py --video data/sample_30s.mp4 --output outputs/tracked_30s.mp4 --detect-ball --ball-model path/to/ball_model.pt --ball-conf 0.10 --ball-imgsz 1280 --ball-raw-output-csv outputs/ball_detections_raw.csv --ball-output-csv outputs/ball_detections_filtered.csv --ball-video-output outputs/ball_detected_filtered.mp4
```

Practical 720p filters are enabled by default:

- top-frame exclusion for scoreboard/broadcast graphics: `--ball-exclude-top-ratio 0.08`
- compact candidate size bounds: `--ball-min-area 20`, `--ball-max-area 500`, `--ball-min-width 4`, `--ball-max-width 30`, `--ball-min-height 4`, `--ball-max-height 30`
- one retained candidate per frame: `--ball-max-detections-per-frame 1`

Optional debug frames can be exported by setting `--ball-debug-frame-stride`, for example:

```bash
python main.py --video data/sample_30s.mp4 --output outputs/tracked_30s.mp4 --detect-ball --ball-debug-frame-stride 25
```

Ball detection CSV columns:

- `frame`
- `timestamp`
- `x1`
- `y1`
- `x2`
- `y2`
- `center_x`
- `center_y`
- `confidence`
- `class_id`
- `class_name`

Current limitation: a generic COCO YOLO model may not reliably detect the football in broadcast video. The ball is small, fast, blurry, and often occluded, so zero detections or false positives are expected on some clips. The next step is to use a soccer-specific ball detector or a YOLO model fine-tuned on football ball annotations.

Ball diagnostics are written to:

- `outputs/ball_debug/ball_detection_summary.csv`
- `outputs/ball_debug/ball_detection_summary.md`

## Ball Tracking And Possession

Ball tracking links filtered ball detections using nearest-neighbor association, a maximum movement threshold, confidence-aware scoring, short track persistence, and interpolation across short gaps. It is designed as an explainable baseline rather than a learned ball tracker.

Run ball tracking from an existing filtered ball detection CSV:

```bash
python -m src.tracking.ball_tracker --video data/sample_30s_720p.mp4 --detections-csv outputs/ball_detections_filtered_30s_720p.csv --output-csv outputs/ball_tracks_30s_720p.csv --output-video outputs/ball_tracked_30s_720p.mp4
```

Possession is an experimental baseline. It is estimated per ball-track frame by finding the nearest eligible tracked player to the ball and mapping that player to the team-classification output. The default QA gates skip interpolated ball points, require minimum ball confidence, exclude unknown/referee/goalkeeper roles, and smooth team switches across consecutive frames. If no eligible player is close enough, the frame is marked as `None`.

Run possession estimation from existing player, team, and ball-track CSVs:

```bash
python -m src.analytics.possession --video data/sample_30s_720p.mp4 --player-tracks-csv outputs/tracks_30s_720p.csv --teams-csv outputs/player_teams_30s_720p.csv --ball-tracks-csv outputs/ball_tracks_30s_720p.csv --output-csv outputs/possession_30s_720p.csv --debug-csv outputs/possession_debug_30s_720p.csv --summary-csv outputs/possession_summary_30s_720p.csv --summary-md outputs/possession_summary_30s_720p.md --qa-summary-md outputs/possession_qa_summary.md --output-video outputs/possession_30s_720p.mp4 --debug-video outputs/possession_debug_30s_720p.mp4
```

Generated outputs:

- `outputs/ball_tracks_30s_720p.csv`
- `outputs/ball_tracked_30s_720p.mp4`
- `outputs/ball_debug/ball_tracking_summary.csv`
- `outputs/ball_debug/ball_tracking_summary.md`
- `outputs/possession_30s_720p.csv`
- `outputs/possession_summary_30s_720p.csv`
- `outputs/possession_summary_30s_720p.md`
- `outputs/possession_30s_720p.mp4`
- `outputs/possession_debug_30s_720p.csv`
- `outputs/possession_debug_30s_720p.mp4`
- `outputs/possession_qa_summary.md`

Current 720p QA summary:

- Ball tracks: 27
- Longest ball track: 83 points
- Interpolated ball points: 195
- Gaps filled: 82
- Team A possession: 2.63%
- Team B possession: 6.64%
- Unknown possession: 0.00%
- No possession: 90.73%
- Possession status: experimental baseline, not validated analytics.

## Team Classification

The team classifier assigns tracked players to `Team A`, `Team B`, or `Unknown` using an explainable jersey-color workflow. It samples only the central torso area of each tracked box, rejects grass-heavy or low-quality crops, aggregates valid jersey samples per `track_id`, and clusters robust track-level LAB colors.

Role-aware classification can also separate outfield players, goalkeeper candidates, referee candidates, and unknown tracks. It uses extra color clusters plus track position and movement summaries. By default, team heatmaps include only outfield players so goalkeeper/referee movement does not distort team movement maps.

This is a baseline only. It can be wrong when detections are noisy, players are occluded, lighting changes, kits have similar colors, or tracks are fragmented. Goalkeeper/referee detection is heuristic and deliberately avoids forcing labels when evidence is weak. It does not use homography, event detection, roster identity, or a learned re-identification model.

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

Use `src.pipeline.run_clip` for the end-to-end 720p workflow. It cuts the clip, runs player and ball detection/tracking, assigns teams, estimates possession, and creates carries, interceptions, passing summaries, passing maps, videos, logs, and QA reports inside one per-match output folder.

PowerShell example:

```powershell
.\venv\Scripts\python.exe -m src.pipeline.run_clip --config configs\default_pipeline.yaml --video "data\SoccerNet\england_epl\2015-2016\2015-09-26 - 17-00 Manchester United 3 - 0 Sunderland\1_720p.mkv" --clip-id england_epl__2015_2016__2015_09_26___17_00_Manchester_United_3___0_Sunderland__h1_720p_golden_test
```

Expected output layout:

- `outputs/<clip_id>/preprocessed/clip.mp4`
- `outputs/<clip_id>/detections/`
- `outputs/<clip_id>/tracks/`
- `outputs/<clip_id>/teams/`
- `outputs/<clip_id>/possession/`
- `outputs/<clip_id>/carries/`
- `outputs/<clip_id>/interceptions/`
- `outputs/<clip_id>/tactical/passing_maps/`
- `outputs/<clip_id>/visualizations/`
- `outputs/<clip_id>/logs/`

Team classification is a non-training heuristic. It samples only the central
upper-body region of each tracked box, rejects grass-heavy or low-quality crops,
aggregates jersey colors per track, then clusters eligible tracks into two team
colors. Role detection then checks additional color clusters, track position,
and movement to mark goalkeeper/referee candidates when the evidence is strong
enough. Tracks without enough clean torso samples are labeled `Unknown`.

## Temporal Event-Candidate Sweep

After clip pipeline outputs exist, run the normal temporal event-candidate sweep from the project root:

```powershell
.\venv\Scripts\python.exe temporal_module\scripts\run_event_candidate_sweep.py --outputs-root outputs --derived-root temporal_module\data\derived --k-nearest 4 --defender-radius-px 100 --max-merged-pass-span-frames 24 --interception-duplicate-frame-tolerance 2
```

This sweep builds temporal frames, weak pass exports, carries, pass candidates, turnover/interception candidates, shot candidates, refined candidates, unified event candidates, and then runs pass scoring as the final downstream stage. Pass scoring runs only after each clip has both `pass_candidates_refined.csv` and `event_candidates_unified.csv`; per-clip sweep status is written to `temporal_module\data\derived\event_candidate_sweep_summary.csv`.

For a one-clip validation run, pass the exact derived clip directory name:

```powershell
python temporal_module\scripts\run_event_candidate_sweep.py ^
--outputs-root outputs ^
--derived-root temporal_module\data\derived ^
--clip-id england_epl__2014_2015__2015_04_11___19_30_Burnley_0___1_Arsenal__h1_720p ^
--k-nearest 4 ^
--defender-radius-px 100 ^
--max-merged-pass-span-frames 24 ^
--interception-duplicate-frame-tolerance 2
```

With `--clip-id`, the summary CSV contains only that clip and sets `selected_clip_id`; without it, the full sweep behavior is unchanged.

## SoccerNet Download Planning

Inspect a local SoccerNet folder and create candidate inventory files:

```powershell
python temporal_module\scripts\inspect_soccernet_dataset.py ^
--soccernet-root "REAL_PATH_TO_SOCCERNET_FOLDER"
```

Build the local existing-clip manifest before adding more SoccerNet clips:

```powershell
python temporal_module\scripts\build_soccernet_download_plan.py ^
--outputs-root outputs ^
--derived-root temporal_module\data\derived ^
--target-new-clips 25
```

After a SoccerNet inventory exists, build a duplicate-aware new-clip plan:

```powershell
python temporal_module\scripts\build_soccernet_download_plan.py ^
--outputs-root outputs ^
--derived-root temporal_module\data\derived ^
--soccernet-inventory temporal_module\data\soccernet_inventory\soccernet_pilot_candidates.csv ^
--target-new-clips 25
```

The planner does not download files or overwrite existing clips. It retains local clips, excludes exact duplicates using conservative normalized path/name matches, and expects only reviewed new rows from the plan to be downloaded later.

Install the official SoccerNet package before downloading reviewed selections:

```powershell
python -m pip install SoccerNet
```

Dry-run selected tracking downloads from the reviewed plan:

```powershell
python temporal_module\scripts\download_soccernet_selected_clips.py ^
--selection-manifest temporal_module\data\soccernet_inventory\soccernet_new_clip_download_plan.csv ^
--soccernet-root "C:\Users\nikoma\Desktop\SoccerNet" ^
--dataset-mode tracking ^
--dry-run ^
--max-downloads 25
```

`--dry-run` validates selected manifest rows and intended paths without requiring the SoccerNet video password. Broadcast downloads require setting the password outside the repository only when downloading for real:

```powershell
set SOCCERNET_VIDEO_PASSWORD=YOUR_PASSWORD
```

The downloader reads only reviewed `candidate_for_new_download` rows, never downloads the full SoccerNet dataset by default, and writes logs only under `temporal_module\data\soccernet_inventory`.

Build a reproducible Premier League new-clip selection manifest without downloading videos:

```powershell
python src\data_tools\download_soccernet_epl.py ^
  --sample-size 25 ^
  --seed 42 ^
  --exclude-manifest "PATH_TO_OLD_MANIFEST.csv" ^
  --outputs-root outputs ^
  --derived-root temporal_module\data\derived ^
  --soccernet-dir "C:\Users\nikoma\Desktop\SoccerNet" ^
  --manifest "temporal_module\data\soccernet_inventory\epl_new_25_selection.csv" ^
  --dry-run
```

Review `epl_new_25_selection.csv`, `epl_new_25_selection_audit.csv`, and `epl_new_25_selection_summary.json`.

Then dry-run the existing selected-clip downloader:

```powershell
python temporal_module\scripts\download_soccernet_selected_clips.py ^
  --selection-manifest "temporal_module\data\soccernet_inventory\epl_new_25_selection.csv" ^
  --soccernet-root "C:\Users\nikoma\Desktop\SoccerNet" ^
  --dataset-mode broadcast_720p ^
  --dry-run ^
  --max-downloads 25
```

Only after reviewing the dry-run output should the selected-clip downloader be run again without `--dry-run`. Do not store SoccerNet passwords in source code, manifests, logs, config files, or README documentation.

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

Generated artifacts are intentionally not committed. After cloning or merging, regenerate local outputs with `src.pipeline.run_clip` and a local source video under `data/SoccerNet/`.
