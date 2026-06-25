# Sports Analytics Project

## Overview

Sports Analytics Project is an end-to-end football video analytics pipeline for broadcast-style match clips. It combines computer vision, tracking, tactical analytics, temporal feature engineering, weak-label event learning, and a CVAT-based gold-label workflow.

The project currently supports:

- player detection
- player tracking
- team classification
- ball detection and filtering
- ball tracking
- possession estimation
- carry, pass, turnover, interception, and shot candidate generation
- temporal feature extraction
- weak-label event dataset construction
- CVAT gold-label import
- BiGRU and CNN-LSTM event classification experiments

Large videos, model checkpoints, generated outputs, and private annotation exports are intentionally kept out of Git. A fresh clone contains the source code, configs, manifests, lightweight summaries, and reproducibility/report artifacts that are small enough to version.

## Key Features

- YOLO/Ultralytics player detection with ByteTrack tracking.
- Explainable jersey-color team classification and role heuristics.
- Optional football detection, filtering, tracking, and possession estimation.
- Per-clip output folders with detections, tracks, teams, possession, tactical maps, logs, and QA summaries.
- Temporal module for engineered per-frame features and event candidate catalogs.
- Weak-label learning from heuristic event candidates without CVAT or gold labels.
- CVAT workflow for manually annotated temporal event intervals.
- Gold supervised BiGRU and CNN-LSTM event classification prototypes.
- Manifest-based batch processing for reproducible clip selection.

## System Architecture

```text
Raw video
   |
   v
YOLO player detection
   |
   v
ByteTrack tracking
   |
   v
Team classification
   |
   v
Ball detection/filtering
   |
   v
Possession estimation
   |
   v
Event candidates
   |
   v
Temporal features
   |
   v
Weak labels / CVAT gold labels
   |
   v
BiGRU / CNN-LSTM
```

## Repository Structure

```text
Sports-Analytics-Project/
|-- configs/              # Pipeline configuration files
|-- data/                 # Local input data; large files are ignored
|   `-- manifests/        # Versioned clip manifests
|-- outputs/              # Generated per-clip analytics outputs
|-- src/                  # Main detection, tracking, analytics, and pipeline code
|-- temporal_module/      # Temporal features, event candidates, weak/gold learning
|   |-- data/             # Local/generated temporal data
|   |-- docs/             # Temporal-module notes
|   |-- reproducibility/  # Lightweight reproducibility artifacts
|   |-- runs_gold/        # Gold supervised model summaries and metrics
|   |-- scripts/          # Temporal workflow scripts
|   `-- src/              # Shared temporal feature/model helpers
|-- main.py               # Standalone YOLO video baseline
|-- requirements.txt      # Runtime Python dependencies
|-- setup.bat             # Windows setup helper
|-- setup.sh              # Linux/macOS setup helper
`-- README.md
```

`PROJECT_SUMMARY.md` is not present on the current `main` branch. Use this README and the summaries under `temporal_module/` as the primary documentation.

## Installation

Clone the repository and create a virtual environment.

```bash
git clone <repo-url>
cd Sports-Analytics-Project
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

Install runtime dependencies:

```bash
pip install -r requirements.txt
```

Optional development dependencies:

```bash
pip install pytest PyYAML
```

Notes:

- `ffmpeg` is required for clip preprocessing. Put it on `PATH` or under `tools/ffmpeg/`.
- Ultralytics may download YOLO weights on first use if the requested model is not already local.
- The `SoccerNet` package is listed in `requirements.txt`, but SoccerNet videos and credentials are not stored in this repository.

## Data Setup

Large SoccerNet videos are not stored in Git. Put local videos under `data/SoccerNet/` or pass an explicit video path to the single-clip runner.

Expected local layout:

```text
data/
|-- SoccerNet/
|   `-- england_epl/
|       `-- <season>/
|           `-- <match>/
|               |-- 1_720p.mkv
|               `-- 2_720p.mkv
`-- manifests/
    `-- dataset_manifest.csv
```

The default manifest is:

```text
data/manifests/dataset_manifest.csv
```

Manifest columns:

```text
clip_id,split,competition,source,game_id,half,video_path,enabled,notes
```

Example row:

```csv
england_epl__2014_2015__2015_04_11___19_30_Burnley_0___1_Arsenal__h1_720p,valid,england_epl,SoccerNet,england_epl/2014-2015/2015-04-11 - 19-30 Burnley 0 - 1 Arsenal,1,data/SoccerNet/england_epl/2014-2015/2015-04-11 - 19-30 Burnley 0 - 1 Arsenal/1_720p.mkv,true,pilot_random_sample
```

Only enabled manifest rows are processed by the batch runner.

## Quick Start: Run One Clip

Requires a local video file.

```bash
python -m src.pipeline.run_clip --config configs/default_pipeline.yaml --video data/sample_30s_720p.mp4 --clip-id sample_30s
```

For a SoccerNet manifest clip, either pass `--video` directly or use a manifest `clip_id`:

```bash
python -m src.pipeline.run_clip --config configs/default_pipeline.yaml --manifest data/manifests/dataset_manifest.csv --clip-id england_epl__2014_2015__2015_04_11___19_30_Burnley_0___1_Arsenal__h1_720p
```

If the video path in the manifest is not available locally, the command will fail at the input stage. This is expected for a fresh clone without SoccerNet data.

## Batch Processing with Manifest

Requires local videos matching the manifest paths.

```bash
python -m src.pipeline.run_dataset --config configs/default_pipeline.yaml --manifest data/manifests/dataset_manifest.csv
```

Process selected clips only:

```bash
python -m src.pipeline.run_dataset --config configs/default_pipeline.yaml --manifest data/manifests/dataset_manifest.csv --clip-id england_epl__2014_2015__2015_04_11___19_30_Burnley_0___1_Arsenal__h1_720p
```

`run_dataset` does not currently expose a dry-run flag. Use manifest inspection and single-clip runs before launching a full batch.

## Main Outputs

Each clip is written under:

```text
outputs/<clip_id>/
|-- preprocessed/
|   `-- clip.mp4
|-- detections/
|   |-- detections.csv
|   |-- ball_detections_raw.csv
|   |-- ball_detections_filtered.csv
|   `-- ball_detection_summary.md
|-- tracks/
|   |-- tracks.csv
|   |-- tracks_top5.csv
|   |-- ball_tracks.csv
|   `-- ball_tracking_summary.md
|-- teams/
|   |-- player_teams.csv
|   |-- heatmap_team_a.png
|   |-- heatmap_team_b.png
|   `-- debug/
|-- possession/
|   |-- possession.csv
|   |-- possession_debug.csv
|   |-- possession_summary.csv
|   |-- possession_summary.md
|   `-- possession_qa_summary.md
|-- carries/
|-- interceptions/
|-- tactical/
|   |-- player_stats.csv
|   |-- player_stats.xlsx
|   |-- passing_summary.csv
|   |-- passing_summary.md
|   `-- passing_maps/
|-- visualizations/
|   |-- tracked.mp4
|   |-- team_tracked.mp4
|   |-- ball_tracked.mp4
|   |-- possession.mp4
|   |-- heatmap_all.png
|   `-- trajectories_all.png
`-- logs/
```

Temporal outputs are written under:

```text
temporal_module/data/derived/<clip_id>/
|-- temporal_frames.csv
|-- passes_weak.csv
`-- events/
    |-- pass_candidates.csv
    |-- turnover_candidates.csv
    |-- shot_candidates.csv
    |-- pass_candidates_refined.csv
    |-- turnover_candidates_refined.csv
    |-- shot_candidates_refined.csv
    `-- event_candidates_unified.csv
```

## Temporal Module

The temporal module converts per-frame pipeline outputs into model-ready event data.

- `temporal_frames.csv`: one row per frame with engineered image-space features.
- 64-frame windows: default sequence length for BiGRU experiments.
- 68 engineered features: current gold temporal feature count in committed gold summaries.
- Weak labels: derived from heuristic event candidates only.
- Gold labels: imported from CVAT temporal event intervals.
- BiGRU: sequence model over engineered temporal features.
- CNN-LSTM: visual frame-window prototype over sampled RGB frames.

Build temporal frames and event candidates after the clip pipeline has produced `outputs/<clip_id>/`:

```bash
python temporal_module/scripts/run_event_candidate_sweep.py --outputs-root outputs --derived-root temporal_module/data/derived --k-nearest 4 --defender-radius-px 100 --max-merged-pass-span-frames 24 --interception-duplicate-frame-tolerance 2
```

Build a weak-label GRU dataset from non-gold event candidates:

```bash
python temporal_module/scripts/build_weak_event_gru_dataset.py --derived-root temporal_module/data/derived --output-dir temporal_module/data/weak_event_gru --window-seconds 8.0 --label-region-seconds 1.0 --stride-seconds 0.5 --seed 42 --train-clips 15 --val-clips 5 --test-clips 5 --balance-strategy none
```

Train the weak-label BiGRU baseline:

```bash
python temporal_module/scripts/train_weak_event_gru.py --dataset-dir temporal_module/data/weak_event_gru --derived-root temporal_module/data/derived --model-dir temporal_module/runs/weak_event_gru/model --report-dir temporal_module/runs/weak_event_gru/report --epochs 40 --batch-size 8 --hidden-size 32 --dropout 0.2 --learning-rate 0.001 --patience 8
```

Build gold temporal windows after CVAT import:

```bash
python temporal_module/scripts/build_gold_event_windows.py --gold-events temporal_module/data/gold_event_project/annotations/gold_event_intervals.csv --derived-root temporal_module/data/derived --output-dir temporal_module/data_gold/event_windows
```

Train the gold BiGRU:

```bash
python temporal_module/scripts/train_gold_event_bigru.py --data temporal_module/data_gold/event_windows/gold_event_windows.npz --output-dir temporal_module/runs_gold/event_bigru
```

Build gold frame windows for the visual prototype:

```bash
python temporal_module/scripts/build_gold_frame_windows.py --gold-events temporal_module/data/gold_event_project/annotations/gold_event_intervals.csv --video-root temporal_module/data/gold_event_project/cvat_uploads --output-dir temporal_module/data_gold/frame_windows
```

Train the gold CNN-LSTM:

```bash
python temporal_module/scripts/train_gold_event_cnn_lstm.py --data temporal_module/data_gold/frame_windows/gold_frame_windows.npz --output-dir temporal_module/runs_gold/event_cnn_lstm
```

The weak and gold commands require local generated outputs. A fresh clone without videos, derived frames, candidate files, gold intervals, or frame-window tensors will not be able to run these stages immediately.

## CVAT Gold Annotation Workflow

CVAT is used for temporal event annotation only. It is not used here for player bounding boxes or ball bounding boxes.

Gold event labels:

- `carry`
- `pass`
- `turnover`
- `shot`
- `uncertain`

Expected CVAT export location:

```text
temporal_module/data/gold_event_project/cvat_exports/
```

The importer creates:

```text
temporal_module/data/gold_event_project/annotations/gold_event_intervals.csv
```

Import command:

```bash
python temporal_module/scripts/import_cvat_gold_annotations.py --manifest temporal_module/data/gold_event_project/manifests/gold_clip_manifest.csv --export-dir temporal_module/data/gold_event_project/cvat_exports --output temporal_module/data/gold_event_project/annotations/gold_event_intervals.csv
```

Dry-run import:

```bash
python temporal_module/scripts/import_cvat_gold_annotations.py --manifest temporal_module/data/gold_event_project/manifests/gold_clip_manifest.csv --export-dir temporal_module/data/gold_event_project/cvat_exports --output temporal_module/data/gold_event_project/annotations/gold_event_intervals.csv --dry-run
```

CVAT exports and upload videos can be large or private. Keep them out of commits unless the team explicitly decides otherwise.

## Reproducing Report Experiments

Compile check:

```bash
python -m compileall -q main.py src temporal_module
```

Single clip pipeline, requires local video:

```bash
python -m src.pipeline.run_clip --config configs/default_pipeline.yaml --video data/sample_30s_720p.mp4 --clip-id sample_30s
```

Dataset pipeline, requires local SoccerNet videos:

```bash
python -m src.pipeline.run_dataset --config configs/default_pipeline.yaml --manifest data/manifests/dataset_manifest.csv
```

Temporal event-candidate sweep, requires per-clip outputs:

```bash
python temporal_module/scripts/run_event_candidate_sweep.py --outputs-root outputs --derived-root temporal_module/data/derived --k-nearest 4 --defender-radius-px 100 --max-merged-pass-span-frames 24 --interception-duplicate-frame-tolerance 2
```

Weak-label reproducibility runner, requires non-gold pipeline outputs and event candidates:

```bash
python temporal_module/scripts/run_weak_25clip_reproducibility.py --skip-training
```

Full weak-label reproducibility run, long running:

```bash
python temporal_module/scripts/run_weak_25clip_reproducibility.py
```

Gold-label training:

```bash
python temporal_module/scripts/train_gold_event_bigru.py --data temporal_module/data_gold/event_windows/gold_event_windows.npz --output-dir temporal_module/runs_gold/event_bigru
python temporal_module/scripts/train_gold_event_cnn_lstm.py --data temporal_module/data_gold/frame_windows/gold_frame_windows.npz --output-dir temporal_module/runs_gold/event_cnn_lstm
```

Detection metric evaluation, requires YOLO/CVAT-style ground-truth labels and prediction CSVs:

```bash
python -m src.metrics.evaluate_metrics --ground-truth data/annotations/video_1/obj_train_data --predictions outputs/<clip_id>
```

## Results Summary

Known committed results should be interpreted carefully. They describe local experiments on limited data, not final production performance.

- Player detection F1: no final player-detection F1 is documented in committed reports.
- Ball detection F1: no final ball-detection F1 is documented in committed reports.
- Gold BiGRU, held-out test: accuracy `0.4953`, macro F1 `0.2689`.
- Gold CNN-LSTM, held-out test: accuracy `0.5814`, macro F1 `0.2424`.
- The gold summaries warn that rare classes are sparse. For example, the validation split has no `shot` windows.
- The CNN-LSTM visual prototype currently has higher test accuracy than the gold BiGRU in the committed run, but both macro F1 scores are low because rare event classes remain difficult.
- Event candidates are experimental heuristic signals, not ground truth.

Result artifacts:

```text
temporal_module/runs_gold/event_bigru/training_summary.md
temporal_module/runs_gold/event_bigru/metrics.json
temporal_module/runs_gold/event_cnn_lstm/training_summary.md
temporal_module/runs_gold/event_cnn_lstm/metrics.json
```

## Limitations

- Coordinates are image-space only; there is no pitch calibration or homography.
- Event candidates are heuristic and can miss or duplicate football events.
- Possession is estimated from tracked player and ball proximity, not ground truth.
- Team classification is color-based and can fail with similar kits, occlusions, lighting changes, or fragmented tracks.
- Ball detection remains fragile because the ball is small, fast, and often occluded.
- The gold-labelled dataset is limited.
- Rare classes such as `shot` and `turnover` are highly imbalanced.
- CNN-LSTM training uses limited visual data and should be treated as a prototype.
- SoccerNet videos, credentials, private CVAT data, and large generated tensors are not included in Git.

## Future Work

- Pitch calibration and homography for field-relative coordinates.
- Larger CVAT annotation corpus.
- More balanced event labels, especially for shots and turnovers.
- Stronger ball detector and tracker.
- Transformer-based temporal event models.
- RF-DETR as an optional detector backend if the team decides to maintain it.
- Better player re-identification across track fragments.
- Real-time or near-real-time inference.
- Clearer packaging of reproducibility runs with smaller committed fixtures.

## Contributors

Project team contributors are not listed in the current repository metadata. Add names here before public release if required by the team.

## License / Acknowledgements

No license file is currently present. Add a `LICENSE` file before public distribution.

Acknowledgements:

- SoccerNet for football video dataset infrastructure.
- Ultralytics YOLO for object detection.
- ByteTrack for multi-object tracking.
- CVAT for temporal annotation workflow.
- PyTorch for neural network experiments.
