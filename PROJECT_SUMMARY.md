# Project Summary

## What This Project Does

This repository is a football video analytics baseline. It processes local match video with a pretrained Ultralytics YOLO model, keeps person detections, tracks players across frames, and produces visual and tabular analytics artifacts.

The current pipeline supports:

- Player detection with YOLO.
- Player tracking with Ultralytics ByteTrack by default.
- A simpler centroid tracker for comparison.
- Detection and tracking CSV exports.
- Movement heatmaps.
- Player trajectory plots.
- Basic per-player movement statistics.
- Improved jersey-color team classification using torso crops, track-level color aggregation, and robust LAB clustering.
- Role-aware classification for outfield players, goalkeeper candidates, referee candidates, and unknown tracks.
- Team-colored tracking video and outfield-only team heatmaps by default.
- A 720p SoccerNet clip pipeline that cuts a 30-second sample and regenerates the main outputs.

## Current Repository Shape

- `main.py`: main CLI for detection, tracking, CSV export, and optional analytics.
- `src/detection/`: YOLO wrapper and drawing helpers.
- `src/tracking/`: centroid tracker fallback.
- `src/analytics/`: heatmaps, trajectories, player stats, and jersey-color team classification.
- `src/utils/`: CSV and video helper functions.
- `scripts/download_soccernet_video.py`: downloads a small SoccerNet validation sample.
- `scripts/run_720p_clip_pipeline.py`: runs the full 720p clip workflow.
- `data/`: local videos and downloaded SoccerNet data, ignored by Git.
- `outputs/`: generated videos, CSVs, plots, and debug artifacts, ignored by Git.

## Existing 720p Output Snapshot

The local workspace already contains a completed 30-second 720p run:

- `data/sample_30s_720p.mp4`
- `outputs/tracked_30s_720p.mp4`
- `outputs/team_tracked_30s_720p.mp4`
- `outputs/detections_30s_720p.csv`
- `outputs/tracks_30s_720p.csv`
- `outputs/heatmap_all_720p.png`
- `outputs/trajectories_all_720p.png`
- `outputs/trajectories_top5_720p.png`
- `outputs/player_stats_30s_720p.csv`
- `outputs/player_stats_30s_720p.xlsx`
- `outputs/player_teams_30s_720p.csv`
- `outputs/heatmap_team_a_720p.png`
- `outputs/heatmap_team_b_720p.png`
- `outputs/team_debug/`

Current generated CSV counts:

- Detection rows: 5,546.
- Tracked rows: 5,372.
- Team assignments: 59 Team A, 89 Team B, 57 Unknown.
- Role assignments: 59 `team_a_player`, 89 `team_b_player`, 1 `goalkeeper_right`, 56 `unknown`, 0 `referee`.
- Goalkeeper QA: track 49 is an orange color outlier near the right-side penalty area and is labeled `goalkeeper_right`.
- Team heatmaps are generated from outfield players only by default.

Top tracked players by visibility:

| Track ID | Frames Seen | Total Distance (px) | Avg Speed (px/sec) |
| --- | ---: | ---: | ---: |
| 63 | 272 | 1472.57 | 134.99 |
| 81 | 251 | 1326.25 | 131.22 |
| 76 | 233 | 1092.04 | 117.36 |
| 106 | 164 | 1137.74 | 168.85 |
| 70 | 152 | 820.65 | 130.33 |

## Baseline Limitations

This is a reproducible baseline, not a production-grade tactical analytics system. Main limitations:

- Team classification is color-based and can fail with occlusion, similar kits, lighting changes, bad crops, or fragmented tracks.
- Goalkeeper/referee role detection is heuristic. It uses color outliers plus track position and movement, and it avoids forcing labels when evidence is weak.
- Tracks are image-space tracks, not field-coordinate tracks. Distances are pixels, not meters.
- There is no homography, camera calibration, possession model, event detection, or player re-identification model.
- Generated IDs are tracking IDs, not roster/player identities.

## Ready-To-Run Checks

Compile-check the code:

```bash
python -m compileall main.py src scripts
```

Regenerate analytics from the existing tracks CSV without rerunning YOLO:

```bash
python -m src.analytics.heatmap --tracks-csv outputs/tracks_30s_720p.csv --output outputs/heatmap_all_720p.png --frame-width 1280 --frame-height 720
python -m src.analytics.trajectories --tracks-csv outputs/tracks_30s_720p.csv --output outputs/trajectories_all_720p.png --frame-width 1280 --frame-height 720
python -m src.analytics.player_stats --tracks-csv outputs/tracks_30s_720p.csv --output outputs/player_stats_30s_720p.csv --excel-output outputs/player_stats_30s_720p.xlsx
```

Run the full 720p pipeline when the SoccerNet source video is available:

```bash
python scripts/run_720p_clip_pipeline.py
```
