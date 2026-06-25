# Temporal Module

This isolated module builds derived temporal-event data under `temporal_module/`.
It reads existing analytics outputs as read-only inputs and never writes inside `outputs/`.

## Single-clip frame table

```cmd
python temporal_module\scripts\build_temporal_frames.py ^
  --clip-id <clip_id> ^
  --tracks outputs\<clip_id>\tracks\tracks.csv ^
  --ball-tracks outputs\<clip_id>\tracks\ball_tracks.csv ^
  --player-teams outputs\<clip_id>\teams\player_teams.csv ^
  --possession outputs\<clip_id>\possession\possession.csv ^
  --possession-debug outputs\<clip_id>\possession\possession_debug.csv ^
  --output temporal_module\data\derived\<clip_id>\temporal_frames.csv
```

## Batch build

```cmd
python temporal_module\scripts\build_all_temporal_frames.py ^
  --outputs-root outputs ^
  --derived-root temporal_module\data\derived ^
  --k-nearest 4 ^
  --defender-radius-px 100
```

The batch builder writes only under `temporal_module\data\derived` and never modifies `outputs/`.

## SoccerNet download planning

Inspect a local SoccerNet folder and create candidate inventory files:

```cmd
python temporal_module\scripts\inspect_soccernet_dataset.py ^
--soccernet-root "<SOCCERNET_ROOT>"
```

Build the local existing-clip manifest before adding more SoccerNet clips:

```cmd
python temporal_module\scripts\build_soccernet_download_plan.py ^
--outputs-root outputs ^
--derived-root temporal_module\data\derived ^
--target-new-clips 25
```

After a SoccerNet inventory exists, build a duplicate-aware new-clip plan:

```cmd
python temporal_module\scripts\build_soccernet_download_plan.py ^
--outputs-root outputs ^
--derived-root temporal_module\data\derived ^
--soccernet-inventory temporal_module\data\soccernet_inventory\soccernet_pilot_candidates.csv ^
--target-new-clips 25
```

This planning tool does not download anything and does not modify existing clips. Existing local clips are retained, exact local duplicates are excluded using conservative normalized path/name matches only, and only rows selected in the reviewed plan should be downloaded later.

Install the official SoccerNet package before downloading reviewed selections:

```cmd
python -m pip install SoccerNet
```

Dry-run selected tracking downloads from the reviewed plan:

```cmd
python temporal_module\scripts\download_soccernet_selected_clips.py ^
--selection-manifest temporal_module\data\soccernet_inventory\soccernet_new_clip_download_plan.csv ^
--soccernet-root "<SOCCERNET_ROOT>" ^
--dataset-mode tracking ^
--dry-run ^
--max-downloads 25
```

`--dry-run` validates selected manifest rows and intended paths without requiring the SoccerNet video password. Broadcast downloads require setting the password outside the repository only when downloading for real:

```cmd
set SOCCERNET_VIDEO_PASSWORD=YOUR_PASSWORD
```

The downloader reads only reviewed `candidate_for_new_download` rows, never downloads the full SoccerNet dataset by default, and writes logs only under `temporal_module\data\soccernet_inventory`.

Build a reproducible Premier League new-clip selection manifest without downloading videos:

```cmd
python src\data_tools\download_soccernet_epl.py ^
  --sample-size 25 ^
  --seed 42 ^
  --exclude-manifest "data\manifests\dataset_manifest.csv" ^
  --outputs-root outputs ^
  --derived-root temporal_module\data\derived ^
  --soccernet-dir "<SOCCERNET_ROOT>" ^
  --manifest "temporal_module\data\soccernet_inventory\epl_new_25_selection.csv" ^
  --dry-run
```

Review `epl_new_25_selection.csv`, `epl_new_25_selection_audit.csv`, and `epl_new_25_selection_summary.json`.

Then dry-run the existing selected-clip downloader:

```cmd
python temporal_module\scripts\download_soccernet_selected_clips.py ^
  --selection-manifest "temporal_module\data\soccernet_inventory\epl_new_25_selection.csv" ^
  --soccernet-root "<SOCCERNET_ROOT>" ^
  --dataset-mode broadcast_720p ^
  --dry-run ^
  --max-downloads 25
```

Only after reviewing the dry-run output should the selected-clip downloader be run again without `--dry-run`. Do not store SoccerNet passwords in source code, manifests, logs, config files, or README documentation.

## Normal event-candidate sweep

Run the full per-clip temporal analytics sweep:

```cmd
python temporal_module\scripts\run_event_candidate_sweep.py ^
  --outputs-root outputs ^
  --derived-root temporal_module\data\derived ^
  --k-nearest 4 ^
  --defender-radius-px 100 ^
  --max-merged-pass-span-frames 24 ^
  --interception-duplicate-frame-tolerance 2
```

The normal sweep runs temporal frames, weak pass exports, carry labels, pass candidates, turnover candidates, shot candidates, refinement, unified event candidates, and then pass scoring as the final downstream stage. The pass scorer is invoked only after `pass_candidates_refined.csv` and `event_candidates_unified.csv` exist for every derived clip, and it uses the standard recovery/post-turnover context arguments documented below. The combined per-clip status is written to `temporal_module\data\derived\event_candidate_sweep_summary.csv`; a pass-scoring failure marks the clip's final sweep status as `failed`.

Run the same sweep for exactly one derived clip directory by passing its exact folder name:

```cmd
python temporal_module\scripts\run_event_candidate_sweep.py ^
--outputs-root outputs ^
--derived-root temporal_module\data\derived ^
--clip-id england_epl__2014_2015__2015_04_11___19_30_Burnley_0___1_Arsenal__h1_720p ^
--k-nearest 4 ^
--defender-radius-px 100 ^
--max-merged-pass-span-frames 24 ^
--interception-duplicate-frame-tolerance 2
```

When `--clip-id` is supplied, the ID must exactly match a directory under `--derived-root`; substring matching is not used. The combined summary still writes to `event_candidate_sweep_summary.csv`, but contains only the selected clip and populates `selected_clip_id`.

## Derived Feature Definitions

- `ball_vx`, `ball_vy`: ball pixel velocity from frame-to-frame position delta divided by timestamp delta.
- `ball_speed`: Euclidean magnitude of ball pixel velocity.
- `ball_ax`, `ball_ay`: ball pixel acceleration from velocity delta divided by timestamp delta.
- `ball_acceleration`: Euclidean magnitude of ball pixel acceleration.
- `ball_velocity_valid`, `ball_acceleration_valid`: masks indicating whether motion values were computed from valid adjacent observations.
- `p{k}_vx`, `p{k}_vy`: nearest-player pixel velocity for player slot `k`.
- `p{k}_speed`: Euclidean magnitude of nearest-player pixel velocity.
- `p{k}_velocity_valid`: mask for nearest-player velocity availability.
- `possession_changed`: one when the possession owner changes from the previous valid owner.
- `frames_since_possession_change`: frame count since the most recent valid possession owner change.
- `nearest_teammate_distance`: nearest same-team player distance from the possessor in pixels.
- `nearest_opponent_distance`: nearest opposing-team player distance from the possessor in pixels.
- `defenders_near_ball`: count of opposing players within `--defender-radius-px` pixels of the ball.
- `team_a_centroid_x/y`, `team_b_centroid_x/y`: per-frame mean image position of tracked Team A or Team B players.
- `team_a_width`, `team_b_width`: per-frame max x minus min x in pixels.
- `team_a_depth`, `team_b_depth`: per-frame max y minus min y in pixels.
- `team_a_spread`, `team_b_spread`: mean Euclidean player distance from the team centroid in pixels.
- `team_a_shape_valid`, `team_b_shape_valid`: masks indicating at least one valid tracked player for that team.

All distances and motion features are in image pixels or pixels per timestamp unit, not meters.

## Weak pass events

```cmd
python temporal_module\scripts\export_pass_events.py ^
  --clip-id <clip_id> ^
  --possession outputs\<clip_id>\possession\possession.csv ^
  --possession-debug outputs\<clip_id>\possession\possession_debug.csv ^
  --output temporal_module\data\derived\<clip_id>\passes_weak.csv
```

Weak pass rows are produced from conservative possession transitions and are not ground truth.

Batch-export weak pass rows for every clip with an existing temporal frame table:

```cmd
python temporal_module\scripts\build_all_pass_events.py ^
  --outputs-root outputs ^
  --derived-root temporal_module\data\derived
```

Build improved weak pass candidates with ball-motion evidence:

```cmd
python temporal_module\scripts\build_pass_candidates.py ^
  --outputs-root outputs ^
  --derived-root temporal_module\data\derived
```

Build weak turnover and interception candidates:

```cmd
python temporal_module\scripts\build_turnover_candidates.py ^
  --outputs-root outputs ^
  --derived-root temporal_module\data\derived
```

Build conservative weak shot candidates:

```cmd
python temporal_module\scripts\build_shot_candidates.py ^
  --outputs-root outputs ^
  --derived-root temporal_module\data\derived
```

Refine raw event candidates for inspection, annotation, and later weak-label selection:

```cmd
python temporal_module\scripts\refine_event_candidates.py ^
  --derived-root temporal_module\data\derived ^
  --max-merged-pass-span-frames 24
```

Build unified event-candidate catalogs:

```cmd
python temporal_module\scripts\build_unified_event_candidates.py ^
  --outputs-root outputs ^
  --derived-root temporal_module\data\derived ^
  --interception-duplicate-frame-tolerance 2
```

Score refined pass candidates with interpretable heuristic evidence:

```cmd
python temporal_module\scripts\score_pass_candidates.py ^
  --derived-root temporal_module\data\derived ^
  --post-turnover-context-frames 5 ^
  --recovery-to-pass-overlap-tolerance-frames 3 ^
  --recovery-to-pass-min-receiver-stable-frames 2 ^
  --recovery-to-pass-min-frames-after-turnover 2
```

This scoring layer does not modify existing candidate files. Its strict score weak-label eligibility remains unchanged and is stored in `is_eligible_for_pass_score_weak_label`. The separate provisional v2 calibrated rule is stored in `is_eligible_for_calibrated_pass_weak_label`; it permits high-evidence same-team passes that the original refined gate excluded when stability and quality checks are satisfied. Post-turnover passes can avoid the turnover-overlap penalty only when the configured frame window, sender/receiver stability, same-team, carry-overlap, and ball-quality checks pass. Recovery-to-pass context is another provisional calibrated-v2 rule for passes overlapping the end of a confirmed turnover, and it requires same-team evidence, stable possession, free-ball evidence, low carry overlap, and acceptable ball quality before suppressing turnover/flicker penalties. Only candidates with validated recovery-to-pass context use `--recovery-to-pass-min-receiver-stable-frames` for calibrated weak-label eligibility; all other calibrated candidates keep the normal `--minimum-stable-frames` receiver-stability requirement. Neither strict nor calibrated labels are ground truth, and manual reviews remain necessary before final training-label decisions.

Build pass review manifests for manual calibration:

```cmd
python temporal_module\scripts\build_pass_review_manifest.py ^
  --derived-root temporal_module\data\derived
```

Build weak carry/background frame labels for every clip with an existing temporal frame table:

```cmd
python temporal_module\scripts\build_carry_labels.py ^
  --outputs-root outputs ^
  --derived-root temporal_module\data\derived
```

Build player movement analytics and model-ready temporal movement features:

```cmd
python temporal_module\scripts\build_player_movement_analytics.py ^
  --outputs-root outputs ^
  --derived-root temporal_module\data\derived ^
  --grid-cols 12 ^
  --grid-rows 8
```

Build team-shape temporal features and 5-second tactical summaries:

```cmd
python temporal_module\scripts\build_shape_temporal_features.py ^
  --derived-root temporal_module\data\derived ^
  --rolling-frames 25 ^
  --window-seconds 5.0
```

Build fixed-length carry/background prototype windows:

```cmd
python temporal_module\scripts\build_carry_windows.py ^
  --derived-root temporal_module\data\derived ^
  --dataset-root temporal_module\data\datasets ^
  --window-length 64 ^
  --stride 16
```

Create the fixed clip-level carry/background split:

```cmd
python temporal_module\scripts\create_carry_clip_split.py ^
  --dataset-root temporal_module\data\datasets
```

Train the carry/background BiGRU prototype:

```cmd
python temporal_module\scripts\train_carry_bigru.py ^
  --dataset-root temporal_module\data\datasets ^
  --runs-root temporal_module\runs
```

Inspect the saved metrics:

```cmd
type temporal_module\runs\carry_bigru_seed42\metrics.json
```
