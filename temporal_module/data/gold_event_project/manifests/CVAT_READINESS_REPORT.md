# CVAT Readiness Report

- Source manifest: `data\manifests\dataset_manifest.csv`
- Selected source-manifest row count: 25
- Processed original-pipeline clip count: 25
- Temporal-frame-ready clip count: 25
- Frame-aligned clip count: 25
- CVAT-upload-copy count: 25
- Blocked clip count: 0
- Corpus status: READY FOR CVAT
- CVAT upload folder: `temporal_module\data\gold_event_project\cvat_uploads`
- CVAT Create Multi Tasks name template: `gold_v1__{{file_name}}`
- Future export folder: `temporal_module\data\gold_event_project\cvat_exports`
- Future import dry-run command: `python -m temporal_module.scripts.import_cvat_gold_annotations --dry-run`
- Future import command: `python -m temporal_module.scripts.import_cvat_gold_annotations`

## Status Counts

- ready_for_cvat: 25
- needs_original_pipeline: 0
- needs_temporal_sweep: 0
- blocked: 0

## Blocked Clips

- None

## One-To-One Mapping Confirmation

All ready clips are required to map as:

`clip_id` <-> `outputs/<clip_id>/preprocessed/clip.mp4` <-> `temporal_frames.csv` <-> `<clip_id>.mp4` <-> `gold_v1__<clip_id>.mp4` <-> future `cvat_export__<clip_id>.zip`
