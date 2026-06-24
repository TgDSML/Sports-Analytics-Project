# CVAT Workflow

## Upload Location

CVAT upload folder:

```text
temporal_module/data/gold_event_project/cvat_uploads/
```

Create Multi Tasks task name template:

```text
gold_v1__{{file_name}}
```

One MP4 equals one task.

## Required Labels

```text
carry
pass
turnover
shot
uncertain
```

## Do Not Annotate

```text
background
player boxes
player IDs
team labels
ball boxes
full halves
8-second windows
```

## Annotation Format

- Use Rectangle + Track mode.
- Draw one tiny fixed marker rectangle at the top-left.
- Do not use automatic object tracking.
- Marker coordinates are ignored later.
- Only active track frames matter.
- Tracks may touch but must not overlap.
- Frame numbering is 0-based.
- Start and end frames are inclusive.
- Frame step must be 1.
- Segment size must cover the entire 30-second clip when possible.
- Use `uncertain` instead of guessing.

## Native Export Requirement

Each completed task must be exported in native `CVAT for video` format and named:

```text
cvat_export__<clip_id>.zip
```

All exports go to:

```text
temporal_module/data/gold_event_project/cvat_exports/
```

## Import Bridge

Dry-run import command:

```powershell
.\venv\Scripts\python.exe -m temporal_module.scripts.import_cvat_gold_annotations --dry-run
```

Final import command:

```powershell
.\venv\Scripts\python.exe -m temporal_module.scripts.import_cvat_gold_annotations
```

The importer writes:

```text
temporal_module/data/gold_event_project/annotations/gold_event_intervals.csv
```
