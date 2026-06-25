# Gold Supervised Temporal Learning Status

## Status

Gold-supervised temporal learning has been executed for the numeric-feature
BiGRU path. CNN-LSTM training remains not ready because no visual frame-window
tensor dataset exists.

## CVAT Export Import

- ZIP exports found: 25
- Expected exports from `gold_clip_manifest.csv`: 25
- Invalid ZIP files: 0
- Canonical annotation file: `temporal_module/data/gold_event_project/annotations/gold_event_intervals.csv`
- Imported clips: 25
- Imported intervals: 226

## Gold Event Counts

- carry: 32
- pass: 118
- turnover: 11
- shot: 3
- uncertain: 62

The `uncertain` intervals are preserved in the canonical annotation file but are
not included as a supervised class in the current gold BiGRU dataset.

## Temporal Feature Readiness

- Gold clips with labels: 25
- Gold clips with processed pipeline outputs: 25
- Gold clips with `temporal_frames.csv`: 25
- Gold temporal readiness CSV: `temporal_module/data_gold/event_windows/gold_temporal_feature_readiness.csv`
- Gold temporal readiness report: `temporal_module/data_gold/event_windows/gold_temporal_feature_readiness.md`

## Gold Temporal Window Dataset

- Dataset: `temporal_module/data_gold/event_windows/gold_event_windows.npz`
- Shape: `(2150, 64, 68)`
- Windows: 2150
- Window size: 64 frames
- Feature count: 68
- Label region: 16 frames

Label distribution:

- background: 1106
- carry: 247
- pass: 731
- turnover: 58
- shot: 8

## Gold BiGRU

- Run directory: `temporal_module/runs_gold/event_bigru`
- Train windows: 1720
- Validation windows: 430
- Epochs: 40
- Selected-model validation accuracy: 0.5233
- Selected-model macro F1: 0.3710

Per-class F1:

- background: 0.6472
- carry: 0.1748
- pass: 0.4891
- turnover: 0.2581
- shot: 0.2857

These results are preliminary. The current split is a random window-level split,
not a clip-disjoint evaluation. The shot and turnover classes are especially
under-represented.

## Gold CNN-LSTM

- Output directory: `temporal_module/runs_gold/event_cnn_lstm`
- Training status: NOT READY
- Required input: `temporal_module/data_gold/frame_windows/gold_frame_windows.npz`
- Expected tensor format: `[windows, frames, height, width, channels]` or
  `[windows, frames, channels, height, width]`
- Reason: no visual frame-window dataset exists, and no safe builder is
  currently implemented.

## Report-Safe Interpretation

The gold BiGRU run demonstrates that the CVAT annotations can be connected to
the existing temporal feature pipeline and used for supervised temporal event
classification. The current metrics should be presented as preliminary pilot
results because the dataset is small, class-imbalanced, and evaluated with a
window-level split.
