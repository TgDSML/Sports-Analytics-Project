# Gold vs Weak BiGRU Comparison

## Summary

The weak-label and gold-label BiGRU runs are not directly equivalent. They use
different label sources, different class definitions, different dataset sizes,
and different feature dimensions.

## Weak-Label Additive BiGRU

- Dataset: `temporal_module/data_additive/event_windows.npz`
- Windows: 78
- Window shape: `(78, 64, 12)`
- Classes: background, carry, pass, interception
- Label source: heuristic filtered event candidates
- Validation accuracy: 0.8125
- Macro F1: 0.6241
- Important limitation: the pass class had 0 windows in the generated dataset.

## Gold-Supervised BiGRU

- Dataset: `temporal_module/data_gold/event_windows/gold_event_windows.npz`
- Windows: 2150
- Window shape: `(2150, 64, 68)`
- Classes: background, carry, pass, turnover, shot
- Label source: manually imported CVAT event intervals
- Selected-model validation accuracy: 0.5233
- Selected-model macro F1: 0.3710

## Why The Metrics Differ

The weak-label run measures agreement with heuristic labels on a small dataset.
The gold run uses human-reviewed labels and a larger feature set, but it is more
challenging because the class distribution is imbalanced and includes sparse
classes such as shot and turnover.

The weak score should not be interpreted as stronger event recognition. It is a
prototype sanity check. The gold run is the more meaningful supervised learning
path, but the current result should still be treated as preliminary because it
uses a random window-level split rather than a clip-disjoint evaluation.
