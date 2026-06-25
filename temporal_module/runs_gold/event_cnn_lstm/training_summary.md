# Gold Event CNN-LSTM Training Summary

This model is a visual-frame prototype for manually labelled event windows.

Evaluation protocol: clip-disjoint train/validation/test split. The validation split selects the best epoch; the test split is held out for final reporting.

- Dataset shape: (1075, 16, 112, 112, 3)
- Train windows: 645
- Validation windows: 215
- Test windows: 215
- Epochs run: 20
- Selected-model validation accuracy: 0.4977
- Selected-model validation macro F1: 0.2185
- Held-out test accuracy: 0.5814
- Held-out test macro F1: 0.2424
- Last-epoch validation accuracy: 0.3860
- Last-epoch macro F1: 0.1904

## Label Distribution

- background: 552
- carry: 124
- pass: 366
- turnover: 28
- shot: 5

## Split Warnings

- val split has no windows for: shot

## Test Per-Class Metrics

- background: precision 0.9722, recall 0.3763, F1 0.5426, support 93
- carry: precision 0.0000, recall 0.0000, F1 0.0000, support 25
- pass: precision 0.5028, recall 1.0000, F1 0.6691, support 90
- turnover: precision 0.0000, recall 0.0000, F1 0.0000, support 4
- shot: precision 0.0000, recall 0.0000, F1 0.0000, support 3