# Gold Event BiGRU Training Summary

This model is trained on manually imported gold event intervals when available.

Evaluation protocol: clip-disjoint train/validation/test split. The validation split selects the best epoch; the test split is held out for final reporting.

- Dataset shape: (2150, 64, 68)
- Train windows: 1290
- Validation windows: 430
- Test windows: 430
- Epochs run: 40
- Selected-model validation accuracy: 0.4047
- Selected-model validation macro F1: 0.2367
- Held-out test accuracy: 0.4953
- Held-out test macro F1: 0.2689
- Last-epoch validation accuracy: 0.3512
- Last-epoch macro F1: 0.1826

## Label Distribution

- background: 1106
- carry: 247
- pass: 731
- turnover: 58
- shot: 8

## Split Warnings

- val split has no windows for: shot

## Test Per-Class Metrics

- background: precision 0.6087, recall 0.5241, F1 0.5632, support 187
- carry: precision 0.1733, recall 0.2600, F1 0.2080, support 50
- pass: precision 0.5763, recall 0.5698, F1 0.5730, support 179
- turnover: precision 0.0000, recall 0.0000, F1 0.0000, support 9
- shot: precision 0.0000, recall 0.0000, F1 0.0000, support 5