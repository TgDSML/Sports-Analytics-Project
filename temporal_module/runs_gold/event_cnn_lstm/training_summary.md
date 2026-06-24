# Gold Event CNN-LSTM Training Summary

This model is a visual-frame prototype for manually labelled event windows.

- Dataset shape: (1075, 16, 112, 112, 3)
- Train windows: 860
- Validation windows: 215
- Epochs run: 20
- Final validation accuracy: 0.6000
- Final macro F1: 0.3076

## Label Distribution

- background: 552
- carry: 124
- pass: 366
- turnover: 28
- shot: 5

## Per-Class Metrics

- background: precision 0.7789, recall 0.6549, F1 0.7115
- carry: precision 0.3333, recall 0.2353, F1 0.2759
- pass: precision 0.5463, recall 0.7662, F1 0.6378
- turnover: precision 0.0000, recall 0.0000, F1 0.0000
- shot: precision 0.0000, recall 0.0000, F1 0.0000