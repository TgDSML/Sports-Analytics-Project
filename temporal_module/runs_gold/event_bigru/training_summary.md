# Gold Event BiGRU Training Summary

This model is trained on manually imported gold event intervals when available.

- Dataset shape: (2150, 64, 68)
- Train windows: 1720
- Validation windows: 430
- Epochs run: 40
- Selected-model validation accuracy: 0.5233
- Selected-model macro F1: 0.3710
- Last-epoch validation accuracy: 0.5233
- Last-epoch macro F1: 0.2945

## Label Distribution

- background: 1106
- carry: 247
- pass: 731
- turnover: 58
- shot: 8

## Per-Class Metrics

- background: precision 0.6344, recall 0.6606, F1 0.6472
- carry: precision 0.1500, recall 0.2093, F1 0.1748
- pass: precision 0.5583, recall 0.4351, F1 0.4891
- turnover: precision 0.2222, recall 0.3077, F1 0.2581
- shot: precision 0.2000, recall 0.5000, F1 0.2857