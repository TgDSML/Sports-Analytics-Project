"""Gold-label temporal event model skeletons.

These models are intentionally small and dependency-light so the training
scripts can be wired before CVAT gold annotations are available.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class SequenceWindowDataset(Dataset):
    """Dataset for fixed-length numeric temporal feature windows."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class FrameWindowDataset(Dataset):
    """Dataset for fixed-length image/frame windows."""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        if self.x.ndim == 5 and self.x.shape[-1] in {1, 3}:
            self.x = self.x.permute(0, 1, 4, 2, 3)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class GoldEventBiGRU(nn.Module):
    """Bidirectional GRU classifier for gold-labelled temporal features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 96,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(x)
        pooled = outputs[:, -1, :]
        return self.classifier(self.dropout(pooled))


class SmallFrameEncoder(nn.Module):
    """Compact CNN encoder for frame-level visual prototypes."""

    def __init__(self, in_channels: int = 3, embedding_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(48, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.net(frames)


class GoldEventCNNLSTM(nn.Module):
    """Prototype CNN-LSTM classifier for gold-labelled video windows."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = SmallFrameEncoder(in_channels=in_channels, embedding_dim=embedding_dim)
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = x.shape
        flat = x.reshape(batch * steps, channels, height, width)
        embeddings = self.encoder(flat).reshape(batch, steps, -1)
        outputs, _ = self.lstm(embeddings)
        pooled = outputs[:, -1, :]
        return self.classifier(self.dropout(pooled))
