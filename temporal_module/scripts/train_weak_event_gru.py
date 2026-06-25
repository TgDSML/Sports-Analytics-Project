"""Train a pilot weakly supervised GRU event classifier."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except ImportError as error:  # pragma: no cover - exercised in missing dependency environments.
    raise SystemExit("PyTorch is required for training. Install project dependencies with: python -m pip install -r requirements.txt") from error

try:
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
except ImportError as error:  # pragma: no cover
    raise SystemExit("scikit-learn is required for evaluation. Install project dependencies with: python -m pip install -r requirements.txt") from error


CLASSES = ["background", "carry", "pass", "turnover", "shot"]
CLASS_TO_ID = {class_name: idx for idx, class_name in enumerate(CLASSES)}
DEFAULT_DATASET_DIR = Path("temporal_module") / "data" / "modeling" / "weak_event_gru"
DEFAULT_MODEL_DIR = Path("temporal_module") / "models" / "weak_event_gru"
DEFAULT_REPORT_DIR = Path("temporal_module") / "reports" / "weak_event_gru"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a weak-label GRU baseline for temporal events.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--derived-root", default=str(Path("temporal_module") / "data" / "derived"))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--no-bidirectional", action="store_true")
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--sampling", choices=["none", "weighted"], default="none")
    parser.add_argument("--class-weighting", choices=["none", "inverse_frequency"], default="inverse_frequency")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_windows(dataset_dir: Path) -> pd.DataFrame:
    path = dataset_dir / "weak_event_windows.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing weak window manifest: {path}")
    windows = pd.read_csv(path)
    required = {"clip_id", "split", "start_frame", "end_frame", "target_class", "selected_after_balancing"}
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"weak_event_windows.csv missing required columns: {', '.join(sorted(missing))}")
    windows = windows[pd.to_numeric(windows["selected_after_balancing"], errors="coerce").fillna(0).astype(int) == 1]
    windows = windows[windows["target_class"].isin(CLASSES)].copy()
    if windows.empty:
        raise ValueError("No selected weak windows available for training.")
    return windows.reset_index(drop=True)


def load_frames(derived_root: Path, clip_ids: list[str], feature_columns: list[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for clip_id in clip_ids:
        path = derived_root / clip_id / "temporal_frames.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing temporal frames for clip {clip_id}: {path}")
        df = pd.read_csv(path, low_memory=False)
        missing = [column for column in feature_columns if column not in df.columns]
        if missing:
            raise ValueError(f"{clip_id} missing feature column(s): {', '.join(missing)}")
        if "frame" not in df.columns:
            raise ValueError(f"{clip_id} temporal_frames.csv has no frame column.")
        df = df[["frame", *feature_columns]].copy()
        df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
        df = df.dropna(subset=["frame"])
        df["frame"] = df["frame"].astype(int)
        for column in feature_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        frames[clip_id] = df.set_index("frame").sort_index()
    return frames


def compute_normalization(
    windows: pd.DataFrame,
    frames_by_clip: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> tuple[pd.Series, pd.Series]:
    train_windows = windows[windows["split"] == "train"]
    pieces: list[pd.DataFrame] = []
    for row in train_windows.itertuples(index=False):
        frame_range = range(int(row.start_frame), int(row.end_frame) + 1)
        seq = frames_by_clip[row.clip_id].reindex(frame_range)[feature_columns]
        pieces.append(seq)
    if not pieces:
        raise ValueError("No train windows available for normalization.")
    train_values = pd.concat(pieces, axis=0)
    means = train_values.mean(axis=0, skipna=True).fillna(0.0)
    stds = train_values.std(axis=0, skipna=True).replace(0, np.nan).fillna(1.0)
    return means, stds


class WeakEventWindowDataset(Dataset):
    def __init__(
        self,
        windows: pd.DataFrame,
        frames_by_clip: dict[str, pd.DataFrame],
        feature_columns: list[str],
        means: pd.Series,
        stds: pd.Series,
    ) -> None:
        self.windows = windows.reset_index(drop=True)
        self.frames_by_clip = frames_by_clip
        self.feature_columns = feature_columns
        self.means = means
        self.stds = stds
        self.sequence_length = int((self.windows["end_frame"] - self.windows["start_frame"] + 1).max())

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.windows.iloc[idx]
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        frame_range = range(start, end + 1)
        raw = self.frames_by_clip[str(row["clip_id"])].reindex(frame_range)[self.feature_columns]
        valid = (~raw.isna().all(axis=1)).to_numpy(dtype=np.float32)
        normalized = ((raw.fillna(self.means) - self.means) / self.stds).fillna(0.0).to_numpy(dtype=np.float32)
        length = normalized.shape[0]
        if length < self.sequence_length:
            pad = self.sequence_length - length
            normalized = np.pad(normalized, ((0, pad), (0, 0)), mode="constant")
            valid = np.pad(valid, (0, pad), mode="constant")
        return {
            "x": torch.from_numpy(normalized),
            "mask": torch.from_numpy(valid),
            "y": torch.tensor(CLASS_TO_ID[str(row["target_class"])], dtype=torch.long),
            "clip_id": str(row["clip_id"]),
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "target_class": str(row["target_class"]),
        }


class GRUClassifier(nn.Module):
    def __init__(
        self,
        feature_count: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        class_count: int,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_count,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        output_size = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_size, class_count)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output, _hidden = self.gru(x)
        lengths = mask.sum(dim=1).clamp(min=1).long()
        batch_index = torch.arange(output.size(0), device=output.device)
        last = output[batch_index, lengths - 1]
        return self.classifier(self.dropout(last))


def train_class_distribution(train_windows: pd.DataFrame) -> dict[str, int]:
    counts = Counter(train_windows["target_class"])
    return {class_name: int(counts.get(class_name, 0)) for class_name in CLASSES}


def class_weights(
    train_windows: pd.DataFrame,
    mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    if mode == "none":
        weights = torch.ones(len(CLASSES), dtype=torch.float32, device=device)
        return weights, {class_name: 1.0 for class_name in CLASSES}

    if mode != "inverse_frequency":
        raise ValueError(f"Unsupported class weighting mode: {mode}")

    counts = train_class_distribution(train_windows)
    weights = []
    total = sum(counts.values())
    for class_name in CLASSES:
        count = counts.get(class_name, 0)
        weights.append(total / (len(CLASSES) * count) if count else 0.0)
    nonzero = [weight for weight in weights if weight > 0]
    average = float(np.mean(nonzero)) if nonzero else 1.0
    normalized = [weight / average if average > 0 else weight for weight in weights]
    tensor = torch.tensor(normalized, dtype=torch.float32, device=device)
    weight_map = {class_name: float(normalized[idx]) for idx, class_name in enumerate(CLASSES)}
    return tensor, weight_map


class FocalLoss(nn.Module):
    def __init__(self, class_weight: torch.Tensor, gamma: float) -> None:
        super().__init__()
        self.register_buffer("class_weight", class_weight.detach().clone())
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        unweighted_ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        weighted_ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.class_weight,
            reduction="none",
        )
        pt = torch.exp(-unweighted_ce).clamp(min=1e-8, max=1.0)
        focal_factor = (1.0 - pt).pow(self.gamma)
        return (focal_factor * weighted_ce).mean()


def make_criterion(loss_mode: str, class_weight: torch.Tensor, focal_gamma: float) -> nn.Module:
    if loss_mode == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weight)
    if loss_mode == "focal":
        return FocalLoss(class_weight=class_weight, gamma=focal_gamma)
    raise ValueError(f"Unsupported loss mode: {loss_mode}")


def make_train_loader(
    dataset: WeakEventWindowDataset,
    batch_size: int,
    sampling: str,
    class_weight_map: dict[str, float],
    seed: int,
) -> DataLoader:
    if sampling == "none":
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if sampling != "weighted":
        raise ValueError(f"Unsupported sampling mode: {sampling}")
    sample_weights = [
        class_weight_map[str(row["target_class"])]
        for _, row in dataset.windows.iterrows()
    ]
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.set_grad_enabled(training):
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y"].to(device)
            logits = model(x, mask)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
    macro_f1 = f1_score(y_true, y_pred, labels=list(range(len(CLASSES))), average="macro", zero_division=0)
    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "macro_f1": float(macro_f1),
        "accuracy": accuracy,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def predict_windows(model: nn.Module, loader: DataLoader, device: torch.device) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(device), batch["mask"].to(device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            for idx, pred_id in enumerate(predictions):
                rows.append(
                    {
                        "clip_id": batch["clip_id"][idx],
                        "start_frame": int(batch["start_frame"][idx]),
                        "end_frame": int(batch["end_frame"][idx]),
                        "target_class": batch["target_class"][idx],
                        "predicted_class": CLASSES[int(pred_id)],
                        "predicted_probability": round(float(probabilities[idx, pred_id]), 6),
                    }
                )
    return rows


def metrics_rows(y_true: list[int], y_pred: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        zero_division=0,
    )
    per_class = []
    for idx, class_name in enumerate(CLASSES):
        per_class.append(
            {
                "target_class": class_name,
                "precision": round(float(precision[idx]), 6),
                "recall": round(float(recall[idx]), 6),
                "f1": round(float(f1[idx]), 6),
                "support": int(support[idx]),
            }
        )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    cm_rows = []
    for idx, class_name in enumerate(CLASSES):
        row = {"actual_class": class_name}
        for pred_idx, pred_name in enumerate(CLASSES):
            row[f"predicted_{pred_name}"] = int(cm[idx, pred_idx])
        cm_rows.append(row)
    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(CLASSES))), average="macro", zero_division=0)),
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))) if y_true else 0.0,
        "window_count": len(y_true),
    }
    return per_class, metrics, cm_rows


def make_report(
    path: Path,
    config: dict[str, Any],
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    class_counts: dict[str, dict[str, int]],
    train_distribution: dict[str, int],
    class_weight_map: dict[str, float],
    warnings: list[str],
) -> None:
    lines = [
        "# Weak Event GRU Pilot Report",
        "",
        "This experiment uses weak labels generated from the current temporal candidate pipeline.",
        "It is a pilot-scale evaluation on the available processable clips and does not establish final multiclass generalization.",
        "No manual labels, review manifests, manual overrides, duel labels, or goal labels were used.",
        "Validation and test loaders are sequential and are not resampled.",
        "",
        "## Configuration",
        "",
        f"- Created: {config['created_at']}",
        f"- Loss: {config['loss']}",
        f"- Focal gamma: {config['focal_gamma']}",
        f"- Sampling: {config['sampling']}",
        f"- Class weighting: {config['class_weighting']}",
        f"- Dataset: {config['dataset_dir']}",
        f"- Feature count: {config['feature_count']}",
        f"- Classes: {', '.join(CLASSES)}",
        f"- Bidirectional GRU: {config['bidirectional']}",
        "",
        "## Train Class Distribution",
        "",
        ", ".join(f"{class_name}={train_distribution.get(class_name, 0)}" for class_name in CLASSES),
        "",
        "## Class Weights",
        "",
        ", ".join(f"{class_name}={class_weight_map.get(class_name, 0.0):.6f}" for class_name in CLASSES),
        "",
        "## Metrics",
        "",
        f"- Best validation macro-F1: {config['best_validation_macro_f1']:.6f}",
        f"- Final validation macro-F1: {val_metrics['macro_f1']:.6f}",
        f"- Test macro-F1: {test_metrics['macro_f1']:.6f}",
        f"- Test accuracy: {test_metrics['accuracy']:.6f}",
        "",
        "## Selected Window Counts By Split",
        "",
    ]
    for split, counts in class_counts.items():
        text = ", ".join(f"{class_name}={counts.get(class_name, 0)}" for class_name in CLASSES)
        lines.append(f"- {split}: {text}")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    dataset_dir = Path(args.dataset_dir)
    derived_root = Path(args.derived_root)
    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    schema = read_json(dataset_dir / "weak_event_feature_schema.json")
    feature_columns = list(schema.get("selected_feature_columns", []))
    if not feature_columns:
        raise SystemExit("Feature schema contains no selected feature columns.")
    windows = load_windows(dataset_dir)
    split_counts = windows.groupby("split")["clip_id"].nunique().to_dict()
    missing_splits = [split for split in ["train", "val", "test"] if split not in split_counts]
    if missing_splits:
        raise SystemExit(f"Missing required split(s): {', '.join(missing_splits)}")
    train_classes = set(windows.loc[windows["split"] == "train", "target_class"])
    missing_train = [class_name for class_name in CLASSES if class_name not in train_classes]
    if missing_train:
        raise SystemExit(f"Required class(es) absent from train split: {', '.join(missing_train)}")
    warnings = []
    for split in ["val", "test"]:
        split_classes = set(windows.loc[windows["split"] == split, "target_class"])
        missing = [class_name for class_name in CLASSES if class_name not in split_classes]
        if missing:
            warnings.append(f"{split} split has no windows for: {', '.join(missing)}")

    frames_by_clip = load_frames(derived_root, sorted(windows["clip_id"].unique()), feature_columns)
    means, stds = compute_normalization(windows, frames_by_clip, feature_columns)
    normalization = {
        "feature_columns": feature_columns,
        "mean": {column: float(means[column]) for column in feature_columns},
        "std": {column: float(stds[column]) for column in feature_columns},
    }
    write_json(model_dir / "normalization.json", normalization)

    datasets = {
        split: WeakEventWindowDataset(
            windows[windows["split"] == split],
            frames_by_clip,
            feature_columns,
            means,
            stds,
        )
        for split in ["train", "val", "test"]
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_windows = windows[windows["split"] == "train"]
    train_distribution = train_class_distribution(train_windows)
    class_weight_tensor, class_weight_map = class_weights(train_windows, args.class_weighting, device)
    loaders = {
        "train": make_train_loader(
            datasets["train"],
            batch_size=args.batch_size,
            sampling=args.sampling,
            class_weight_map=class_weight_map,
            seed=args.seed,
        ),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False),
    }
    bidirectional = not args.no_bidirectional
    model = GRUClassifier(
        feature_count=len(feature_columns),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=bidirectional,
        class_count=len(CLASSES),
    ).to(device)
    criterion = make_criterion(args.loss, class_weight_tensor, args.focal_gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, Any]] = []
    best_val = -1.0
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_metrics = run_epoch(model, loaders["val"], criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_macro_f1": round(train_metrics["macro_f1"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "validation_loss": round(val_metrics["loss"], 6),
            "validation_macro_f1": round(val_metrics["macro_f1"], 6),
            "validation_accuracy": round(val_metrics["accuracy"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_macro_f1": round(val_metrics["macro_f1"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
        }
        history.append(row)
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_eval = run_epoch(model, loaders["val"], criterion, device)
    test_eval = run_epoch(model, loaders["test"], criterion, device)
    val_per_class, val_summary, _val_cm = metrics_rows(val_eval["y_true"], val_eval["y_pred"])
    test_per_class, test_summary, test_cm = metrics_rows(test_eval["y_true"], test_eval["y_pred"])
    test_predictions = predict_windows(model, loaders["test"], device)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
            "feature_columns": feature_columns,
            "model_config": {
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "bidirectional": bidirectional,
            },
        },
        model_dir / "weak_event_gru.pt",
    )
    experiment_config = {
        "created_at": utc_now(),
        "dataset_dir": str(dataset_dir),
        "derived_root": str(derived_root),
        "classes": CLASSES,
        "feature_count": len(feature_columns),
        "seed": args.seed,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma,
        "sampling": args.sampling,
        "class_weighting": args.class_weighting,
        "class_weights": class_weight_map,
        "train_class_distribution": train_distribution,
        "epochs_requested": args.epochs,
        "epochs_run": len(history),
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "bidirectional": bidirectional,
        "best_validation_macro_f1": float(best_val),
        "notes": "Pilot GRU trained on weak labels from generated candidate artifacts only; not production-ready.",
    }
    write_json(model_dir / "experiment_config.json", experiment_config)
    write_json(report_dir / "experiment_config.json", experiment_config)
    write_csv(report_dir / "training_history.csv", history, list(history[0].keys()) if history else ["epoch"])
    metrics_payload = {
        "validation": val_summary,
        "test": test_summary,
        "class_weights": class_weight_map,
        "train_class_distribution": train_distribution,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma,
        "sampling": args.sampling,
        "class_weighting": args.class_weighting,
        "warnings": warnings,
        "weak_label_notice": "Labels are weak labels generated from current candidate pipeline outputs; no manual labels were used.",
    }
    write_json(report_dir / "evaluation_metrics.json", metrics_payload)
    per_class_rows = [
        {"split": "validation", **row} for row in val_per_class
    ] + [
        {"split": "test", **row} for row in test_per_class
    ]
    write_csv(report_dir / "per_class_metrics.csv", per_class_rows, ["split", "target_class", "precision", "recall", "f1", "support"])
    write_csv(report_dir / "confusion_matrix.csv", test_cm, ["actual_class", *[f"predicted_{name}" for name in CLASSES]])
    write_csv(
        report_dir / "prediction_windows.csv",
        test_predictions,
        ["clip_id", "start_frame", "end_frame", "target_class", "predicted_class", "predicted_probability"],
    )

    class_counts: dict[str, dict[str, int]] = {}
    for split in ["train", "val", "test"]:
        counts = Counter(windows.loc[windows["split"] == split, "target_class"])
        class_counts[split] = {class_name: int(counts.get(class_name, 0)) for class_name in CLASSES}
    make_report(
        report_dir / "weak_event_gru_report.md",
        experiment_config,
        val_summary,
        test_summary,
        class_counts,
        train_distribution,
        class_weight_map,
        warnings,
    )
    print(f"Epochs run: {len(history)}")
    print(f"Best validation macro-F1: {best_val:.6f}")
    print(f"Final validation macro-F1: {val_summary['macro_f1']:.6f}")
    print(f"Test macro-F1: {test_summary['macro_f1']:.6f}")
    print(f"Test accuracy: {test_summary['accuracy']:.6f}")
    print(
        "Class weights: "
        + ", ".join(f"{class_name}={class_weight_map[class_name]:.6f}" for class_name in CLASSES)
    )
    print(f"Reports written under: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
