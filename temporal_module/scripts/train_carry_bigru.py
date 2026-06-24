"""Train a binary BiGRU carry/background classifier on prepared windows."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as error:  # pragma: no cover - handled at runtime for clear CLI failure.
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_IMPORT_ERROR = error
else:
    TORCH_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import PROJECT_ROOT, ensure_output_parent, reject_outputs_path, utc_now_iso  # noqa: E402


RUN_NAME_TEMPLATE = "carry_bigru_seed{seed}"
EPS_STD = 1e-8


@dataclass
class RunConfig:
    dataset_root: str
    runs_root: str
    seed: int
    hidden_size: int
    num_layers: int
    dropout: float
    batch_size: int
    learning_rate: float
    max_epochs: int
    patience: int


class CarryBiGRU(nn.Module):
    """BiGRU classifier using concatenated final forward/backward states."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0 if num_layers == 1 else dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        forward_last = hidden[-2]
        backward_last = hidden[-1]
        representation = torch.cat([forward_last, backward_last], dim=1)
        return self.classifier(self.dropout(representation))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train carry/background BiGRU on temporal windows.")
    parser.add_argument("--dataset-root", default=str(Path("temporal_module") / "data" / "datasets"))
    parser.add_argument("--runs-root", default=str(Path("temporal_module") / "runs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    return parser.parse_args()


def enforce_runs_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = (PROJECT_ROOT / "temporal_module" / "runs").resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as error:
        raise ValueError(f"Training outputs must be under {allowed}: {resolved}") from error
    reject_outputs_path(resolved)
    return resolved


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_inputs(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict]:
    npz_path = dataset_root / "carry_background_windows.npz"
    split_path = dataset_root / "carry_background_split.csv"
    metadata_path = dataset_root / "carry_background_windows_metadata.json"
    for path in [npz_path, split_path, metadata_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required dataset input not found: {path}")

    data = np.load(npz_path, allow_pickle=True)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    clip_ids = data["clip_ids"].astype(str)
    center_frames = data["center_frames"].astype(np.int64)
    split = pd.read_csv(split_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    required_split_columns = {"clip_id", "split"}
    missing = required_split_columns - set(split.columns)
    if missing:
        raise ValueError(f"{split_path} missing column(s): {', '.join(sorted(missing))}")
    if len(split) != len(y):
        raise ValueError(f"Row-count mismatch: NPZ has {len(y)} rows, split CSV has {len(split)} rows")
    split_clip_ids = split["clip_id"].astype(str).to_numpy()
    if not np.array_equal(clip_ids, split_clip_ids):
        mismatch_index = int(np.flatnonzero(clip_ids != split_clip_ids)[0])
        raise ValueError(
            "NPZ clip_ids do not match split CSV clip_id order at row "
            f"{mismatch_index}: {clip_ids[mismatch_index]} != {split_clip_ids[mismatch_index]}"
        )
    return x, y, clip_ids, center_frames, split, metadata


def split_indices(split: pd.DataFrame) -> dict[str, np.ndarray]:
    split_values = split["split"].astype(str)
    return {
        "train": np.flatnonzero(split_values == "train"),
        "validation": np.flatnonzero(split_values == "validation"),
        "test": np.flatnonzero(split_values == "test"),
        "excluded": np.flatnonzero(split_values == "excluded"),
    }


def fit_scaler(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std[np.abs(std) < EPS_STD] = 1.0
    return mean, std


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


def class_counts(y: np.ndarray) -> dict[str, int]:
    return {"background": int((y == 0).sum()), "carry": int((y == 1).sum())}


def class_weights_from_train(y_train: np.ndarray) -> np.ndarray:
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    total = float(counts.sum())
    weights = np.ones(2, dtype=np.float32)
    for class_id in range(2):
        if counts[class_id] > 0:
            weights[class_id] = total / (2.0 * counts[class_id])
    return weights


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.int64)))
    generator = torch.Generator()
    generator.manual_seed(0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, dict, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    predictions = []
    labels = []
    probabilities = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += float(loss.item()) * len(batch_y)
            total_items += int(len(batch_y))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probabilities.append(probs)
            predictions.append(probs.argmax(axis=1))
            labels.append(batch_y.cpu().numpy())
    y_true = np.concatenate(labels) if labels else np.empty((0,), dtype=np.int64)
    y_pred = np.concatenate(predictions) if predictions else np.empty((0,), dtype=np.int64)
    probs = np.concatenate(probabilities) if probabilities else np.empty((0, 2), dtype=np.float32)
    avg_loss = total_loss / total_items if total_items else 0.0
    return avg_loss, compute_metrics(y_true, y_pred), y_pred, probs


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(batch_y)
        total_items += int(len(batch_y))
    return total_loss / total_items if total_items else 0.0


def save_checkpoint(path: Path, model: nn.Module, epoch: int, validation_loss: float, validation_metrics: dict) -> None:
    ensure_output_parent(path)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "validation_loss": validation_loss,
            "validation_metrics": validation_metrics,
        },
        path,
    )


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_output_parent(path)
    fieldnames = [
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_accuracy",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(
    path: Path,
    sample_indices: np.ndarray,
    clip_ids: np.ndarray,
    center_frames: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    ensure_output_parent(path)
    fieldnames = [
        "sample_index",
        "clip_id",
        "center_frame",
        "true_label",
        "predicted_label",
        "background_probability",
        "carry_probability",
        "split",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for idx, true_label, pred_label, probs in zip(sample_indices, y_true, y_pred, probabilities):
            writer.writerow(
                {
                    "sample_index": int(idx),
                    "clip_id": str(clip_ids[idx]),
                    "center_frame": int(center_frames[idx]),
                    "true_label": int(true_label),
                    "predicted_label": int(pred_label),
                    "background_probability": float(probs[0]),
                    "carry_probability": float(probs[1]),
                    "split": "test",
                }
            )


def main() -> int:
    args = parse_args()
    try:
        if TORCH_IMPORT_ERROR is not None:
            raise ImportError(f"PyTorch is required but could not be imported: {TORCH_IMPORT_ERROR}")
        config = RunConfig(
            dataset_root=args.dataset_root,
            runs_root=args.runs_root,
            seed=args.seed,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        set_seed(config.seed)

        dataset_root = Path(config.dataset_root)
        runs_root = enforce_runs_root(Path(config.runs_root))
        run_dir = enforce_runs_root(runs_root / RUN_NAME_TEMPLATE.format(seed=config.seed))
        run_dir.mkdir(parents=True, exist_ok=True)

        x_raw, y, clip_ids, center_frames, split, dataset_metadata = load_inputs(dataset_root)
        indices = split_indices(split)
        if len(indices["train"]) == 0 or len(indices["validation"]) == 0 or len(indices["test"]) == 0:
            raise ValueError("Train, validation, and test splits must all contain at least one window.")

        feature_names = dataset_metadata.get("feature_names", [])
        if len(feature_names) != int(x_raw.shape[2]):
            raise ValueError(
                f"Feature-name count {len(feature_names)} does not match X feature dimension {x_raw.shape[2]}"
            )

        mean, std = fit_scaler(x_raw[indices["train"]])
        x = normalize(x_raw, mean, std)
        train_x, train_y = x[indices["train"]], y[indices["train"]]
        val_x, val_y = x[indices["validation"]], y[indices["validation"]]
        test_x, test_y = x[indices["test"]], y[indices["test"]]

        class_weights = class_weights_from_train(train_y)
        device = torch.device("cpu")
        model = CarryBiGRU(
            input_size=int(x.shape[2]),
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        train_loader = make_loader(train_x, train_y, config.batch_size, shuffle=True)
        val_loader = make_loader(val_x, val_y, config.batch_size, shuffle=False)
        test_loader = make_loader(test_x, test_y, config.batch_size, shuffle=False)

        best_epoch = 0
        best_val_f1 = -1.0
        best_val_loss = float("inf")
        best_val_metrics: dict[str, Any] = {}
        best_train_metrics: dict[str, Any] = {}
        patience_used = 0
        history_rows = []

        for epoch in range(1, config.max_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
            train_eval_loss, train_metrics, _, _ = evaluate(model, train_loader, criterion, device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_precision": val_metrics["precision"],
                "validation_recall": val_metrics["recall"],
                "validation_f1": val_metrics["f1"],
                "validation_accuracy": val_metrics["accuracy"],
            }
            history_rows.append(row)
            print(
                f"epoch={epoch} train_loss={train_loss:.6f} validation_loss={val_loss:.6f} "
                f"validation_precision={val_metrics['precision']:.4f} "
                f"validation_recall={val_metrics['recall']:.4f} "
                f"validation_f1={val_metrics['f1']:.4f} "
                f"validation_accuracy={val_metrics['accuracy']:.4f}"
            )

            improved = (val_metrics["f1"] > best_val_f1) or (
                val_metrics["f1"] == best_val_f1 and val_loss < best_val_loss
            )
            if improved:
                best_epoch = epoch
                best_val_f1 = float(val_metrics["f1"])
                best_val_loss = float(val_loss)
                best_val_metrics = dict(val_metrics)
                best_train_metrics = dict(train_metrics)
                save_checkpoint(run_dir / "best_model.pt", model, epoch, val_loss, val_metrics)
                patience_used = 0
            else:
                patience_used += 1

            save_checkpoint(run_dir / "last_model.pt", model, epoch, val_loss, val_metrics)
            if patience_used >= config.patience:
                print(f"Early stopping after epoch {epoch}; patience={config.patience}")
                break

        checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss, test_metrics, test_pred, test_probs = evaluate(model, test_loader, criterion, device)

        write_history(run_dir / "training_history.csv", history_rows)
        write_predictions(
            path=run_dir / "test_predictions.csv",
            sample_indices=indices["test"],
            clip_ids=clip_ids,
            center_frames=center_frames,
            y_true=test_y,
            y_pred=test_pred,
            probabilities=test_probs,
        )

        feature_scaler = {
            "build_timestamp": utc_now_iso(),
            "feature_names": feature_names,
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "fit_split": "train",
            "near_zero_std_replacement": 1.0,
        }
        feature_scaler_path = ensure_output_parent(run_dir / "feature_scaler.json")
        feature_scaler_path.write_text(json.dumps(feature_scaler, indent=2), encoding="utf-8")
        run_config_path = ensure_output_parent(run_dir / "run_config.json")
        run_config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

        split_counts = {
            name: {
                "samples": int(len(idx)),
                "class_counts": class_counts(y[idx]),
            }
            for name, idx in indices.items()
        }
        metrics_payload = {
            "warning": "Results use weak labels and a very small held-out test set; do not treat as ground-truth event performance.",
            "selected_best_epoch": int(best_epoch),
            "class_weights": {
                "background": float(class_weights[0]),
                "carry": float(class_weights[1]),
            },
            "sample_and_class_counts": split_counts,
            "train": best_train_metrics,
            "validation": best_val_metrics,
            "test": test_metrics,
            "test_loss": float(test_loss),
            "data_shapes": {
                "X": list(x_raw.shape),
                "y": list(y.shape),
            },
        }
        metrics_path = ensure_output_parent(run_dir / "metrics.json")
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

        print("Final summary:")
        print(f"Selected device: {device}")
        print(f"Data shapes: X={x_raw.shape}, y={y.shape}")
        print(f"Split counts: {split_counts}")
        print(f"Training class weights: background={class_weights[0]:.6f}, carry={class_weights[1]:.6f}")
        print(f"Best epoch: {best_epoch}")
        print(f"Validation metrics at best epoch: {best_val_metrics}")
        print(f"Final test metrics: {test_metrics}")
        print(f"Output directory: {run_dir}")
        return 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
