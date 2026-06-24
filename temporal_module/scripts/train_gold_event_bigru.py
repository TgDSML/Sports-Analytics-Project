"""Train a gold-supervised BiGRU on temporal feature windows."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gold_event_models import GoldEventBiGRU, SequenceWindowDataset  # noqa: E402


def main() -> int:
    args = parse_args()
    return train(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train gold-supervised temporal BiGRU")
    parser.add_argument("--data", type=Path, default=Path("temporal_module/data_gold/event_windows/gold_event_windows.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("temporal_module/runs_gold/event_bigru"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.data.exists():
        write_not_ready(args.output_dir, args.data, "Gold window dataset does not exist yet.")
        print(f"Gold dataset not found: {args.data}")
        return 0

    set_seed(args.seed)
    data = np.load(args.data, allow_pickle=True)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    label_names = [str(item) for item in data["label_names"]]
    feature_names = [str(item) for item in data["feature_names"]]
    if len(x) < 2 or len(set(y.tolist())) < 2:
        write_not_ready(args.output_dir, args.data, "Dataset has too few samples or classes for training.")
        print("Dataset has too few samples or classes for training.")
        return 0

    train_idx, val_idx = split_indices(len(x), args.seed)
    dataset = SequenceWindowDataset(x, y)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)
    model = GoldEventBiGRU(input_dim=x.shape[2], num_classes=len(label_names), hidden_dim=args.hidden_dim)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y[train_idx], len(label_names)))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_state = None
    best_macro_f1 = -1.0
    for epoch in range(1, max(1, args.epochs) + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, targets, preds, probs = evaluate(model, val_loader, criterion)
        metrics = classification_metrics(targets, preds, label_names)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics["overall"]}
        history.append(row)
        if row["macro_f1"] > best_macro_f1:
            best_macro_f1 = row["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_loss, targets, preds, probs = evaluate(model, val_loader, criterion)
    metrics = classification_metrics(targets, preds, label_names)
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": int(x.shape[2]),
            "num_classes": len(label_names),
            "label_names": label_names,
            "feature_names": feature_names,
            "hidden_dim": args.hidden_dim,
        },
        args.output_dir / "best_model.pt",
    )
    write_history(history, args.output_dir / "metrics.csv")
    write_confusion(metrics["confusion"], label_names, args.output_dir / "confusion_matrix.csv")
    write_predictions(val_idx, targets, preds, probs, label_names, args.output_dir / "predictions_val.csv")
    write_summary(args.output_dir, args, x, y, train_idx, val_idx, metrics, history)
    print(f"Wrote gold BiGRU run to {args.output_dir}")
    return 0


def run_epoch(model, loader, criterion, optimizer) -> float:
    model.train()
    total_loss, total = 0.0, 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(y_batch)
        total += len(y_batch)
    return total_loss / total if total else 0.0


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total = 0.0, 0
    targets, preds, probs = [], [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            prob = torch.softmax(logits, dim=1)
            total_loss += float(loss.item()) * len(y_batch)
            total += len(y_batch)
            targets.extend(y_batch.cpu().numpy().tolist())
            preds.extend(torch.argmax(prob, dim=1).cpu().numpy().tolist())
            probs.extend(prob.cpu().numpy().tolist())
    return total_loss / total if total else 0.0, targets, preds, probs


def classification_metrics(targets: list[int], preds: list[int], label_names: list[str]) -> dict:
    confusion = np.zeros((len(label_names), len(label_names)), dtype=int)
    for target, pred in zip(targets, preds):
        confusion[int(target), int(pred)] += 1
    per_class = {}
    f1_values = []
    for idx, label in enumerate(label_names):
        tp = int(confusion[idx, idx])
        fp = int(confusion[:, idx].sum() - tp)
        fn = int(confusion[idx, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1_values.append(f1)
    accuracy = float(np.trace(confusion) / confusion.sum()) if confusion.sum() else 0.0
    return {"overall": {"val_accuracy": accuracy, "macro_f1": float(np.mean(f1_values))}, "per_class": per_class, "confusion": confusion}


def split_indices(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    val_count = max(1, int(round(n * 0.2)))
    return np.asarray(indices[val_count:]), np.asarray(indices[:val_count])


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = Counter(int(item) for item in y.tolist())
    values = np.asarray([1.0 / max(1, counts.get(idx, 0)) for idx in range(num_classes)], dtype=np.float32)
    values = values / values.mean() if values.mean() > 0 else values
    return torch.tensor(values, dtype=torch.float32)


def write_history(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy", "macro_f1"])
        writer.writeheader()
        writer.writerows(rows)


def write_confusion(confusion: np.ndarray, label_names: list[str], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["actual\\predicted", *label_names])
        for label, row in zip(label_names, confusion.tolist()):
            writer.writerow([label, *row])


def write_predictions(indices, targets, preds, probs, label_names, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        fields = ["window_id", "gold_label", "predicted_label", "confidence", *[f"prob_{label}" for label in label_names]]
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for window_id, target, pred, prob in zip(indices, targets, preds, probs):
            row = {
                "window_id": int(window_id),
                "gold_label": label_names[int(target)],
                "predicted_label": label_names[int(pred)],
                "confidence": max(prob),
            }
            row.update({f"prob_{label}": value for label, value in zip(label_names, prob)})
            writer.writerow(row)


def write_summary(output_dir: Path, args: argparse.Namespace, x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, metrics: dict, history: list[dict]) -> None:
    counts = Counter(int(item) for item in y.tolist())
    label_names = [str(item) for item in np.load(args.data, allow_pickle=True)["label_names"]]
    final = history[-1] if history else {}
    selected = metrics.get("overall", {}) if isinstance(metrics, dict) else {}
    lines = [
        "# Gold Event BiGRU Training Summary",
        "",
        "This model is trained on manually imported gold event intervals when available.",
        "",
        f"- Dataset shape: {tuple(x.shape)}",
        f"- Train windows: {len(train_idx)}",
        f"- Validation windows: {len(val_idx)}",
        f"- Epochs run: {len(history)}",
        f"- Selected-model validation accuracy: {selected.get('val_accuracy', 0.0):.4f}",
        f"- Selected-model macro F1: {selected.get('macro_f1', 0.0):.4f}",
        f"- Last-epoch validation accuracy: {final.get('val_accuracy', 0.0):.4f}",
        f"- Last-epoch macro F1: {final.get('macro_f1', 0.0):.4f}",
        "",
        "## Label Distribution",
        "",
    ]
    lines.extend(f"- {label}: {counts.get(idx, 0)}" for idx, label in enumerate(label_names))
    lines.extend(["", "## Per-Class Metrics", ""])
    for label in label_names:
        item = metrics["per_class"].get(label, {})
        lines.append(f"- {label}: precision {item.get('precision', 0.0):.4f}, recall {item.get('recall', 0.0):.4f}, F1 {item.get('f1', 0.0):.4f}")
    (output_dir / "training_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "metrics.json").write_text(
        json.dumps({"selected_model": selected, "last_epoch": final, "per_class": metrics["per_class"]}, indent=2),
        encoding="utf-8",
    )


def write_not_ready(output_dir: Path, data_path: Path, reason: str) -> None:
    lines = [
        "# Gold Event BiGRU Training Summary",
        "",
        "Status: NOT READY",
        "",
        f"Dataset path: `{data_path}`",
        f"Reason: {reason}",
        "",
        "Build `gold_event_windows.npz` after CVAT annotation import, then rerun training.",
    ]
    (output_dir / "training_summary.md").write_text("\n".join(lines), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    raise SystemExit(main())
