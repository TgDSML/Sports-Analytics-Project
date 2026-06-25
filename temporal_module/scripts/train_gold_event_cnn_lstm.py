"""Train a gold-supervised CNN-LSTM on prepared frame-window tensors."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gold_event_models import FrameWindowDataset, GoldEventCNNLSTM  # noqa: E402


def main() -> int:
    args = parse_args()
    return train(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train gold-supervised CNN-LSTM event prototype")
    parser.add_argument("--data", type=Path, default=Path("temporal_module/data_gold/frame_windows/gold_frame_windows.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("temporal_module/runs_gold/event_cnn_lstm"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    return parser.parse_args()


def train(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.data.exists():
        write_not_ready(args.output_dir, args.data, "Prepared frame-window dataset does not exist yet.")
        print(f"Frame-window dataset not found: {args.data}")
        return 0

    set_seed(args.seed)
    data = np.load(args.data, allow_pickle=True)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    label_names = [str(item) for item in data["label_names"]]
    if x.ndim != 5:
        write_not_ready(args.output_dir, args.data, "Expected X with shape [windows, frames, height, width, channels] or [windows, frames, channels, height, width].")
        print("Frame-window X must be 5-dimensional.")
        return 0
    if len(x) < 3 or len(set(y.tolist())) < 2:
        write_not_ready(args.output_dir, args.data, "Dataset has too few samples or classes for train/validation/test evaluation.")
        print("Dataset has too few samples or classes for train/validation/test evaluation.")
        return 0

    metadata = load_window_metadata(args.data, len(x))
    splits, split_warnings = clip_disjoint_split(
        y=y,
        label_names=label_names,
        metadata=metadata,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        write_not_ready(args.output_dir, args.data, "Clip-disjoint split produced an empty train, validation, or test split.")
        print("Clip-disjoint split produced an empty train, validation, or test split.")
        return 0

    dataset = FrameWindowDataset(x, y)
    sample_x, _ = dataset[0]
    in_channels = int(sample_x.shape[1])
    model = GoldEventCNNLSTM(
        num_classes=len(label_names),
        in_channels=in_channels,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss(weight=class_weights(y[train_idx], len(label_names)))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_state = None
    best_macro_f1 = -1.0
    for epoch in range(1, max(1, args.epochs) + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, targets, preds, _probs = evaluate(model, val_loader, criterion)
        metrics = classification_metrics(targets, preds, label_names)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": metrics["overall"]["accuracy"],
            "macro_f1": metrics["overall"]["macro_f1"],
        }
        history.append(row)
        if row["macro_f1"] > best_macro_f1:
            best_macro_f1 = row["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    val_loss, val_targets, val_preds, val_probs = evaluate(model, val_loader, criterion)
    test_loss, test_targets, test_preds, test_probs = evaluate(model, test_loader, criterion)
    val_metrics = classification_metrics(val_targets, val_preds, label_names)
    test_metrics = classification_metrics(test_targets, test_preds, label_names)
    val_metrics["overall"]["loss"] = val_loss
    test_metrics["overall"]["loss"] = test_loss

    torch.save(
        {
            "model_state": model.state_dict(),
            "num_classes": len(label_names),
            "label_names": label_names,
            "in_channels": in_channels,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "split_policy": "clip_disjoint",
        },
        args.output_dir / "best_model.pt",
    )
    write_history(history, args.output_dir / "metrics.csv")
    write_confusion(val_metrics["confusion"], label_names, args.output_dir / "confusion_matrix_val.csv")
    write_confusion(test_metrics["confusion"], label_names, args.output_dir / "confusion_matrix_test.csv")
    write_confusion(test_metrics["confusion"], label_names, args.output_dir / "confusion_matrix.csv")
    write_predictions(val_idx, val_targets, val_preds, val_probs, label_names, metadata, args.output_dir / "predictions_val.csv")
    write_predictions(test_idx, test_targets, test_preds, test_probs, label_names, metadata, args.output_dir / "predictions_test.csv")
    write_split_manifest(args.output_dir / "clip_split_manifest.csv", metadata, y, label_names, splits)
    write_summary(args.output_dir, args, x, y, train_idx, val_idx, test_idx, val_metrics, test_metrics, history, split_warnings)
    print(f"Wrote gold CNN-LSTM run to {args.output_dir}")
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
        support = int(confusion[idx, :].sum())
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1_values.append(f1)
    accuracy = float(np.trace(confusion) / confusion.sum()) if confusion.sum() else 0.0
    return {"overall": {"accuracy": accuracy, "macro_f1": float(np.mean(f1_values))}, "per_class": per_class, "confusion": confusion}


def load_window_metadata(data_path: Path, expected_rows: int) -> list[dict[str, str]]:
    metadata_path = data_path.with_name(f"{data_path.stem}_metadata.csv")
    if not metadata_path.exists():
        return [{"window_id": str(idx), "clip_id": f"window_{idx}", "target_class": ""} for idx in range(expected_rows)]
    with metadata_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if len(rows) != expected_rows:
        raise ValueError(f"{metadata_path} has {len(rows)} rows, expected {expected_rows}")
    for idx, row in enumerate(rows):
        row.setdefault("window_id", str(idx))
        row.setdefault("clip_id", f"window_{idx}")
    return rows


def clip_disjoint_split(
    y: np.ndarray,
    label_names: list[str],
    metadata: list[dict[str, str]],
    seed: int,
    val_ratio: float,
    test_ratio: float,
) -> tuple[dict[str, np.ndarray], list[str]]:
    rng = random.Random(seed)
    clip_to_indices: dict[str, list[int]] = {}
    for idx, row in enumerate(metadata):
        clip_to_indices.setdefault(str(row.get("clip_id", f"window_{idx}")), []).append(idx)

    clips = sorted(clip_to_indices)
    shuffled = clips[:]
    rng.shuffle(shuffled)
    target_test = max(1, int(round(len(clips) * test_ratio)))
    target_val = max(1, int(round(len(clips) * val_ratio)))
    assignments: dict[str, str] = {}

    class_to_clips: dict[int, list[str]] = {}
    for class_id in range(len(label_names)):
        class_clips = [clip for clip in clips if any(int(y[idx]) == class_id for idx in clip_to_indices[clip])]
        rng.shuffle(class_clips)
        class_to_clips[class_id] = class_clips

    def split_has_class(split: str, class_id: int) -> bool:
        return any(
            assigned_split == split and any(int(y[idx]) == class_id for idx in clip_to_indices[clip])
            for clip, assigned_split in assignments.items()
        )

    def choose_clip(class_clips: list[str], blocked_splits: set[str]) -> str | None:
        for clip in class_clips:
            if clip not in assignments:
                return clip
        for clip in class_clips:
            if assignments.get(clip) not in blocked_splits:
                return clip
        return None

    for class_id in sorted(class_to_clips, key=lambda item: len(class_to_clips[item])):
        class_clips = class_to_clips[class_id]
        if not class_clips:
            continue
        if not split_has_class("train", class_id):
            clip = choose_clip(class_clips, {"test", "val"})
            if clip is not None:
                assignments[clip] = "train"
        if len(class_clips) >= 2 and not split_has_class("test", class_id):
            clip = choose_clip(class_clips, {"train", "val"})
            if clip is not None:
                assignments[clip] = "test"
        if len(class_clips) >= 3 and not split_has_class("val", class_id):
            clip = choose_clip(class_clips, {"train", "test"})
            if clip is not None:
                assignments[clip] = "val"

    for clip in shuffled:
        if clip in assignments:
            continue
        if list(assignments.values()).count("test") < target_test:
            assignments[clip] = "test"
        elif list(assignments.values()).count("val") < target_val:
            assignments[clip] = "val"
        else:
            assignments[clip] = "train"

    split_indices = {"train": [], "val": [], "test": []}
    for clip in clips:
        split = assignments.get(clip, "train")
        split_indices[split].extend(clip_to_indices[clip])

    warnings = []
    for split, indices in split_indices.items():
        present = {label_names[int(y[idx])] for idx in indices}
        missing = [label for label in label_names if label not in present]
        if missing:
            warnings.append(f"{split} split has no windows for: {', '.join(missing)}")

    return {split: np.asarray(sorted(indices), dtype=np.int64) for split, indices in split_indices.items()}, warnings


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


def write_predictions(indices, targets, preds, probs, label_names, metadata, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        fields = [
            "window_id",
            "clip_id",
            "start_frame",
            "end_frame",
            "gold_label",
            "predicted_label",
            "confidence",
            *[f"prob_{label}" for label in label_names],
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for window_id, target, pred, prob in zip(indices, targets, preds, probs):
            source = metadata[int(window_id)]
            row = {
                "window_id": int(window_id),
                "clip_id": source.get("clip_id", ""),
                "start_frame": source.get("start_frame", ""),
                "end_frame": source.get("end_frame", ""),
                "gold_label": label_names[int(target)],
                "predicted_label": label_names[int(pred)],
                "confidence": max(prob),
            }
            row.update({f"prob_{label}": value for label, value in zip(label_names, prob)})
            writer.writerow(row)


def write_split_manifest(path: Path, metadata: list[dict[str, str]], y: np.ndarray, label_names: list[str], splits: dict[str, np.ndarray]) -> None:
    index_to_split = {int(idx): split for split, indices in splits.items() for idx in indices}
    clip_rows: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(metadata):
        clip_id = str(row.get("clip_id", ""))
        split = index_to_split.get(idx, "train")
        out = clip_rows.setdefault(
            clip_id,
            {"split": split, "clip_id": clip_id, "windows": 0, **{f"{label}_windows": 0 for label in label_names}},
        )
        out["windows"] += 1
        out[f"{label_names[int(y[idx])]}_windows"] += 1
    fields = ["split", "clip_id", "windows", *[f"{label}_windows" for label in label_names]]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(clip_rows.values(), key=lambda item: (item["split"], item["clip_id"])))


def split_counts(y: np.ndarray, label_names: list[str], splits: dict[str, np.ndarray]) -> dict[str, dict[str, int]]:
    return {
        split: {label: int(sum(1 for idx in indices if int(y[idx]) == class_id)) for class_id, label in enumerate(label_names)}
        for split, indices in splits.items()
    }


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    val_metrics: dict,
    test_metrics: dict,
    history: list[dict],
    split_warnings: list[str],
) -> None:
    label_names = [str(item) for item in np.load(args.data, allow_pickle=True)["label_names"]]
    counts = Counter(int(item) for item in y.tolist())
    final = history[-1] if history else {}
    val_selected = val_metrics.get("overall", {})
    test_selected = test_metrics.get("overall", {})
    lines = [
        "# Gold Event CNN-LSTM Training Summary",
        "",
        "This model is a visual-frame prototype for manually labelled event windows.",
        "",
        "Evaluation protocol: clip-disjoint train/validation/test split. The validation split selects the best epoch; the test split is held out for final reporting.",
        "",
        f"- Dataset shape: {tuple(x.shape)}",
        f"- Train windows: {len(train_idx)}",
        f"- Validation windows: {len(val_idx)}",
        f"- Test windows: {len(test_idx)}",
        f"- Epochs run: {len(history)}",
        f"- Selected-model validation accuracy: {val_selected.get('accuracy', 0.0):.4f}",
        f"- Selected-model validation macro F1: {val_selected.get('macro_f1', 0.0):.4f}",
        f"- Held-out test accuracy: {test_selected.get('accuracy', 0.0):.4f}",
        f"- Held-out test macro F1: {test_selected.get('macro_f1', 0.0):.4f}",
        f"- Last-epoch validation accuracy: {final.get('val_accuracy', 0.0):.4f}",
        f"- Last-epoch macro F1: {final.get('macro_f1', 0.0):.4f}",
        "",
        "## Label Distribution",
        "",
    ]
    lines.extend(f"- {label}: {counts.get(idx, 0)}" for idx, label in enumerate(label_names))
    if split_warnings:
        lines.extend(["", "## Split Warnings", ""])
        lines.extend(f"- {warning}" for warning in split_warnings)
    lines.extend(["", "## Test Per-Class Metrics", ""])
    for label in label_names:
        item = test_metrics["per_class"].get(label, {})
        lines.append(
            f"- {label}: precision {item.get('precision', 0.0):.4f}, "
            f"recall {item.get('recall', 0.0):.4f}, F1 {item.get('f1', 0.0):.4f}, "
            f"support {item.get('support', 0)}"
        )
    (output_dir / "training_summary.md").write_text("\n".join(lines), encoding="utf-8")
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "validation": val_selected,
                "test": test_selected,
                "last_epoch": final,
                "test_per_class": test_metrics["per_class"],
                "validation_per_class": val_metrics["per_class"],
                "split_counts": split_counts(y, label_names, splits),
                "split_warnings": split_warnings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_not_ready(output_dir: Path, data_path: Path, reason: str) -> None:
    lines = [
        "# Gold Event CNN-LSTM Training Summary",
        "",
        "Status: NOT READY",
        "",
        f"Dataset path: `{data_path}`",
        f"Reason: {reason}",
        "",
        "Prepare frame-window tensors from gold event intervals before training this model.",
    ]
    (output_dir / "training_summary.md").write_text("\n".join(lines), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    raise SystemExit(main())
