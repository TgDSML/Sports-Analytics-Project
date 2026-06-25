"""Evaluate player and ball detections against YOLO/CVAT labels.

Example:
    python -m src.metrics.evaluate_metrics \
        --ground-truth data/annotations/video_1/obj_train_data \
        --tracks-csv outputs/<clip_id>/tracks/tracks.csv \
        --ball-csv outputs/<clip_id>/tracks/ball_tracks.csv
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import pandas as pd


VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
IOU_THRESHOLD = 0.5

CLASS_MAP = {0: "Player", 2: "Ball"}
EVALUATED_CLASSES = list(CLASS_MAP.keys())


def yolo_to_bbox(cx, cy, w, h, img_w, img_h):
    """Convert normalized YOLO format to absolute [x1, y1, x2, y2] pixels."""
    abs_cx, abs_cy = cx * img_w, cy * img_h
    abs_w, abs_h = w * img_w, h * img_h
    return [abs_cx - abs_w / 2, abs_cy - abs_h / 2, abs_cx + abs_w / 2, abs_cy + abs_h / 2]


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return intersection_area / float(box1_area + box2_area - intersection_area)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate player and ball detections against YOLO/CVAT labels."
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        help="Directory containing YOLO/CVAT .txt labels.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help=(
            "Optional prediction root. If a clip output root is supplied, the script "
            "looks for tracks/tracks.csv and tracks/ball_tracks.csv under it."
        ),
    )
    parser.add_argument("--tracks-csv", type=Path, help="Player tracks CSV.")
    parser.add_argument("--ball-csv", type=Path, help="Ball tracks CSV.")
    parser.add_argument("--frame-width", type=int, default=VIDEO_WIDTH)
    parser.add_argument("--frame-height", type=int, default=VIDEO_HEIGHT)
    parser.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD)
    return parser


def resolve_prediction_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    tracks_csv = args.tracks_csv
    ball_csv = args.ball_csv

    if args.predictions and args.predictions.is_dir():
        tracks_csv = tracks_csv or args.predictions / "tracks" / "tracks.csv"
        ball_csv = ball_csv or args.predictions / "tracks" / "ball_tracks.csv"
    elif args.predictions and args.predictions.is_file():
        tracks_csv = tracks_csv or args.predictions

    return tracks_csv, ball_csv


def validate_inputs(
    ground_truth: Path | None,
    tracks_csv: Path | None,
    ball_csv: Path | None,
    parser: argparse.ArgumentParser | None = None,
) -> bool:
    missing = []
    if ground_truth is None:
        missing.append("--ground-truth")
    elif not ground_truth.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {ground_truth}")

    if tracks_csv is None:
        missing.append("--tracks-csv or --predictions")
    elif not tracks_csv.exists():
        raise FileNotFoundError(f"Tracks CSV not found: {tracks_csv}")

    if ball_csv is None:
        missing.append("--ball-csv or --predictions")
    elif not ball_csv.exists():
        raise FileNotFoundError(f"Ball CSV not found: {ball_csv}")

    if missing:
        if parser:
            parser.print_help()
        print("\nMissing required input(s): " + ", ".join(missing))
        return False
    return True


def evaluate(
    labels_dir: Path,
    tracks_csv: Path,
    ball_csv: Path,
    frame_width: int,
    frame_height: int,
    iou_threshold: float,
) -> dict[int, dict[str, int]]:
    print("1. Loading predictions (YOLO CSVs)...")
    df_tracks = pd.read_csv(tracks_csv)
    df_ball = pd.read_csv(ball_csv)

    # Prepare predictions. Referee is intentionally excluded because this pipeline
    # does not currently produce a referee detector output to evaluate.
    if "class_name" in df_tracks.columns:
        df_tracks = df_tracks[df_tracks["class_name"].astype(str).str.lower() != "referee"].copy()
    df_tracks["class_id"] = 0
    df_ball["class_id"] = 2
    df_ball["x1"] = df_ball["center_x"] - 7.5  # Approximate 15 px ball box.
    df_ball["y1"] = df_ball["center_y"] - 7.5
    df_ball["x2"] = df_ball["center_x"] + 7.5
    df_ball["y2"] = df_ball["center_y"] + 7.5

    preds_all = pd.concat(
        [
            df_tracks[["frame", "class_id", "x1", "y1", "x2", "y2"]],
            df_ball[["frame", "class_id", "x1", "y1", "x2", "y2"]],
        ]
    )

    print("2. Calculating metrics...")
    results = {class_id: {"TP": 0, "FP": 0, "FN": 0} for class_id in EVALUATED_CLASSES}
    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))

    for txt_file in txt_files:
        frame_id = int(os.path.basename(txt_file).split("_")[1].split(".")[0])

        gt_boxes = {class_id: [] for class_id in EVALUATED_CLASSES}
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                parts = [float(x) for x in line.strip().split()]
                class_id = int(parts[0])
                if class_id not in gt_boxes:
                    continue
                bbox = yolo_to_bbox(parts[1], parts[2], parts[3], parts[4], frame_width, frame_height)
                gt_boxes[class_id].append(bbox)

        pred_frame = preds_all[preds_all["frame"] == frame_id]

        for class_id in EVALUATED_CLASSES:
            gts = gt_boxes[class_id]
            preds = pred_frame[pred_frame["class_id"] == class_id][["x1", "y1", "x2", "y2"]].values.tolist()

            matched_gt = set()

            # Greedy matching: each ground-truth box can be matched only once.
            for pred_box in preds:
                best_iou = 0
                best_gt_idx = -1
                for g_idx, gt_box in enumerate(gts):
                    if g_idx in matched_gt:
                        continue
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= iou_threshold:
                    results[class_id]["TP"] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    results[class_id]["FP"] += 1

            results[class_id]["FN"] += len(gts) - len(matched_gt)

    return results


def print_results(results: dict[int, dict[str, int]], iou_threshold: float) -> None:
    print(f"\n=== FINAL RESULTS (IoU Threshold: {iou_threshold}) ===")
    for class_id, cls_name in CLASS_MAP.items():
        tp = results[class_id]["TP"]
        fp = results[class_id]["FP"]
        fn = results[class_id]["FN"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"--- {cls_name} ---")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1_score:.4f}")
        print(f"(TP: {tp}, FP: {fp}, FN: {fn})\n")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    tracks_csv, ball_csv = resolve_prediction_paths(args)
    if not validate_inputs(args.ground_truth, tracks_csv, ball_csv, parser):
        return 2

    results = evaluate(
        labels_dir=args.ground_truth,
        tracks_csv=tracks_csv,
        ball_csv=ball_csv,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        iou_threshold=args.iou_threshold,
    )
    print_results(results, args.iou_threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
