import glob
import os

import pandas as pd


# Settings: update these paths for the clip/annotation set you want to evaluate.
CVAT_LABELS_DIR = r"data\annotations\video_1\obj_train_data"
TRACKS_CSV = r"outputs\england_epl__2014_2015__2015_04_11___19_30_Burnley_0___1_Arsenal__h1_720p\tracks\tracks.csv"
BALL_CSV = r"outputs\england_epl__2014_2015__2015_04_11___19_30_Burnley_0___1_Arsenal__h1_720p\tracks\ball_tracks.csv"

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


print("1. Loading predictions (YOLO CSVs)...")
df_tracks = pd.read_csv(TRACKS_CSV)
df_ball = pd.read_csv(BALL_CSV)

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
txt_files = glob.glob(os.path.join(CVAT_LABELS_DIR, "*.txt"))

for txt_file in txt_files:
    frame_id = int(os.path.basename(txt_file).split("_")[1].split(".")[0])

    gt_boxes = {class_id: [] for class_id in EVALUATED_CLASSES}
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f.readlines():
            parts = [float(x) for x in line.strip().split()]
            class_id = int(parts[0])
            if class_id not in gt_boxes:
                continue
            bbox = yolo_to_bbox(parts[1], parts[2], parts[3], parts[4], VIDEO_WIDTH, VIDEO_HEIGHT)
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

            if best_iou >= IOU_THRESHOLD:
                results[class_id]["TP"] += 1
                matched_gt.add(best_gt_idx)
            else:
                results[class_id]["FP"] += 1

        results[class_id]["FN"] += len(gts) - len(matched_gt)

print("\n=== FINAL RESULTS (IoU Threshold: 0.5) ===")
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
