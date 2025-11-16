import os
import glob
import cv2
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.ops import box_iou
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def evaluate_yolo_custom(
        weights_path,
        images_dir,
        labels_dir,
        save_dir="custom_eval_results",
        iou_thresholds=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        conf_thresholds=[0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
):
    os.makedirs(save_dir, exist_ok=True)

    # Load YOLO model
    model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path, force_reload=True)
    model.cuda() if torch.cuda.is_available() else model.cpu()

    # Collect images
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.*")))

    # Prepare metric grids
    P_grid = np.zeros((len(iou_thresholds), len(conf_thresholds)))
    R_grid = np.zeros((len(iou_thresholds), len(conf_thresholds)))
    F1_grid = np.zeros((len(iou_thresholds), len(conf_thresholds)))

    # ====================================================
    # Calculate P, R, F1 for all IoU × Confidence pairs
    # ====================================================
    for i_iou, iou_thres in enumerate(iou_thresholds):
        for i_conf, conf_thres in enumerate(conf_thresholds):

            TP = FP = FN = 0

            for img_path in image_paths:

                img = cv2.imread(img_path)
                h, w = img.shape[:2]

                # Load label
                label_path = os.path.join(labels_dir, Path(img_path).stem + ".txt")
                gt_boxes = []

                if os.path.exists(label_path):
                    for line in open(label_path, "r"):
                        c, xc, yc, bw, bh = map(float, line.split())
                        x1 = (xc - bw / 2) * w
                        y1 = (yc - bh / 2) * h
                        x2 = (xc + bw / 2) * w
                        y2 = (yc + bh / 2) * h
                        gt_boxes.append([x1, y1, x2, y2])

                gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)

                # Run inference
                results = model(img_path, size=640)
                pred = results.xyxy[0].cpu()

                # Confidence filter
                pred = pred[pred[:, 4] >= conf_thres]

                # No labels
                if len(gt_boxes) == 0:
                    FP += len(pred)
                    continue

                # No predictions
                if len(pred) == 0:
                    FN += len(gt_boxes)
                    continue

                # IoU matrix
                ious = box_iou(gt_boxes, pred[:, :4])

                matched_gt = set()
                matched_pred = set()

                for g in range(len(gt_boxes)):
                    for p in range(len(pred)):
                        if ious[g, p] >= iou_thres:
                            if g not in matched_gt and p not in matched_pred:
                                matched_gt.add(g)
                                matched_pred.add(p)

                TP += len(matched_gt)
                FP += len(pred) - len(matched_pred)
                FN += len(gt_boxes) - len(matched_gt)

            # Compute metrics
            P = TP / (TP + FP + 1e-9)
            R = TP / (TP + FN + 1e-9)
            F1 = 2 * P * R / (P + R + 1e-9)

            P_grid[i_iou, i_conf] = P
            R_grid[i_iou, i_conf] = R
            F1_grid[i_iou, i_conf] = F1

    # ====================================================
    # Heatmap drawing
    # ====================================================
    def heatmap(data, title, file_name):
        plt.figure(figsize=(10, 6))
        sns.heatmap(data, annot=True, cmap="YlOrRd",
                    xticklabels=conf_thresholds,
                    yticklabels=iou_thresholds,
                    fmt=".3f")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("IoU Threshold")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, file_name))
        plt.close()

    heatmap(P_grid, "Precision Heatmap", "precision_heatmap.png")
    heatmap(R_grid, "Recall Heatmap", "recall_heatmap.png")
    heatmap(F1_grid, "F1 Heatmap", "f1_heatmap.png")

    # ====================================================
    # P-Curve and R-Curve (using ONLY P_grid / R_grid data)
    # ====================================================
    def plot_curve(grid, title, file_name, ylabel):
        plt.figure(figsize=(10, 6))
        for i, iou in enumerate(iou_thresholds):
            plt.plot(conf_thresholds, grid[i], marker='o', label=f"IoU {iou}")
        plt.xlabel("Confidence Threshold")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc="best")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, file_name))
        plt.close()

    plot_curve(P_grid, "P-Curve (Precision vs Confidence)", "p_curve.png", "Precision")
    plot_curve(R_grid, "R-Curve (Recall vs Confidence)", "r_curve.png", "Recall")

    print(f"All heatmaps and curves saved to: {save_dir}")


# ============================================================
# Run evaluation
# ============================================================

evaluate_yolo_custom(
    weights_path=r"C:\Users\doron\PycharmProjects\yolov5-fire-detection\yolov5\runs\train\exp9\weights\best.pt",
    images_dir=r"C:\Users\doron\OneDrive\Desktop\MaofAIDatasets\test\images",
    labels_dir=r"C:\Users\doron\OneDrive\Desktop\MaofAIDatasets\test\labels",
    save_dir=r"C:\Users\doron\PycharmProjects\yolov5-fire-detection\yolov5\utils\heatmaps_results_graphs.py"
)
