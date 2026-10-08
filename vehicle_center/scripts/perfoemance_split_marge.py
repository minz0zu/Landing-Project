import os
import cv2
import torch
import numpy as np
import glob
from ultralytics import YOLO
from torchvision.ops import nms

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1, area2 = (box1[2] - box1[0]) * (box1[3] - box1[1]), (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def run_evaluation_with_margin():
    MODEL_PATH = "/home/minji/landing_project/vehicle_center/train_results/SGD_aug/weights/best.pt"
    IMG_DIR = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/Test/images"
    LABEL_DIR = "/mnt/hdd_6tb/minji/processed/Test/labels"
    TARGET_CONF = 0.48
    MARGIN_RATE = 0.2
    DEVICE = 1 if torch.cuda.is_available() else "cpu"

    model = YOLO(MODEL_PATH)
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        images.extend(glob.glob(os.path.join(IMG_DIR, "**", ext), recursive=True))
    
    total_gt, all_results = 0, []

    print(f"🚀 [4분할 Margin 0.2 평가] 시작...")

    for img_file in images:
        img = cv2.imread(img_file)
        if img is None: continue
        h, w = img.shape[:2]
        mid_h, mid_w = h // 2, w // 2
        m_h, m_w = int(mid_h * MARGIN_RATE), int(mid_w * MARGIN_RATE)

        base_name = os.path.splitext(os.path.basename(img_file))[0]
        label_file = os.path.join(LABEL_DIR, f"{base_name}.txt")
        gt_boxes = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.split(); cx, cy, bw, bh = map(float, parts[1:])
                    gt_boxes.append([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]); total_gt += 1

        slices = [
            img[0:mid_h+m_h, 0:mid_w+m_w],
            img[0:mid_h+m_h, mid_w-m_w:w],
            img[mid_h-m_h:h, 0:mid_w+m_w],
            img[mid_h-m_h:h, mid_w-m_w:w]
        ]
        offsets = [(0, 0), (mid_w-m_w, 0), (0, mid_h-m_h), (mid_w-m_w, mid_h-m_h)]

        results = model.predict(source=slices, conf=0.001, imgsz=640, device=DEVICE, verbose=False)
        m_boxes, m_scores = [], []
        for i, res in enumerate(results):
            for box, score in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                m_boxes.append([box[0]+offsets[i][0], box[1]+offsets[i][1], box[2]+offsets[i][0], box[3]+offsets[i][1]])
                m_scores.append(score)

        if len(m_boxes) > 0:
            boxes_t = torch.tensor(m_boxes, dtype=torch.float32).to(DEVICE)
            scores_t = torch.tensor(m_scores, dtype=torch.float32).to(DEVICE)
            keep = nms(boxes_t, scores_t, iou_threshold=0.2)
            f_boxes, f_scores = boxes_t[keep].tolist(), scores_t[keep].tolist()
            
            matched_gt = [False] * len(gt_boxes)
            for idx in np.argsort(f_scores)[::-1]:
                box, score = f_boxes[idx], f_scores[idx]
                best_iou, best_idx = 0, -1
                for i, gt_box in enumerate(gt_boxes):
                    if not matched_gt[i]:
                        iou = calculate_iou(gt_box, box)
                        if iou >= 0.5 and iou > best_iou: best_iou, best_idx = iou, i
                if best_idx != -1: all_results.append((score, 1)); matched_gt[best_idx] = True
                else: all_results.append((score, 0))

    # 지표 계산 로직 (출력 형식 통일)
    all_results.sort(key=lambda x: x[0], reverse=True)
    scores = np.array([x[0] for x in all_results]); tps = np.array([x[1] for x in all_results])
    recalls = np.cumsum(tps) / total_gt
    precisions = np.cumsum(tps) / (np.cumsum(tps) + np.cumsum(1-tps))
    mrec = np.concatenate(([0.0], recalls, [1.0])); mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre)-1, 0, -1): mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
    ap = np.sum((mrec[1:] - mrec[:-1]) * mpre[1:])
    mask = scores >= TARGET_CONF
    f_tp, f_total = np.sum(tps[mask]), np.sum(mask)
    prec = (f_tp / f_total) * 100 if f_total > 0 else 0
    rec = (f_tp / total_gt) * 100 if total_gt > 0 else 0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    print("\n" + "="*45)
    print(f"4분할 (Margin {MARGIN_RATE}) 평가 결과 (Conf: {TARGET_CONF})")
    print("-" * 45)
    print(f"📊 총 정답 개수(GT):    {total_gt}")
    print(f"🔎 총 예측 개수(Pred):  {int(f_total)}")
    print("-" * 45)
    print(f"✅ TP: {int(f_tp)} | ❌ FP: {int(f_total-f_tp)} | 📉 FN: {int(total_gt-f_tp)}")
    print("-" * 45)
    print(f"📍 Recall: {rec:.2f}% | Precision: {prec:.2f}%")
    print(f"📍 F1-Score: {f1/100:.4f} | ⭐ AP: {ap:.5f}")
    print("="*45)

if __name__ == "__main__":
    run_evaluation_with_margin()