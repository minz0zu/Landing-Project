import os
import cv2
import glob
import numpy as np
from ultralytics import YOLO

def calculate_iou(box1, box2):
    """IoU(Intersection over Union) 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0

def evaluate_with_txt(model_dir, img_dir, label_dir, target_conf=0.48):
    """
    YOLO txt 라벨 기반 상세 성능 평가
    """
    model = YOLO(model_dir)

    formats = ['*.jpg', '*.jpeg', '*.png']
    images = []
    for fmt in formats:
        images.extend(glob.glob(os.path.join(img_dir, "**", fmt), recursive=True))

    total_gt = 0
    all_results = []
    iou_threshold = 0.5

    print(f"평가 시작: 총 {len(images)}개의 이미지")

    for img_file in images:
        img = cv2.imread(img_file)
        if img is None: continue
        h, w = img.shape[:2]
        
        base_name = os.path.splitext(os.path.basename(img_file))[0]
        label_file = os.path.join(label_dir, f"{base_name}.txt")
        
        if not os.path.exists(label_file): continue

        # 1. 정답(GT) 박스 로드
        gt_boxes = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.split()
                if not parts: continue
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = (cx - bw/2) * w
                y1 = (cy - bh/2) * h
                x2 = (cx + bw/2) * w
                y2 = (cy + bh/2) * h
                gt_boxes.append([x1, y1, x2, y2])
                total_gt += 1

        # 2. 모델 예측 (AP 계산을 위해 0.001 사용)
        results = model.predict(source=img, conf=0.001, verbose=False)
        p_boxes = results[0].boxes.xyxy.cpu().tolist()
        p_scores = results[0].boxes.conf.cpu().tolist()

        # 3. 매칭 로직
        matched_gt = [False] * len(gt_boxes)
        for box, score in zip(p_boxes, p_scores):
            best_iou = 0
            best_idx = -1
            for i, gt_box in enumerate(gt_boxes):
                if matched_gt[i]: continue
                iou = calculate_iou(gt_box, box)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_idx != -1:
                all_results.append((score, 1))
                matched_gt[best_idx] = True
            else:
                all_results.append((score, 0))

    # 4. 지표 계산
    all_results.sort(key=lambda x: x[0], reverse=True)
    scores = np.array([x[0] for x in all_results])
    tps = np.array([x[1] for x in all_results])
    fps = 1 - tps
    
    # AP 계산
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)
    recalls = tp_cum / total_gt if total_gt > 0 else np.zeros_like(tp_cum)
    precisions = tp_cum / (tp_cum + fp_cum)
    
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    ap = np.sum((mrec[1:] - mrec[:-1]) * mpre[1:])

    # 5. 설정한 target_conf 기준 최종 성적 필터링
    final_preds_mask = scores >= target_conf
    final_tp = np.sum(tps[final_preds_mask])
    final_fp = np.sum(fps[final_preds_mask])
    final_total_preds = np.sum(final_preds_mask) # 총 예측 개수 (Conf 기준)
    final_fn = total_gt - final_tp
    
    precision = (final_tp / final_total_preds) * 100 if final_total_preds > 0 else 0
    recall = (final_tp / total_gt) * 100 if total_gt > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*45)
    print(f"상세 성능 평가 결과 (IoU: {iou_threshold}, Conf: {target_conf})")
    print("-" * 45)
    print(f"📊 총 정답 개수(GT):    {total_gt}")
    print(f"🔎 총 예측 개수(Pred):  {int(final_total_preds)}")
    print("-" * 45)
    print(f"✅ TP (정확히 탐지):    {int(final_tp)}")
    print(f"❌ FP (오탐지):        {int(final_fp)}")
    print(f"📉 FN (미탐지/놓침):    {int(final_fn)}")
    print("-" * 45)
    print(f"📍 Recall:          {recall:.2f}%")
    print(f"📍 Precision:       {precision:.2f}%")
    print(f"📍 F1-Score:        {f1/100:.4f}")
    print(f"⭐ AP:              {ap:.5f} (종합 실력)")
    print("="*45)

if __name__ == "__main__":
    MODEL_PATH = "/home/minji/landing_project/train_results/SGD/weights/best.pt"
    IMG_PATH = "/mnt/hdd_6tb/minji/processed_quarter/Test/images"
    LABEL_PATH = "/mnt/hdd_6tb/minji/processed_quarter/Test/labels"
    
    evaluate_with_txt(MODEL_PATH, IMG_PATH, LABEL_PATH, target_conf=0.48)