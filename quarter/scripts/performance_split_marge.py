import os
import cv2
import torch
import numpy as np
import glob
from ultralytics import YOLO
from torchvision.ops import nms

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

def evaluate_split_and_merge(model_dir, img_dir, label_dir, target_conf=0.48, margin_rate=0.2):
    """
    이미지를 4분할(마진 포함)하여 추론 후 결과를 병합하여 성능 평가
    """
    # 모델 로드 및 장치 설정
    device = 3 if torch.cuda.is_available() else "cpu"
    model = YOLO(model_dir)
    formats = ['*.jpg', '*.jpeg', '*.png']
    images = []
    for fmt in formats:
        images.extend(glob.glob(os.path.join(img_dir, "**", fmt), recursive=True))

    total_gt = 0
    all_results = [] # (score, match_status) 저장
    iou_threshold = 0.5
    nms_iou_threshold = 0.2 # 분할 영역 간 중복 박스 제거용

    print(f"마진 병합 성능 평가 시작: 총 {len(images)}개의 이미지")

    for img_file in images:
        img = cv2.imread(img_file)
        if img is None: continue
        h, w = img.shape[:2]
        
        # 1. 정답(GT) 라벨 로드 (YOLO txt 포맷)
        base_name = os.path.splitext(os.path.basename(img_file))[0]
        label_file = os.path.join(label_dir, f"{base_name}.txt")
        
        gt_boxes = []
        if os.path.exists(label_file):
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

        # 2. 4분할 슬라이싱 (두 번째 코드 알고리즘)
        mid_h, mid_w = h // 2, w // 2
        m_h, m_w = int(mid_h * margin_rate), int(mid_w * margin_rate)

        slices = [
            img[0 : mid_h + m_h, 0 : mid_w + m_w],
            img[0 : mid_h + m_h, mid_w - m_w : w],
            img[mid_h - m_h : h, 0 : mid_w + m_w],
            img[mid_h - m_h : h, mid_w - m_w : w]
        ]
        offsets = [
            (0, 0), (mid_w - m_w, 0), (0, mid_h - m_h), (mid_w - m_w, mid_h - m_h)
        ]

        # 3. 분할 이미지 배치 추론 (AP 계산을 위해 매우 낮은 conf 사용)
        results = model.predict(source=slices, conf=0.001, imgsz=640, device=device, verbose=False)

        merged_boxes = []
        merged_scores = []

        for i, res in enumerate(results):
            if len(res.boxes) == 0: continue
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            off_x, off_y = offsets[i]

            for box, score in zip(boxes, scores):
                # 글로벌 좌표로 변환
                global_box = [box[0] + off_x, box[1] + off_y, box[2] + off_x, box[3] + off_y]
                merged_boxes.append(global_box)
                merged_scores.append(score)

        # 4. NMS로 분할 경계면 중복 박스 제거
        final_p_boxes = []
        final_p_scores = []
        if len(merged_boxes) > 0:
            current_device = next(model.model.parameters()).device

            boxes_t = torch.tensor(merged_boxes, dtype=torch.float32).to(current_device)
            scores_t = torch.tensor(merged_scores, dtype=torch.float32).to(current_device)
            keep = nms(boxes_t, scores_t, iou_threshold=nms_iou_threshold)
            
            final_p_boxes = boxes_t[keep].tolist()
            final_p_scores = scores_t[keep].tolist()

        # 5. GT와 매칭 (성능 측정용)
        matched_gt = [False] * len(gt_boxes)
        # 높은 점수 순으로 정렬하여 매칭
        p_indices = np.argsort(final_p_scores)[::-1]
        
        for idx in p_indices:
            box = final_p_boxes[idx]
            score = final_p_scores[idx]
            best_iou = 0
            best_idx = -1
            
            for i, gt_box in enumerate(gt_boxes):
                if matched_gt[i]: continue
                iou = calculate_iou(gt_box, box)
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            if best_idx != -1:
                all_results.append((score, 1)) # TP
                matched_gt[best_idx] = True
            else:
                all_results.append((score, 0)) # FP

    # --- 데이터가 없는 경우 처리 ---
    if not all_results:
        print("❌ 탐지된 결과가 없습니다.")
        return

    # 6. 지표 계산 (정렬)
    all_results.sort(key=lambda x: x[0], reverse=True)
    scores = np.array([x[0] for x in all_results])
    tps = np.array([x[1] for x in all_results])
    fps = 1 - tps
    
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)
    recalls = tp_cum / total_gt if total_gt > 0 else np.zeros_like(tp_cum)
    precisions = tp_cum / (tp_cum + fp_cum)
    
    # AP 계산 (11-point interpolation 혹은 AUC)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    ap = np.sum((mrec[1:] - mrec[:-1]) * mpre[1:])

    # 7. 설정한 target_conf 기준 최종 성적 필터링
    final_preds_mask = scores >= target_conf
    final_tp = np.sum(tps[final_preds_mask])
    final_fp = np.sum(fps[final_preds_mask])
    final_total_preds = np.sum(final_preds_mask)
    final_fn = total_gt - final_tp
    
    precision = (final_tp / final_total_preds) * 100 if final_total_preds > 0 else 0
    recall = (final_tp / total_gt) * 100 if total_gt > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # 8. 결과 출력
    print("\n" + "="*45)
    print(f"분할 병합 성능 평가 결과 (Margin: {margin_rate}, Conf: {target_conf})")
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
    print(f"⭐ AP:              {ap:.5f}")
    print("="*45)

if __name__ == "__main__":
    MODEL_PATH = "/home/minji/landing_project/train_results/SGD/weights/best.pt"
    IMG_PATH = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/Test/images"
    LABEL_PATH = "/mnt/hdd_6tb/minji/processed/Test/labels"
    
    evaluate_split_and_merge(MODEL_PATH, IMG_PATH, LABEL_PATH, target_conf=0.48, margin_rate=0.2)