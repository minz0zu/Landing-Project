import os
import cv2
import torch
import numpy as np
import glob
from ultralytics import YOLO
from torchvision.ops import nms

def run_inference_with_margins():
    model_path = "/home/minji/landing_project/train_results/SGD/weights/best.pt"
    test_img_dir = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/Test/01.원천데이터"
    save_dir = "/mnt/hdd_6tb/minji/inference_results/SGD_split_merged02"
    os.makedirs(save_dir, exist_ok=True)

    # 장치 설정
    device = 3 if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    
    margin_rate = 0.2
    conf_threshold = 0.25

    img_files = glob.glob(os.path.join(test_img_dir, "**", "*.jpg"), recursive=True)
    print(f"총 {len(img_files)}개의 이미지를 추론합니다.")

    for img_path in img_files:
        img = cv2.imread(img_path)
        if img is None: continue 
        
        h, w, _ = img.shape
        mid_h, mid_w = h // 2, w // 2
        m_h, m_w = int(mid_h * margin_rate), int(mid_w * margin_rate)

        # 4개 분할 (Overlap 포함)
        slices = [
            img[0 : mid_h + m_h, 0 : mid_w + m_w],
            img[0 : mid_h + m_h, mid_w - m_w : w],
            img[mid_h - m_h : h, 0 : mid_w + m_w],
            img[mid_h - m_h : h, mid_w - m_w : w]
        ]
        
        offsets = [
            (0, 0),
            (mid_w - m_w, 0),
            (0, mid_h - m_h),
            (mid_w - m_w, mid_h - m_h)
        ]

        # Batch Inference 실행
        results = model.predict(source=slices, conf=conf_threshold, imgsz=640, device=device, verbose=False)

        all_boxes = []
        all_scores = []
        all_classes = []

        for i, res in enumerate(results):
            if len(res.boxes) == 0: continue
            
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy() # 클래스 정보 가져오기
            off_x, off_y = offsets[i]

            for box, score, cls in zip(boxes, scores, classes):
                # 글로벌 좌표로 변환
                global_box = [box[0] + off_x, box[1] + off_y, box[2] + off_x, box[3] + off_y]
                all_boxes.append(global_box)
                all_scores.append(score)
                all_classes.append(cls)

        if len(all_boxes) > 0:
            boxes_tensor = torch.tensor(all_boxes)
            scores_tensor = torch.tensor(all_scores)
            
            # NMS (전체 영역에서 중복 제거)
            keep = nms(boxes_tensor, scores_tensor, iou_threshold=0.2)
            
            for idx in keep:
                box = boxes_tensor[idx].numpy()
                cls = int(all_classes[idx])
                conf = all_scores[idx]
                
                # 시각화: 박스 + 클래스 이름/점수
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)
                cv2.putText(img, label, (int(box[0]), int(box[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 결과 저장 (파일명 중복 방지를 위해 원본 경로 구조 일부 활용 권장)
        save_path = os.path.join(save_dir, os.path.basename(img_path))
        cv2.imwrite(save_path, img)

if __name__ == "__main__":
    run_inference_with_margins()