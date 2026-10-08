import cv2
import numpy as np
from ultralytics import YOLO
import os

# 1. 초기 설정
MODEL_PATH = "/home/minji/landing_project/quarter/train_results/SGD_aug/weights/best.pt"
VIDEO_PATH = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/original_video.mp4"
SAVE_PATH = "/home/minji/landing_project/vehicle_center/train_results/custom.mp4"

model = YOLO(MODEL_PATH).to('cuda:0')
cap = cv2.VideoCapture(VIDEO_PATH)

# 저장 설정 추가
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
out = cv2.VideoWriter(SAVE_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

class CustomPlateTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.max_lost = 30
        
    def compute_similarity(self, crop1, crop2):
        if crop1 is None or crop2 is None or crop1.size == 0 or crop2.size == 0: return 0
        crop2_res = cv2.resize(crop2, (crop1.shape[1], crop1.shape[0]))
        diff = cv2.absdiff(crop1, crop2_res)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        return 1.0 - (np.count_nonzero(thresh) / thresh.size)

    def update(self, frame, detections):
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_tracks = {}
        
        for det_box in detections:
            x1, y1, x2, y2 = map(int, det_box)
            # 좌표 범위를 프레임 크기 내로 제한 (IndexError 방지)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            roi = current_frame_gray[y1:y2, x1:x2]
            best_match_id = -1
            max_sim = 0
            
            for tid, tdata in self.tracks.items():
                prev_box = tdata['box']
                dist_x = abs((x1+x2)/2 - (prev_box[0]+prev_box[2])/2)
                dist_y = abs((y1+y2)/2 - (prev_box[1]+prev_box[3])/2) / 1.5 
                
                sim = self.compute_similarity(tdata['feature'], roi)
                if dist_x < 100 and dist_y < 150 and sim > max_sim:
                    max_sim = sim
                    best_match_id = tid

            if best_match_id != -1 and max_sim > 0.1:
                new_tracks[best_match_id] = {'box': [x1, y1, x2, y2], 'feature': roi, 'frames_lost': 0}
            else:
                new_tracks[self.next_id] = {'box': [x1, y1, x2, y2], 'feature': roi, 'frames_lost': 0}
                self.next_id += 1

        for tid in list(self.tracks.keys()):
            if tid not in new_tracks and self.tracks[tid]['frames_lost'] < self.max_lost:
                self.tracks[tid]['frames_lost'] += 1
                new_tracks[tid] = self.tracks[tid]
                
        self.tracks = new_tracks
        return self.tracks

tracker = CustomPlateTracker()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    results = model.predict(frame, conf=0.1, imgsz=1280, verbose=False)[0]
    det_boxes = results.boxes.xyxy.cpu().numpy()
    current_tracks = tracker.update(frame, det_boxes)

    for tid, tdata in current_tracks.items():
        if tdata['frames_lost'] == 0:
            x1, y1, x2, y2 = map(int, tdata['box'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{tid}", (x1, y1-10), 0, 0.6, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
print(f"작업 완료! 결과 파일: {SAVE_PATH}")