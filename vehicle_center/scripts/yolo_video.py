import cv2
from ultralytics import YOLO
import os

MODEL_PATH = "/home/minji/landing_project/quarter/train_results/SGD_aug/weights/best.pt"
model = YOLO(MODEL_PATH).to('cuda:0') 

video_path = "/mnt/hdd_6tb/YOLO_Object_Detection_Dataset/original_video.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

save_path = "/home/minji/landing_project/vehicle_center/train_results/tracking_result_yolo.mp4"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame, 
        persist=True, 
        tracker="bytetrack.yaml",
        conf=0.15,
        iou=0.5,
        imgsz=1280,
        augment=False,
        half=True,
        verbose=False
    )

    if results[0].boxes.id is not None:
        annotated_frame = results[0].plot()
    else:
        annotated_frame = frame

    out.write(annotated_frame)

print(f"작업 완료! 저장 위치: {save_path}")
cap.release()
out.release()