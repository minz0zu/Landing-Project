from ultralytics import YOLO
import os

def run_inference():
    model_path = "/home/minji/landing_project/train_results/SGD/weights/best.pt"
        
    model = YOLO(model_path)

    test_set = "/mnt/hdd_6tb/minji/processed_quarter/Test/images"

    results = model.predict(
        source=test_set,
        device=3,
        conf=0.25, # 25% 이상인 것만
        imgsz=640, # 사진 크기
        project="/mnt/hdd_6tb/minji/inference_results",
        name="SGD_split",
        exist_ok=True,
        save=True # 폴더 안에 사진 저장
    )

if __name__ == "__main__":
    run_inference()